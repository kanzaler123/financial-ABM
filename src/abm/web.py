"""Loopback-only FastAPI service for Stage 1 live runs and replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import queue
import re
import secrets
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Stage1Config, load_stage1_config
from .run_service import controlled_run_worker
from .telemetry import TelemetryEvent, read_telemetry

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COMMANDS = {"pause", "resume", "step", "speed"}
ARTIFACTS = {
    "config": "config.json",
    "summary": "summary.json",
    "manifest": "run_manifest.json",
    "daily-audit": "daily_audit.jsonl",
    "learning-audit": "learning_audit.json",
    "state-arrays": "state_arrays.npz",
    "telemetry-parquet": "telemetry.parquet",
    "telemetry-duckdb": "telemetry.duckdb",
    "overview-png": "stage1_overview.png",
    "overview-svg": "stage1_overview.svg",
    "microstructure-png": "stage1_microstructure.png",
    "microstructure-svg": "stage1_microstructure.svg",
}


@dataclass(slots=True)
class ActiveRun:
    name: str
    config: Stage1Config
    output_dir: Path
    process: Any
    telemetry_queue: Any
    command_queue: Any
    completion_queue: Any
    status_queue: Any
    events: list[TelemetryEvent] = field(default_factory=list)
    state: str = "starting"
    current_day: int = 0
    fingerprint: str | None = None
    published_frames: int = 0
    dropped_frames: int = 0
    error: str | None = None

    def refresh(self) -> None:
        while True:
            try:
                self.events.append(
                    TelemetryEvent.from_dict(self.telemetry_queue.get_nowait())
                )
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError):
                break
        while True:
            try:
                update = self.status_queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            self._apply(update)
        try:
            completion = self.completion_queue.get_nowait()
        except queue.Empty:
            completion = None
        except (EOFError, OSError, ValueError):
            completion = None
        if completion:
            self._apply(completion)

    def _apply(self, update: Mapping[str, Any]) -> None:
        self.state = str(update.get("state", self.state))
        self.current_day = int(update.get("current_day", self.current_day))
        self.fingerprint = update.get("fingerprint") or self.fingerprint
        self.published_frames = int(
            update.get("published_frames", self.published_frames)
        )
        self.dropped_frames = int(
            update.get("dropped_frames", self.dropped_frames)
        )
        self.error = update.get("error") or self.error

    def metadata(self) -> dict[str, object]:
        self.refresh()
        return {
            "id": self.name,
            "state": self.state,
            "active": self.process.is_alive(),
            "current_day": self.current_day,
            "trading_days": self.config.trading_days,
            "seed": self.config.seed,
            "population_size": self.config.population_size,
            "latest_sequence_no": (
                self.events[-1].sequence_no if self.events else 0
            ),
            "fingerprint": self.fingerprint,
            "published_frames": self.published_frames,
            "dropped_frames": self.dropped_frames,
            "error": self.error,
        }


class RunRegistry:
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root.resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, ActiveRun] = {}

    def path_for(self, name: str) -> Path:
        if not SAFE_NAME.fullmatch(name):
            raise ValueError("run name must use 1-64 safe ASCII characters")
        candidate = (self.runs_root / name).resolve()
        if candidate.parent != self.runs_root:
            raise ValueError("run path escapes the runs root")
        return candidate

    def list_runs(self) -> list[dict[str, object]]:
        records = {name: run.metadata() for name, run in self.active.items()}
        for child in self.runs_root.iterdir():
            if not child.is_dir() or child.name in records:
                continue
            summary_path = child / "summary.json"
            config_path = child / "config.json"
            if not summary_path.is_file() or not config_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records[child.name] = {
                "id": child.name,
                "state": "completed",
                "active": False,
                "current_day": int(summary.get("trading_days", 0)),
                "trading_days": int(summary.get("trading_days", 0)),
                "seed": int(config.get("seed", 0)),
                "population_size": int(config.get("population_size", 0)),
                "latest_sequence_no": int(summary.get("telemetry_events", 0)),
                "fingerprint": summary.get("fingerprint"),
                "published_frames": int(summary.get("telemetry_events", 0)),
                "dropped_frames": 0,
                "error": None,
            }
        return sorted(records.values(), key=lambda item: str(item["id"]), reverse=True)

    def start(self, name: str, config: Stage1Config) -> ActiveRun:
        output_dir = self.path_for(name)
        if output_dir.exists() or name in self.active:
            raise FileExistsError(name)
        context = multiprocessing.get_context("spawn")
        telemetry_queue = context.Queue(maxsize=4096)
        command_queue = context.Queue(maxsize=64)
        completion_queue = context.Queue(maxsize=4)
        status_queue = context.Queue(maxsize=max(config.trading_days + 8, 64))
        process = context.Process(
            target=controlled_run_worker,
            args=(
                config.to_dict(),
                str(output_dir),
                telemetry_queue,
                command_queue,
                completion_queue,
                status_queue,
            ),
            name=f"abm-web-{name}",
        )
        active = ActiveRun(
            name=name,
            config=config,
            output_dir=output_dir,
            process=process,
            telemetry_queue=telemetry_queue,
            command_queue=command_queue,
            completion_queue=completion_queue,
            status_queue=status_queue,
        )
        self.active[name] = active
        process.start()
        return active

    def get(self, name: str) -> ActiveRun | None:
        self.path_for(name)
        return self.active.get(name)


def create_app(
    *,
    runs_root: str | Path = "runs",
    configs_root: str | Path = "configs",
    reports_root: str | Path = "reports",
    web_root: str | Path = "web/dist",
    auth_token: str | None = None,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    registry = RunRegistry(Path(runs_root))
    config_root = Path(configs_root).resolve()
    report_root = Path(reports_root).resolve()
    static_root = Path(web_root).resolve()
    token = auth_token or secrets.token_urlsafe(32)
    app.state.registry = registry
    app.state.auth_token = token

    def authorize(authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        candidate = authorization.removeprefix("Bearer ")
        if not secrets.compare_digest(candidate, token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid bearer token")

    def json_file(root: Path, name: str) -> Path:
        if not SAFE_NAME.fullmatch(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        path = (root / f"{name}.json").resolve()
        if path.parent != root or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return path

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/session")
    def session(request: Request) -> dict[str, str]:
        if request.client is None or request.client.host not in (
            "127.0.0.1",
            "::1",
            "testclient",
        ):
            raise HTTPException(status.HTTP_403_FORBIDDEN)
        return {"token": token}

    @app.get("/api/configs")
    def configs(authorization: str | None = Header(default=None)) -> list[dict[str, object]]:
        authorize(authorization)
        records = []
        for path in sorted(config_root.glob("*.json")):
            try:
                config = load_stage1_config(path)
            except (TypeError, ValueError, KeyError):
                continue
            records.append(
                {
                    "id": path.stem,
                    "population_size": config.population_size,
                    "trading_days": config.trading_days,
                    "seed": config.seed,
                }
            )
        return records

    @app.get("/api/configs/{name}")
    def config(name: str, authorization: str | None = Header(default=None)) -> object:
        authorize(authorization)
        path = json_file(config_root, name)
        try:
            return load_stage1_config(path).to_dict()
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    @app.get("/api/runs")
    def runs(authorization: str | None = Header(default=None)) -> list[dict[str, object]]:
        authorize(authorization)
        return registry.list_runs()

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authorize(authorization)
        body = await request.body()
        if len(body) > 256 * 1024:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        try:
            payload = json.loads(body)
            name = str(payload["name"])
            if "config" in payload:
                config = Stage1Config.from_dict(payload["config"])
            else:
                config = load_stage1_config(
                    json_file(config_root, str(payload["config_name"]))
                )
            if config.population_size > 100_000 or config.trading_days > 100_000:
                raise ValueError("requested run exceeds local resource limits")
            active = registry.start(name, config)
        except FileExistsError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "run already exists") from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        return active.metadata()

    @app.get("/api/runs/{name}")
    def run_detail(name: str, authorization: str | None = Header(default=None)) -> dict[str, object]:
        authorize(authorization)
        try:
            active = registry.get(name)
            if active is not None:
                return active.metadata()
            path = registry.path_for(name)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND) from exc
        summary = path / "summary.json"
        config_path = path / "config.json"
        if not summary.is_file() or not config_path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return next(item for item in registry.list_runs() if item["id"] == name)

    @app.post("/api/runs/{name}/commands", status_code=status.HTTP_202_ACCEPTED)
    async def command_run(
        name: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        authorize(authorization)
        active = registry.get(name)
        if active is None or not active.process.is_alive():
            raise HTTPException(status.HTTP_409_CONFLICT, "run is not active")
        payload = await request.json()
        action = str(payload.get("action", ""))
        if action not in COMMANDS:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown command")
        active.command_queue.put_nowait(
            {"action": action, "value": payload.get("value")}
        )
        return {"status": "accepted"}

    def events_for(name: str, after: int, limit: int | None) -> tuple[TelemetryEvent, ...]:
        active = registry.get(name)
        if active is not None:
            active.refresh()
            events = tuple(event for event in active.events if event.sequence_no > after)
            return events if limit is None else events[:limit]
        path = registry.path_for(name)
        try:
            return read_telemetry(path, after_sequence=after, limit=limit)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND) from exc

    @app.get("/api/runs/{name}/events")
    def events(
        name: str,
        after: int = Query(default=-1, ge=-1),
        limit: int = Query(default=1000, ge=1, le=10_000),
        authorization: str | None = Header(default=None),
    ) -> list[dict[str, object]]:
        authorize(authorization)
        try:
            return [event.to_dict() for event in events_for(name, after, limit)]
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND) from exc

    @app.get("/api/runs/{name}/stream")
    async def stream(
        name: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        authorize(authorization)
        try:
            cursor = int(last_event_id) if last_event_id is not None else -1
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid Last-Event-ID") from exc

        async def generate():
            nonlocal cursor
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                batch = events_for(name, cursor, 1000)
                for event in batch:
                    cursor = event.sequence_no
                    data = json.dumps(
                        event.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    yield f"id: {cursor}\nevent: telemetry\ndata: {data}\n\n"
                active = registry.get(name)
                if active is None or not active.process.is_alive():
                    state = "completed" if active is None else active.metadata()["state"]
                    yield f"event: complete\ndata: {{\"state\":\"{state}\"}}\n\n"
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/reports")
    def reports(authorization: str | None = Header(default=None)) -> list[dict[str, object]]:
        authorize(authorization)
        records = []
        if not report_root.is_dir():
            return records
        for path in sorted(report_root.glob("*/mechanism_report.json"), reverse=True):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            gate = report.get("gate", {})
            protocol = report.get("protocol", {})
            protocol_version = protocol.get("version")
            protocol_stage = protocol.get("stage", "development")
            decision_path = path.parent / "final_decision.json"
            decision: dict[str, object] = {}
            if decision_path.is_file():
                try:
                    loaded_decision = json.loads(
                        decision_path.read_text(encoding="utf-8")
                    )
                    if isinstance(loaded_decision, dict):
                        decision = loaded_decision
                except (OSError, json.JSONDecodeError):
                    decision = {}
            freeze_verification = report.get("freeze_verification") or {}
            if not isinstance(freeze_verification, dict):
                freeze_verification = {}
            superseded = bool(
                protocol_stage == "development"
                and isinstance(protocol_version, str)
                and "acceptance-v1" in protocol_version
            )
            formal_pass = bool(
                protocol_stage == "formal"
                and decision.get("stage1_complete") is True
                and freeze_verification.get("verified") is True
            )
            if superseded:
                scientific_status = "superseded"
            elif formal_pass:
                scientific_status = "formal_pass"
            elif protocol_stage == "formal":
                scientific_status = "formal_fail"
            elif gate.get("passed"):
                scientific_status = "development_pass"
            else:
                scientific_status = "failed"
            records.append(
                {
                    "id": path.parent.name,
                    "passed": gate.get("passed"),
                    "passed_checks": gate.get("passed_checks"),
                    "total_checks": gate.get("total_checks"),
                    "protocol": protocol_version,
                    "protocol_stage": protocol_stage,
                    "scientific_status": scientific_status,
                    "stage1_complete": formal_pass,
                    "freeze_verified": bool(
                        freeze_verification.get("verified")
                    ),
                    "superseded": superseded,
                }
            )
        return records

    @app.get("/api/reports/{name}")
    def report(name: str, authorization: str | None = Header(default=None)) -> object:
        authorize(authorization)
        if not SAFE_NAME.fullmatch(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        path = (report_root / name / "mechanism_report.json").resolve()
        if path.parent.parent != report_root or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/runs/{name}/artifacts/{artifact_key}")
    def artifact(
        name: str,
        artifact_key: str,
        authorization: str | None = Header(default=None),
    ) -> FileResponse:
        authorize(authorization)
        filename = ARTIFACTS.get(artifact_key)
        if filename is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        active = registry.get(name)
        if active is not None:
            active.refresh()
            if active.state != "completed":
                raise HTTPException(status.HTTP_409_CONFLICT, "run is still active")
            active.process.join(timeout=1.0)
        path = registry.path_for(name) / filename
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return FileResponse(path, filename=filename)

    if static_root.is_dir():
        app.mount("/assets", StaticFiles(directory=static_root / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            candidate = (static_root / path).resolve()
            if candidate.is_file() and static_root in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_root / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                {"message": "Frontend is not built. Run npm ci && npm run build in web/."},
                status_code=503,
            )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--open", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{args.port}"
    print(f"ABM Web: {url}")
    print(f"Session token: {token}")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(
        create_app(runs_root=args.runs_root, auth_token=token),
        host="127.0.0.1",
        port=args.port,
        proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
