"""Verify and rebuild Stage 3 runs from their archived inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .manifest import build_run_manifest, canonical_json_bytes, sha256_file
from .schemas import RunManifest


def rebuild_run(run_dir: str | Path, output_dir: str | Path):
    from .stage2 import load_simulation_config
    from .stage3 import Stage3Config, Stage3Harness

    source = Path(run_dir).resolve()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    manifest = RunManifest.from_dict(json.loads((source / "run_manifest.json").read_text(encoding="utf-8")))
    config = load_simulation_config(source / "config.json")
    if not isinstance(config, Stage3Config) or not config.stage3_enabled:
        raise ValueError("rebuild requires an active Stage 3 run with archived inputs")
    if manifest.failures:
        raise ValueError("a failed run cannot be verified as a completed reconstruction")
    if hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest() != manifest.config_sha256:
        raise ValueError("archived configuration hash mismatch")
    if config.seed != manifest.random_seed:
        raise ValueError("archived random seed mismatch")
    archive = json.loads((source / "reproducibility.json").read_text(encoding="utf-8"))
    files = {**archive["data_files"], **archive["source_files"]}
    if set(files) != set(manifest.data_sha256):
        raise ValueError("archived input inventory does not match the manifest")
    paths = {}
    for logical_name, relative_path in files.items():
        path = (source / relative_path).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"missing or invalid archived input: {logical_name}")
        if sha256_file(path) != manifest.data_sha256[logical_name]:
            raise ValueError(f"archived input hash mismatch: {logical_name}")
        paths[logical_name] = path
    for logical_name in archive["source_files"]:
        filename = logical_name.removeprefix("source:")
        current = Path(__file__).parent / filename
        if not current.is_file() or sha256_file(current) != manifest.data_sha256[logical_name]:
            raise ValueError(f"source changed: {filename}; rebuild in a separate checkout of the archived sources")
    identity = build_run_manifest(
        config=config.to_dict(), data_files=paths, random_seed=config.seed,
        code_revision=manifest.code_revision, schema_version=manifest.schema_version,
        agent_models=manifest.agent_models, prompt_sha256=manifest.prompt_sha256,
        graph_version=manifest.graph_version, enabled_plugins=manifest.enabled_plugins,
    )
    if identity.run_id != manifest.run_id:
        raise ValueError("archived manifest identity mismatch")
    overrides = {original: paths[logical] for logical, original in archive["original_paths"].items()}
    harness = Stage3Harness(
        config, data_file_overrides=overrides, code_revision=manifest.code_revision,
    )
    if harness.manifest.run_id != manifest.run_id:
        raise ValueError("reconstructed run identity mismatch")
    journal = output.parent / f".{output.name}.events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x", encoding="utf-8"):
        pass
    harness.event_log = journal
    try:
        result = harness.run()
        expected = json.loads((source / "summary.json").read_text(encoding="utf-8"))["fingerprint"]
        if result.fingerprint() != expected:
            raise ValueError("reconstructed result fingerprint mismatch")
        result.write(output, config, code_revision=manifest.code_revision)
    except Exception as exc:
        record_run_failure(harness, config, output, exc)
        raise
    return result


def record_run_failure(harness, config, output_dir: str | Path, error: Exception) -> None:
    """Keep a terminal failure record without manufacturing a success summary."""
    from .runner import resolve_code_revision

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    original_manifest = getattr(harness, "manifest", None)
    if original_manifest is None:
        original_manifest = build_run_manifest(
            config=config.to_dict(), data_files={}, random_seed=config.seed,
            code_revision=resolve_code_revision(), schema_version=config.schema_version,
        )
    ended_at = datetime.now(timezone.utc)
    message = f"{type(error).__name__}: {error}"
    current_day = getattr(harness, "next_day_index", 0)
    telemetry_events = getattr(harness, "telemetry_events", ())
    manifest = replace(original_manifest, ended_at=ended_at,
                       failures=(*original_manifest.failures, message))
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (destination / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (destination / "failure.json").write_text(
        json.dumps({"state": "failed", "current_day": current_day,
                    "trading_days": config.trading_days,
                    "telemetry_events": len(telemetry_events),
                    "error": message, "ended_at": ended_at.isoformat()},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    journal = getattr(harness, "event_log", None)
    archived_events = destination / "events.jsonl"
    if not archived_events.exists():
        if journal is not None and Path(journal).is_file():
            shutil.copyfile(journal, archived_events)
        else:
            with archived_events.open("w", encoding="utf-8") as stream:
                for event in telemetry_events:
                    stream.write(json.dumps(event.to_dict(), ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False) + "\n")
