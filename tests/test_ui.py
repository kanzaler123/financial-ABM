"""Interface tests: desktop monitor, visualization, telemetry, web lab."""
import json
import multiprocessing
import queue
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from abm.config import load_stage1_config
from abm.desktop import (
    MarketMonitorWindow,
    ReplayController,
    TelemetryModel,
    _application,
    live_worker,
)
from abm.harness import run_stage1
from abm.telemetry import (
    NonBlockingQueueSink,
    TelemetryEvent,
    VisualizerConfig,
    read_telemetry,
    write_telemetry,
)
from abm.visualization import create_stage1_visualizations
from abm.web import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def _event(sequence_no: int) -> TelemetryEvent:
    return TelemetryEvent(
        run_id="run",
        sim_time=sequence_no,
        sequence_no=sequence_no,
        event_type="daily_market",
        payload={
            "price": 100.0 + sequence_no,
            "return": 0.01,
            "volatility_20d": 0.02,
            "fundamental_value": 100.0,
            "volume": 10.0,
            "value_share": 0.4,
            "trend_share": 0.4,
            "noise_share": 0.2,
            "cash_relative_error": 0.0,
            "share_relative_error": 0.0,
        },
    )


def _client(tmp_path):
    app = create_app(
        runs_root=tmp_path / "runs",
        configs_root=PROJECT_ROOT / "configs",
        reports_root=PROJECT_ROOT / "reports",
        web_root=tmp_path / "missing-web",
        auth_token="test-token",
    )
    return TestClient(app), {"Authorization": "Bearer test-token"}


# --- Desktop monitor ------------------------------------------------------


def test_replay_window_renders_all_stage1_metrics(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=40,
        trading_days=60,
        burn_in_days=0,
        announcements=(),
    )
    events = run_stage1(config).telemetry_events
    application = _application()
    window = MarketMonitorWindow(
        mode="replay",
        visualizer_config=VisualizerConfig(mode="replay"),
        replay_events=events,
    )
    window.resize(1200, 800)
    window.show()
    window.ingest(events)
    application.processEvents()
    screenshot = tmp_path / "desktop.png"

    assert window.grab().save(str(screenshot))
    assert screenshot.stat().st_size > 20_000
    assert len(window.price_curve.xData) == 60
    assert len(window.return_curve.yData) == 60
    assert len(window.volatility_curve.yData) == 60
    assert len(window.volume_bars.opts["height"]) == 60
    assert len(window.value_curve.yData) == 60
    assert len(window.cash_error_curve.yData) == 60
    window.close()


def test_spawned_live_worker_survives_detached_full_display_queue(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=40,
        trading_days=10,
        burn_in_days=0,
        announcements=(),
    )
    expected = run_stage1(config)
    output = tmp_path / "live-run"
    context = multiprocessing.get_context("spawn")
    telemetry_queue = context.Queue(maxsize=1)
    command_queue = context.Queue(maxsize=8)
    completion_queue = context.Queue(maxsize=2)
    command_queue.put({"action": "detach", "value": None})
    process = context.Process(
        target=live_worker,
        args=(
            config.to_dict(),
            str(output),
            telemetry_queue,
            command_queue,
            completion_queue,
        ),
    )

    process.start()
    process.join(timeout=30)

    assert not process.is_alive()
    assert process.exitcode == 0
    completion = completion_queue.get(timeout=2)
    assert "error" not in completion
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["fingerprint"] == expected.fingerprint()
    assert summary["telemetry_events"] == 10
    assert (output / "telemetry.parquet").is_file()
    assert (output / "telemetry.duckdb").is_file()


# --- Static visualization -------------------------------------------------


def test_stage1_visualizations_are_exported_as_png_and_svg(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=30,
        trading_days=60,
        burn_in_days=0,
        announcements=(),
    )
    run_dir = tmp_path / "run"
    run_stage1(config).write(run_dir, config)

    expected = (
        run_dir / "stage1_overview.png",
        run_dir / "stage1_overview.svg",
        run_dir / "stage1_microstructure.png",
        run_dir / "stage1_microstructure.svg",
    )
    assert all(path.is_file() and path.stat().st_size > 1000 for path in expected)
    assert expected[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in expected[1].read_text(encoding="utf-8")[:1000]

    regenerated = create_stage1_visualizations(run_dir)
    assert regenerated == expected


# --- Telemetry ------------------------------------------------------------


def test_telemetry_round_trip_through_duckdb_and_parquet(tmp_path) -> None:
    expected = tuple(_event(index) for index in range(1, 6))

    database_path, parquet_path = write_telemetry(expected, tmp_path)
    actual = read_telemetry(tmp_path)

    assert database_path.is_file()
    assert parquet_path.is_file()
    assert actual == expected
    assert read_telemetry(tmp_path, after_sequence=2, limit=2) == expected[2:4]


def test_display_queue_full_drops_frames_without_blocking() -> None:
    display_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
    sink = NonBlockingQueueSink(display_queue)

    started = time.perf_counter()
    assert sink.publish(_event(1))
    assert not sink.publish(_event(2))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert sink.published_count == 1
    assert sink.dropped_count == 1


def test_visualizer_sink_does_not_change_simulation_hash() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=100,
        trading_days=80,
        burn_in_days=0,
        announcements=(),
    )
    display_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
    with_visualizer = run_stage1(
        config, telemetry_sink=NonBlockingQueueSink(display_queue)
    )
    without_visualizer = run_stage1(config)

    assert with_visualizer.fingerprint() == without_visualizer.fingerprint()
    assert with_visualizer.telemetry_events == without_visualizer.telemetry_events


def test_live_and_replay_models_have_identical_daily_metrics(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=100,
        trading_days=60,
        burn_in_days=0,
        announcements=(),
    )
    result = run_stage1(config)
    write_telemetry(result.telemetry_events, tmp_path)

    live_model = TelemetryModel()
    live_model.extend(result.telemetry_events)
    replay_model = TelemetryModel()
    replay_model.extend(read_telemetry(tmp_path))

    assert live_model.signature() == replay_model.signature()


def test_replay_controller_supports_pause_resume_step_and_speed() -> None:
    controller = ReplayController(tuple(_event(index) for index in range(1, 6)))

    assert controller.advance() == ()
    assert controller.step_once()[0].sequence_no == 1
    controller.speed = 2.0
    controller.resume()
    assert [event.sequence_no for event in controller.advance()] == [2, 3]
    controller.pause()
    assert controller.advance() == ()
    controller.reset()
    assert controller.cursor == 0
    assert controller.paused


# --- Web lab --------------------------------------------------------------


def test_health_and_api_authentication(tmp_path) -> None:
    client, headers = _client(tmp_path)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers=headers).status_code == 200


def test_config_discovery_excludes_non_stage1_protocols(tmp_path) -> None:
    client, headers = _client(tmp_path)

    response = client.get("/api/configs", headers=headers)

    assert response.status_code == 200
    assert "stage1" in {item["id"] for item in response.json()}
    assert "stage1_mechanism_dev_v8" not in {
        item["id"] for item in response.json()
    }


def test_run_path_traversal_and_unknown_artifact_are_rejected(tmp_path) -> None:
    client, headers = _client(tmp_path)

    traversal = client.post(
        "/api/runs",
        headers=headers,
        json={"name": "../escape", "config_name": "stage1"},
    )
    unknown = client.get(
        "/api/runs/not-found/artifacts/../../config",
        headers=headers,
    )

    assert traversal.status_code == 422
    assert unknown.status_code in (404, 422)


def test_reports_distinguish_superseded_and_formal_status(tmp_path) -> None:
    reports_root = tmp_path / "reports"
    legacy = reports_root / "legacy"
    formal = reports_root / "formal"
    legacy.mkdir(parents=True)
    formal.mkdir()
    (legacy / "mechanism_report.json").write_text(
        '{"protocol":{"version":"stage1-mechanism-acceptance-v1"},'
        '"gate":{"passed":true,"passed_checks":16,"total_checks":16}}',
        encoding="utf-8",
    )
    (formal / "mechanism_report.json").write_text(
        '{"protocol":{"version":"stage1-mechanism-acceptance-v6",'
        '"stage":"formal"},"freeze_verification":{"verified":true},'
        '"gate":{"passed":true,"passed_checks":20,"total_checks":20}}',
        encoding="utf-8",
    )
    (formal / "final_decision.json").write_text(
        '{"stage1_complete":true}', encoding="utf-8"
    )
    app = create_app(
        runs_root=tmp_path / "runs",
        configs_root=PROJECT_ROOT / "configs",
        reports_root=reports_root,
        web_root=tmp_path / "missing-web",
        auth_token="test-token",
    )
    client = TestClient(app)

    response = client.get(
        "/api/reports", headers={"Authorization": "Bearer test-token"}
    )

    assert response.status_code == 200
    records = {record["id"]: record for record in response.json()}
    assert records["legacy"]["scientific_status"] == "superseded"
    assert not records["legacy"]["stage1_complete"]
    assert records["formal"]["scientific_status"] == "formal_pass"
    assert records["formal"]["stage1_complete"]


def test_small_web_run_completes_and_replays_events(tmp_path) -> None:
    client, headers = _client(tmp_path)
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=30,
        trading_days=4,
        burn_in_days=0,
        announcements=(),
    )

    started = client.post(
        "/api/runs",
        headers=headers,
        json={"name": "web-smoke", "config": config.to_dict()},
    )
    assert started.status_code == 202

    detail = None
    for _ in range(200):
        detail = client.get("/api/runs/web-smoke", headers=headers).json()
        if detail["state"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert detail is not None
    assert detail["state"] == "completed"
    events = client.get(
        "/api/runs/web-smoke/events?after=1&limit=2",
        headers=headers,
    )
    assert events.status_code == 200
    assert [event["sequence_no"] for event in events.json()] == [2, 3]
    artifact = client.get(
        "/api/runs/web-smoke/artifacts/summary",
        headers=headers,
    )
    assert artifact.status_code == 200
