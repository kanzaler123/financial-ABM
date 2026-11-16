"""Regression checks for detached observers, durable streaming and failed runs."""

import asyncio
import json
import multiprocessing
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from abm.config import load_stage1_config
from abm.harness import MarketHarness, run_stage1
from abm.reproduction import record_run_failure
from abm.run_service import controlled_run_worker
from abm.telemetry import (
    ObserverQueue,
    TELEMETRY_COLUMNS,
    TelemetryEvent,
    TelemetryJournalCursor,
    finish_observer_queue,
    read_telemetry,
    write_telemetry,
)
from abm.web import ActiveRun, RunRegistry, create_app

ROOT = Path(__file__).resolve().parents[1]
AUTH = {"Authorization": "Bearer audit"}


def short_config(days=4):
    return replace(load_stage1_config(ROOT / "configs/stage1.json"),
                   population_size=10, trading_days=days, burn_in_days=0,
                   announcements=())


def event(sequence, text="完整的 UTF-8 记录"):
    payload = dict.fromkeys(TELEMETRY_COLUMNS, 0.0)
    payload.update(price=100.0, label=text)
    return TelemetryEvent("audit", sequence, sequence, "daily", payload)


def encoded(item):
    return (json.dumps(item.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")


def client_for(tmp_path):
    app = create_app(runs_root=tmp_path / "runs", configs_root=ROOT / "configs",
                     reports_root=tmp_path / "reports", web_root=tmp_path / "web",
                     auth_token="audit")
    return app, TestClient(app)


@pytest.mark.parametrize("guarded", [False, True])
def test_completed_worker_exits_with_large_unread_observer_queues(tmp_path, guarded):
    """Both pipes exceed their buffer; no observer drains them, even at exit."""
    config = short_config(500)
    expected = run_stage1(config).fingerprint()
    context = multiprocessing.get_context("spawn")
    factory = (lambda size: ObserverQueue.create(context, size)) if guarded else (
        lambda size: context.Queue(maxsize=size))
    telemetry = factory(4096)
    statuses = factory(1024)
    commands = context.Queue(maxsize=8)
    completion = context.Queue(maxsize=4)
    commands.put({"action": "detach"})
    output = tmp_path / "run"
    process = context.Process(target=controlled_run_worker,
                              args=(config.to_dict(), str(output), telemetry,
                                    commands, completion, statuses))
    process.start()
    try:
        # Receiving completion alone must not be necessary for worker exit.
        process.join(timeout=30)
        assert not process.is_alive(), "observer feeder kept the completed worker alive"
        assert process.exitcode == 0
        status = completion.get(timeout=2)
        assert status["state"] == "completed"
        assert status["published_frames"] == 500
        assert status["fingerprint"] == expected
        assert len(read_telemetry(output)) == 500
        if guarded:
            with pytest.raises(queue.Empty):
                telemetry.get_nowait()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for channel in (telemetry, statuses, commands, completion):
            if hasattr(channel, "cancel_join_thread"):
                channel.cancel_join_thread()
            channel.close()
        process.close()


def _large_frame_producer(observer, first_received, completion):
    for index in range(32):
        observer.put_nowait({"index": index, "payload": "x" * (128 * 1024)})
    first_received.wait(timeout=5)
    completion.put("finished")
    finish_observer_queue(observer)


def test_observer_shutdown_does_not_leave_reader_in_a_partial_frame():
    context = multiprocessing.get_context("spawn")
    observer = ObserverQueue.create(context, 64)
    first_received = context.Event()
    completion = context.Queue(maxsize=1)
    received = []

    def observe():
        while not observer.finished.is_set():
            try:
                received.append(observer.get_nowait())
                first_received.set()
            except queue.Empty:
                time.sleep(0.001)

    process = context.Process(target=_large_frame_producer,
                              args=(observer, first_received, completion))
    reader = threading.Thread(target=observe, daemon=True)
    reader.start()
    process.start()
    try:
        assert completion.get(timeout=10) == "finished"
        process.join(timeout=10)
        reader.join(timeout=5)
        assert not process.is_alive()
        assert not reader.is_alive(), "reader waited for a discarded partial frame"
        assert process.exitcode == 0
        assert received
        assert all(len(frame["payload"]) == 128 * 1024 for frame in received)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        observer.close()
        completion.close()
        process.close()


def test_incremental_journal_reads_do_not_parse_old_records_again(tmp_path, monkeypatch):
    run = tmp_path / "run"
    journal = tmp_path / ".run.events.jsonl"
    journal.write_bytes(b"".join(encoded(event(index)) for index in range(1, 101)))
    cursor = TelemetryJournalCursor()
    assert len(read_telemetry(run, journal_cursor=cursor)) == 100
    original_loads = json.loads
    decoded = []

    def counting_loads(value, *args, **kwargs):
        decoded.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr("abm.telemetry.json.loads", counting_loads)
    with journal.open("ab") as stream:
        stream.write(encoded(event(101)))
    assert read_telemetry(run, after_sequence=100, journal_cursor=cursor) == (event(101),)
    assert len(decoded) == 1
    assert read_telemetry(run, after_sequence=101, journal_cursor=cursor) == ()
    assert len(decoded) == 1
    # A replay or a fresh reconnect must remain lossless.
    assert read_telemetry(run, after_sequence=99, journal_cursor=cursor) == (event(100), event(101))
    assert read_telemetry(run, after_sequence=100,
                          journal_cursor=TelemetryJournalCursor()) == (event(101),)


def test_partial_utf8_line_is_retried_only_after_its_newline(tmp_path):
    run = tmp_path / "run"
    journal = tmp_path / ".run.events.jsonl"
    second = encoded(event(2))
    split = second.index("完".encode("utf-8")) + 1
    journal.write_bytes(encoded(event(1)) + second[:split])
    cursor = TelemetryJournalCursor()
    assert read_telemetry(run, journal_cursor=cursor) == (event(1),)
    assert read_telemetry(run, after_sequence=1, journal_cursor=cursor) == ()
    with journal.open("ab") as stream:
        stream.write(second[split:-1])
    assert read_telemetry(run, after_sequence=1, journal_cursor=cursor) == ()
    with journal.open("ab") as stream:
        stream.write(b"\n")
    assert read_telemetry(run, after_sequence=1, journal_cursor=cursor) == (event(2),)


def test_cursor_survives_pagination_rename_and_archive_transition(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    journal = tmp_path / ".run.events.jsonl"
    items = tuple(event(index) for index in range(1, 5))
    journal.write_bytes(b"".join(map(encoded, items)))
    cursor = TelemetryJournalCursor()
    assert read_telemetry(run, limit=2, journal_cursor=cursor) == items[:2]
    journal.rename(run / "events.jsonl")
    assert read_telemetry(run, after_sequence=2, limit=1, journal_cursor=cursor) == items[2:3]
    write_telemetry(items, run)
    assert read_telemetry(run, after_sequence=3, journal_cursor=cursor) == items[3:]
    (run / "telemetry.parquet").unlink()
    assert read_telemetry(run, after_sequence=3, journal_cursor=cursor) == items[3:]


def test_cursor_restarts_after_journal_truncation_or_replacement(tmp_path):
    run = tmp_path / "run"
    journal = tmp_path / ".run.events.jsonl"
    journal.write_bytes(encoded(event(1, "x" * 2000)))
    cursor = TelemetryJournalCursor()
    read_telemetry(run, journal_cursor=cursor)
    journal.write_bytes(encoded(event(2)))
    assert read_telemetry(run, after_sequence=1, journal_cursor=cursor) == (event(2),)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(encoded(event(3)))
    replacement.replace(journal)
    assert read_telemetry(run, after_sequence=2, journal_cursor=cursor) == (event(3),)


def test_failed_run_without_success_summary_stays_visible_and_downloadable(tmp_path):
    config = short_config()
    harness = MarketHarness(config)
    harness.step(0)
    run = tmp_path / "runs" / "failed"
    record_run_failure(harness, config, run, RuntimeError("audit interruption"))
    assert not (run / "summary.json").exists()
    record, = RunRegistry(tmp_path / "runs").list_runs()
    assert record["state"] == "failed"
    assert record["current_day"] == 1
    assert record["trading_days"] == 4
    assert record["fingerprint"] is None
    assert read_telemetry(run) == tuple(harness.telemetry_events)
    app, client = client_for(tmp_path)
    assert client.get("/api/runs/failed", headers=AUTH).json()["state"] == "failed"
    app.state.registry.active["failed"] = SimpleNamespace(
        state="failed", refresh=lambda: None,
        process=SimpleNamespace(join=lambda **kwargs: None),
    )
    for artifact in ("manifest", "failure", "config", "events"):
        assert client.get(f"/api/runs/failed/artifacts/{artifact}", headers=AUTH).status_code == 200
    app.state.registry.active["failed"].state = "running"
    assert client.get("/api/runs/failed/artifacts/manifest", headers=AUTH).status_code == 409


def test_initialization_failure_with_valid_config_is_archived(tmp_path, monkeypatch):
    def fail_initialization(*args, **kwargs):
        raise OSError("initial input unavailable")

    monkeypatch.setattr("abm.run_service.create_market_harness", fail_initialization)
    config = short_config()
    run = tmp_path / "runs" / "initialization-failed"
    completion = queue.Queue()
    controlled_run_worker(config.to_dict(), str(run), queue.Queue(), queue.Queue(), completion)
    status = completion.get_nowait()
    assert status["state"] == "failed"
    assert status["current_day"] == 0
    assert status["trading_days"] == config.trading_days
    assert json.loads((run / "config.json").read_text()) == config.to_dict()
    assert (run / "events.jsonl").read_bytes() == b""
    assert not (run / "summary.json").exists()
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["failures"] == ["OSError: initial input unavailable"]
    record, = RunRegistry(tmp_path / "runs").list_runs()
    assert record["state"] == "failed"
    assert record["current_day"] == 0
    assert record["trading_days"] == config.trading_days


def test_existing_output_is_never_rewritten_by_failure_archiving(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "config.json").write_bytes(b'{"preserve":"original config"}\n')
    (output / "run_manifest.json").write_bytes(b'{"preserve":"original evidence"}\n')
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    completion = queue.Queue()
    controlled_run_worker(short_config().to_dict(), str(output), queue.Queue(), queue.Queue(), completion)
    status = completion.get_nowait()
    assert status["state"] == "failed"
    assert "FileExistsError" in status["error"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_invalid_configuration_reports_failure_without_creating_a_run(tmp_path):
    payload = short_config().to_dict()
    payload["trading_days"] = "not an integer"
    output = tmp_path / "invalid"
    completion = queue.Queue()
    controlled_run_worker(payload, str(output), queue.Queue(), queue.Queue(), completion)
    status = completion.get_nowait()
    assert status["state"] == "failed"
    assert status["current_day"] == 0
    assert status["trading_days"] == 0
    assert "trading_days" in status["error"]
    assert not output.exists()


@pytest.mark.parametrize("manifest", [[], None, {"failures": {}}, {"failures": [1]}])
def test_one_invalid_manifest_cannot_break_the_run_list(tmp_path, manifest):
    for name in ("good", "bad", "incomplete"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "config.json").write_text('{"trading_days":4}', encoding="utf-8")
        if name != "incomplete":
            (run / "summary.json").write_text('{"trading_days":4}', encoding="utf-8")
        (run / "run_manifest.json").write_text(
            json.dumps(manifest if name == "bad" else {"failures": []}), encoding="utf-8")
    _, client = client_for(tmp_path)
    response = client.get("/api/runs", headers=AUTH)
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == ["good"]
    assert client.get("/api/runs/incomplete", headers=AUTH).status_code == 404


def test_overflowing_numeric_metadata_is_isolated_from_valid_runs(tmp_path):
    for name, config in (("good", '{"seed":1}'), ("bad", '{"seed":1e999}')):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "config.json").write_text(config, encoding="utf-8")
        (run / "summary.json").write_text('{"trading_days":4}', encoding="utf-8")
    _, client = client_for(tmp_path)
    response = client.get("/api/runs", headers=AUTH)
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == ["good"]


def test_sse_io_runs_off_event_loop_and_drains_after_completion(tmp_path, monkeypatch):
    app, client = client_for(tmp_path)
    app.state.registry.active["run"] = ActiveRun(
        name="run", config=short_config(), output_dir=tmp_path / "runs/run",
        process=SimpleNamespace(is_alive=lambda: False, exitcode=0),
        telemetry_queue=queue.Queue(), command_queue=queue.Queue(),
        completion_queue=queue.Queue(), status_queue=queue.Queue(), state="completed",
    )
    calls = []

    def finishing_archive(*args, **kwargs):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        calls.append(kwargs["after_sequence"])
        # The worker finishes after the first empty poll, making its final
        # durable event available just before the completion state is checked.
        return (event(1),) if len(calls) == 2 else ()

    monkeypatch.setattr("abm.web.read_telemetry", finishing_archive)
    response = client.get("/api/runs/run/stream", headers=AUTH)
    assert response.status_code == 200
    assert calls == [-1, -1, 1]
    assert response.text.index("event: telemetry") < response.text.index("event: complete")


def test_sse_preserves_failed_state_and_validates_before_streaming(tmp_path):
    run = tmp_path / "runs/failed"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "run_manifest.json").write_text('{"failures":["interrupted"]}', encoding="utf-8")
    (run / "events.jsonl").write_bytes(encoded(event(1)) + encoded(event(2)))
    _, client = client_for(tmp_path)
    response = client.get("/api/runs/failed/stream", headers={**AUTH, "Last-Event-ID": "1"})
    assert response.status_code == 200
    assert '"sequence_no":1' not in response.text
    assert '"sequence_no":2' in response.text
    assert 'event: complete\ndata: {"state":"failed"}' in response.text
    assert client.get("/api/runs/missing/stream", headers=AUTH).status_code == 404
    assert client.get("/api/runs/failed/stream", headers={**AUTH, "Last-Event-ID": "-2"}).status_code == 422


def test_sse_rejects_incomplete_or_missing_store_before_sending_headers(tmp_path):
    for name in ("incomplete", "missing-store"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")
        if name == "missing-store":
            (run / "summary.json").write_text("{}", encoding="utf-8")
    _, client = client_for(tmp_path)
    for name in ("incomplete", "missing-store"):
        assert client.get(f"/api/runs/{name}/stream", headers=AUTH).status_code == 404


def test_desktop_completion_replaces_lossy_preview_with_complete_archive(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("vispy")
    from abm.desktop import MarketMonitorWindow, _application
    from abm.telemetry import VisualizerConfig

    application = _application()
    result = run_stage1(short_config())
    write_telemetry(result.telemetry_events, tmp_path)
    telemetry = queue.Queue()
    telemetry.put(result.telemetry_events[0].to_dict())
    completion = queue.Queue()
    completion.put({"fingerprint": result.fingerprint()})
    window = MarketMonitorWindow(
        mode="live", visualizer_config=VisualizerConfig("live"),
        telemetry_queue=telemetry, completion_queue=completion, live_run_dir=tmp_path,
    )
    try:
        window._on_timer()
        assert window.model.events == list(result.telemetry_events)
        assert "complete" in window.status_label.text()
    finally:
        window.close()
        application.processEvents()
