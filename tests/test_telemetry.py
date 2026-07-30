import queue
import time
from dataclasses import replace
from pathlib import Path

from abm.config import load_stage1_config
from abm.desktop import ReplayController, TelemetryModel
from abm.harness import run_stage1
from abm.telemetry import (
    NonBlockingQueueSink,
    TelemetryEvent,
    read_telemetry,
    write_telemetry,
)

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


def test_telemetry_round_trip_through_duckdb_and_parquet(tmp_path) -> None:
    expected = tuple(_event(index) for index in range(1, 6))

    database_path, parquet_path = write_telemetry(expected, tmp_path)
    actual = read_telemetry(tmp_path)

    assert database_path.is_file()
    assert parquet_path.is_file()
    assert actual == expected


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

