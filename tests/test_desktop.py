import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

from abm.config import load_stage1_config
from abm.desktop import (
    MarketMonitorWindow,
    _application,
    live_worker,
)
from abm.harness import run_stage1
from abm.telemetry import VisualizerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_replay_window_renders_all_stage1_metrics(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=40,
        trading_days=60,
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

