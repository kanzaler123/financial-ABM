"""PySide6 desktop monitor for batch, live, and replay operation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from vispy import scene

from .config import Stage1Config, load_stage1_config
from .harness import MarketHarness, run_stage1
from .telemetry import (
    NonBlockingQueueSink,
    TelemetryEvent,
    VisualizerConfig,
    read_telemetry,
)

INK = "#1F2937"
MUTED = "#6B7280"
BLUE = "#2F5D8C"
GOLD = "#C9941A"
ORANGE = "#D97706"
PINK = "#B65378"
OLIVE = "#788542"


@dataclass(slots=True)
class TelemetryModel:
    events: list[TelemetryEvent] = field(default_factory=list)

    def append(self, event: TelemetryEvent | dict[str, Any]) -> None:
        parsed = (
            event
            if isinstance(event, TelemetryEvent)
            else TelemetryEvent.from_dict(event)
        )
        if self.events and parsed.sequence_no <= self.events[-1].sequence_no:
            raise ValueError("telemetry events must have increasing sequence numbers")
        self.events.append(parsed)

    def extend(self, events: Sequence[TelemetryEvent | dict[str, Any]]) -> None:
        for event in events:
            self.append(event)

    def values(self, field_name: str) -> np.ndarray:
        return np.array(
            [float(event.payload[field_name]) for event in self.events],
            dtype=np.float64,
        )

    @property
    def days(self) -> np.ndarray:
        return np.array([event.sim_time for event in self.events], dtype=np.float64)

    def signature(self) -> tuple[tuple[int, tuple[tuple[str, float], ...]], ...]:
        return tuple(
            (
                event.sequence_no,
                tuple(
                    sorted(
                        (key, float(value)) for key, value in event.payload.items()
                    )
                ),
            )
            for event in self.events
        )


@dataclass(slots=True)
class ReplayController:
    events: tuple[TelemetryEvent, ...]
    cursor: int = 0
    paused: bool = True
    speed: float = 1.0
    accumulator: float = 0.0

    def resume(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def reset(self) -> None:
        self.cursor = 0
        self.accumulator = 0.0
        self.paused = True

    def step_once(self) -> tuple[TelemetryEvent, ...]:
        if self.cursor >= len(self.events):
            return ()
        event = self.events[self.cursor]
        self.cursor += 1
        return (event,)

    def advance(self) -> tuple[TelemetryEvent, ...]:
        if self.paused or self.cursor >= len(self.events):
            return ()
        self.accumulator += self.speed
        count = int(self.accumulator)
        if count < 1:
            return ()
        self.accumulator -= count
        end = min(self.cursor + count, len(self.events))
        batch = self.events[self.cursor : end]
        self.cursor = end
        return batch


class VisPyNetworkView(QtWidgets.QWidget):
    """Stage 2 network canvas adapter, reserved by the Stage 1 shell."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "VisPy network view is reserved for Stage 2 message propagation."
        )
        label.setStyleSheet(f"color: {MUTED}; padding: 8px;")
        layout.addWidget(label)
        canvas = scene.SceneCanvas(keys=None, bgcolor="white", show=False)
        view = canvas.central_widget.add_view()
        view.camera = scene.PanZoomCamera(rect=(-1, -1, 2, 2))
        scene.visuals.GridLines(color=(0.88, 0.9, 0.93, 1), parent=view.scene)
        layout.addWidget(canvas.native, 1)
        self.canvas = canvas


class MarketMonitorWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        mode: str,
        visualizer_config: VisualizerConfig,
        telemetry_queue: Any | None = None,
        command_queue: Any | None = None,
        completion_queue: Any | None = None,
        replay_events: tuple[TelemetryEvent, ...] = (),
    ):
        super().__init__()
        self.mode = mode
        self.visualizer_config = visualizer_config
        self.telemetry_queue = telemetry_queue
        self.command_queue = command_queue
        self.completion_queue = completion_queue
        self.model = TelemetryModel()
        self.replay = (
            ReplayController(replay_events) if mode == "replay" else None
        )
        self.setWindowTitle(f"Auditable Market ABM — {mode.title()}")
        self.resize(1400, 900)
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(round(1000 / visualizer_config.max_fps))
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=True, background="w", foreground=INK)
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Stage 1 Artificial Market Monitor")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {INK}; padding: 4px;"
        )
        self.status_label = QtWidgets.QLabel(f"{self.mode.upper()} · waiting")
        self.status_label.setStyleSheet(f"color: {MUTED};")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.status_label)
        outer.addLayout(title_row)

        controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.step_button = QtWidgets.QPushButton("Step")
        self.reset_button = QtWidgets.QPushButton("Reset replay")
        self.speed_combo = QtWidgets.QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.resume_button)
        controls.addWidget(self.step_button)
        controls.addWidget(self.reset_button)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Playback speed"))
        controls.addWidget(self.speed_combo)
        controls.addStretch(1)
        outer.addLayout(controls)

        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.step_button.clicked.connect(self.step_once)
        self.reset_button.clicked.connect(self.reset_replay)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.reset_button.setEnabled(self.mode == "replay")

        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self.price_plot = self._plot("Price and fundamental value", "Price")
        self.return_plot = self._plot("Daily return", "Return")
        self.volatility_plot = self._plot("20-day volatility", "Volatility")
        self.volume_plot = self._plot("Executed volume", "Shares")
        self.strategy_plot = self._plot("Strategy shares", "Share")
        self.conservation_plot = self._plot(
            "Accounting conservation error", "Relative error"
        )
        self.conservation_plot.setLogMode(y=True)

        grid.addWidget(self.price_plot, 0, 0)
        grid.addWidget(self.return_plot, 0, 1)
        grid.addWidget(self.volatility_plot, 1, 0)
        grid.addWidget(self.volume_plot, 1, 1)
        grid.addWidget(self.strategy_plot, 2, 0)
        grid.addWidget(self.conservation_plot, 2, 1)
        for row in range(3):
            grid.setRowStretch(row, 1)
        for column in range(2):
            grid.setColumnStretch(column, 1)
        outer.addWidget(grid_widget, 1)
        self.setCentralWidget(central)

        self.price_curve = self.price_plot.plot(
            pen=pg.mkPen(BLUE, width=2), name="Market price"
        )
        self.fundamental_curve = self.price_plot.plot(
            pen=pg.mkPen(GOLD, width=2, style=QtCore.Qt.PenStyle.DashLine),
            name="Fundamental",
        )
        self.return_curve = self.return_plot.plot(pen=pg.mkPen(BLUE, width=1.5))
        self.volatility_curve = self.volatility_plot.plot(
            pen=pg.mkPen(OLIVE, width=2)
        )
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush=ORANGE)
        self.volume_plot.addItem(self.volume_bars)
        self.value_curve = self.strategy_plot.plot(
            pen=pg.mkPen(BLUE, width=2), name="Value"
        )
        self.trend_curve = self.strategy_plot.plot(
            pen=pg.mkPen(GOLD, width=2), name="Trend"
        )
        self.noise_curve = self.strategy_plot.plot(
            pen=pg.mkPen(PINK, width=2), name="Noise"
        )
        self.cash_error_curve = self.conservation_plot.plot(
            pen=pg.mkPen(OLIVE, width=2), name="Cash"
        )
        self.share_error_curve = self.conservation_plot.plot(
            pen=pg.mkPen(BLUE, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Shares",
        )

    def _plot(self, title: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setLabel("bottom", "Trading day")
        plot.setLabel("left", y_label)
        plot.showGrid(x=False, y=True, alpha=0.25)
        plot.addLegend(offset=(8, 8))
        return plot

    def _send_command(self, action: str, value: float | None = None) -> None:
        if self.command_queue is None:
            return
        try:
            self.command_queue.put_nowait({"action": action, "value": value})
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            self.status_label.setText(f"{self.mode.upper()} · control channel detached")

    def pause(self) -> None:
        if self.replay is not None:
            self.replay.pause()
        self._send_command("pause")

    def resume(self) -> None:
        if self.replay is not None:
            self.replay.resume()
        self._send_command("resume")

    def step_once(self) -> None:
        if self.replay is not None:
            self.replay.pause()
            self.ingest(self.replay.step_once())
        self._send_command("step")

    def reset_replay(self) -> None:
        if self.replay is None:
            return
        self.replay.reset()
        self.model = TelemetryModel()
        self.refresh_plots()

    def _speed_changed(self) -> None:
        speed = float(self.speed_combo.currentData())
        if self.replay is not None:
            self.replay.speed = speed
        self._send_command("speed", speed)

    def ingest(self, events: Sequence[TelemetryEvent | dict[str, Any]]) -> None:
        if not events:
            return
        self.model.extend(events)
        self.refresh_plots()

    def refresh_plots(self) -> None:
        if not self.model.events:
            for curve in (
                self.price_curve,
                self.fundamental_curve,
                self.return_curve,
                self.volatility_curve,
                self.value_curve,
                self.trend_curve,
                self.noise_curve,
                self.cash_error_curve,
                self.share_error_curve,
            ):
                curve.setData([], [])
            self.volume_bars.setOpts(x=[], height=[])
            self.status_label.setText(f"{self.mode.upper()} · waiting")
            return
        days = self.model.days
        self.price_curve.setData(days, self.model.values("price"))
        self.fundamental_curve.setData(
            days, self.model.values("fundamental_value")
        )
        self.return_curve.setData(days, self.model.values("return"))
        self.volatility_curve.setData(days, self.model.values("volatility_20d"))
        self.volume_bars.setOpts(x=days, height=self.model.values("volume"))
        self.value_curve.setData(days, self.model.values("value_share"))
        self.trend_curve.setData(days, self.model.values("trend_share"))
        self.noise_curve.setData(days, self.model.values("noise_share"))
        self.cash_error_curve.setData(
            days,
            np.maximum(self.model.values("cash_relative_error"), 1e-18),
        )
        self.share_error_curve.setData(
            days,
            np.maximum(self.model.values("share_relative_error"), 1e-18),
        )
        self.status_label.setText(
            f"{self.mode.upper()} · day {int(days[-1])} · "
            f"{len(self.model.events)} frames"
        )

    def _drain_live_queue(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.telemetry_queue is None:
            return events
        while True:
            try:
                events.append(self.telemetry_queue.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError):
                break
        return events

    def _on_timer(self) -> None:
        if self.mode == "live":
            self.ingest(self._drain_live_queue())
            if self.completion_queue is not None:
                try:
                    completion = self.completion_queue.get_nowait()
                except queue.Empty:
                    completion = None
                if completion:
                    if completion.get("error"):
                        self.status_label.setText(
                            f"LIVE · worker error: {completion['error']}"
                        )
                    else:
                        self.status_label.setText(
                            f"LIVE · complete · {completion['fingerprint'][:12]}"
                        )
        elif self.replay is not None:
            self.ingest(self.replay.advance())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._send_command("detach")
        super().closeEvent(event)


def _drain_commands(command_queue: Any) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    while True:
        try:
            commands.append(command_queue.get_nowait())
        except queue.Empty:
            break
        except (EOFError, OSError, ValueError):
            break
    return commands


def live_worker(
    config_payload: dict[str, Any],
    output_dir: str,
    telemetry_queue: Any,
    command_queue: Any,
    completion_queue: Any,
) -> None:
    try:
        config = Stage1Config.from_dict(config_payload)
        sink = NonBlockingQueueSink(telemetry_queue)
        harness = MarketHarness(config, telemetry_sink=sink)
        paused = False
        detached = False
        speed = 1.0
        step_budget = 0
        while harness.next_day_index < config.trading_days:
            for command in _drain_commands(command_queue):
                action = command.get("action")
                if action == "pause":
                    paused = True
                elif action == "resume":
                    paused = False
                elif action == "step":
                    paused = True
                    step_budget += 1
                elif action == "speed":
                    speed = max(0.01, float(command.get("value") or 1.0))
                elif action == "detach":
                    detached = True
                    paused = False
            if paused and step_budget <= 0:
                time.sleep(0.01)
                continue
            harness.step(harness.next_day_index)
            if step_budget > 0:
                step_budget -= 1
            if not detached:
                time.sleep(0.1 / speed)
        result = harness.run()
        result.write(output_dir, config)
        completion_queue.put(
            {
                "fingerprint": result.fingerprint(),
                "published_frames": sink.published_count,
                "dropped_frames": sink.dropped_count,
            }
        )
    except Exception as exc:
        completion_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def _application() -> QtWidgets.QApplication:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv
    )
    font_candidates = (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for font_path in font_candidates:
        if not font_path.is_file():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QtGui.QFont(families[0], 10))
            break
    return application


def run_replay_window(
    run_dir: Path,
    visualizer_config: VisualizerConfig,
    *,
    screenshot: Path | None = None,
) -> int:
    application = _application()
    events = read_telemetry(run_dir)
    window = MarketMonitorWindow(
        mode="replay",
        visualizer_config=visualizer_config,
        replay_events=events,
    )
    window.show()
    if screenshot is not None:
        window.ingest(events)
        application.processEvents()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(screenshot)):
            raise RuntimeError(f"failed to save desktop screenshot to {screenshot}")
        window.close()
        return 0
    return application.exec()


def run_live_window(
    config: Stage1Config,
    run_dir: Path,
    visualizer_config: VisualizerConfig,
) -> int:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    context = multiprocessing.get_context("spawn")
    telemetry_queue = context.Queue(maxsize=visualizer_config.queue_size)
    command_queue = context.Queue(maxsize=64)
    completion_queue = context.Queue(maxsize=4)
    worker = context.Process(
        target=live_worker,
        args=(
            config.to_dict(),
            str(run_dir),
            telemetry_queue,
            command_queue,
            completion_queue,
        ),
        name="abm-harness",
    )
    worker.start()
    application = _application()
    window = MarketMonitorWindow(
        mode="live",
        visualizer_config=visualizer_config,
        telemetry_queue=telemetry_queue,
        command_queue=command_queue,
        completion_queue=completion_queue,
    )
    window.show()
    exit_code = application.exec()
    worker.join()
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch", "live", "replay"), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage1.json")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-fps", type=int, default=10)
    parser.add_argument("--screenshot", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visualizer_config = VisualizerConfig(mode=args.mode, max_fps=args.max_fps)
    if args.mode == "batch":
        config = load_stage1_config(args.config)
        result = run_stage1(config)
        result.write(args.run_dir, config)
        print(json.dumps(result.summary(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.mode == "replay":
        return run_replay_window(
            args.run_dir,
            visualizer_config,
            screenshot=args.screenshot,
        )
    config = load_stage1_config(args.config)
    return run_live_window(config, args.run_dir, visualizer_config)


if __name__ == "__main__":
    raise SystemExit(main())
