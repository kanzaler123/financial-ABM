"""Shared process-safe control loop for live Stage 1 simulations."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .config import Stage1Config
from .harness import MarketHarness
from .runner import resolve_code_revision
from .telemetry import NonBlockingQueueSink

RunState = Literal["starting", "running", "paused", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class RunStatus:
    state: RunState
    current_day: int
    trading_days: int
    fingerprint: str | None = None
    published_frames: int = 0
    dropped_frames: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "current_day": self.current_day,
            "trading_days": self.trading_days,
            "fingerprint": self.fingerprint,
            "published_frames": self.published_frames,
            "dropped_frames": self.dropped_frames,
            "error": self.error,
        }


def _drain_commands(command_queue: Any) -> list[Mapping[str, Any]]:
    commands: list[Mapping[str, Any]] = []
    while True:
        try:
            command = command_queue.get_nowait()
        except queue.Empty:
            break
        except (EOFError, OSError, ValueError):
            break
        if isinstance(command, Mapping):
            commands.append(command)
    return commands


def _publish_status(status_queue: Any | None, status: Mapping[str, Any]) -> None:
    if status_queue is None:
        return
    try:
        status_queue.put_nowait(status)
    except (queue.Full, EOFError, OSError, ValueError):
        # Completion is delivered separately on the reliable completion queue.
        # A slow or disconnected observer must never abort the simulation.
        pass


def controlled_run_worker(
    config_payload: Mapping[str, Any],
    output_dir: str,
    telemetry_queue: Any,
    command_queue: Any,
    completion_queue: Any,
    status_queue: Any | None = None,
) -> None:
    """Run one simulation without allowing observers to affect its state."""
    sink = NonBlockingQueueSink(telemetry_queue)
    try:
        config = Stage1Config.from_dict(config_payload)
        harness = MarketHarness(config, telemetry_sink=sink)
        paused = False
        detached = False
        speed = 1.0
        step_budget = 0
        if status_queue is not None:
            _publish_status(status_queue,
                RunStatus("starting", 0, config.trading_days).to_dict()
            )
        while harness.next_day_index < config.trading_days:
            state_changed = False
            for command in _drain_commands(command_queue):
                action = command.get("action")
                if action == "pause":
                    paused = True
                    state_changed = True
                elif action == "resume":
                    paused = False
                    state_changed = True
                elif action == "step":
                    paused = True
                    step_budget += 1
                    state_changed = True
                elif action == "speed":
                    speed = max(0.01, float(command.get("value") or 1.0))
                elif action == "detach":
                    detached = True
                    paused = False
                    state_changed = True
            if paused and step_budget <= 0:
                if state_changed and status_queue is not None:
                    _publish_status(status_queue,
                        RunStatus(
                            "paused",
                            harness.next_day_index,
                            config.trading_days,
                            published_frames=sink.published_count,
                            dropped_frames=sink.dropped_count,
                        ).to_dict()
                    )
                time.sleep(0.01)
                continue
            harness.step(harness.next_day_index)
            if step_budget > 0:
                step_budget -= 1
            if status_queue is not None:
                _publish_status(status_queue,
                    RunStatus(
                        "paused" if paused else "running",
                        harness.next_day_index,
                        config.trading_days,
                        published_frames=sink.published_count,
                        dropped_frames=sink.dropped_count,
                    ).to_dict()
                )
            if not detached:
                time.sleep(0.1 / speed)
        result = harness.run()
        result.write(
            Path(output_dir),
            config,
            code_revision=resolve_code_revision(),
        )
        status = RunStatus(
            "completed",
            config.trading_days,
            config.trading_days,
            fingerprint=result.fingerprint(),
            published_frames=sink.published_count,
            dropped_frames=sink.dropped_count,
        )
        completion = status.to_dict()
        completion.pop("error")
        completion_queue.put(completion)
        if status_queue is not None:
            _publish_status(status_queue, status.to_dict())
    except Exception as exc:
        failure = RunStatus(
            "failed",
            0,
            int(config_payload.get("trading_days", 0)),
            published_frames=sink.published_count,
            dropped_frames=sink.dropped_count,
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()
        completion_queue.put(failure)
        if status_queue is not None:
            _publish_status(status_queue, failure)
