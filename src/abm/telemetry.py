"""Read-only telemetry contracts and non-blocking display transport."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

import duckdb

TelemetryMode = Literal["batch", "live", "replay"]
JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class VisualizerConfig:
    mode: TelemetryMode
    max_fps: int = 10
    snapshot_interval: int = 1
    queue_size: int = 2048

    def __post_init__(self) -> None:
        if self.mode not in ("batch", "live", "replay"):
            raise ValueError("mode must be batch, live, or replay")
        if not 1 <= self.max_fps <= 60:
            raise ValueError("max_fps must be between 1 and 60")
        if self.snapshot_interval < 1 or self.queue_size < 1:
            raise ValueError("snapshot_interval and queue_size must be positive")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    run_id: str
    sim_time: int
    sequence_no: int
    event_type: str
    payload: dict[str, JsonScalar]

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.event_type.strip():
            raise ValueError("run_id and event_type must not be empty")
        if self.sim_time < 0 or self.sequence_no < 0:
            raise ValueError("sim_time and sequence_no must be nonnegative")
        if any(
            not isinstance(value, (str, int, float, bool)) and value is not None
            for value in self.payload.values()
        ):
            raise TypeError("telemetry payload values must be JSON scalars")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "sim_time": self.sim_time,
            "sequence_no": self.sequence_no,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TelemetryEvent":
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("telemetry payload must be an object")
        return cls(
            run_id=str(value["run_id"]),
            sim_time=int(value["sim_time"]),
            sequence_no=int(value["sequence_no"]),
            event_type=str(value["event_type"]),
            payload={str(key): item for key, item in payload.items()},
        )


class TelemetrySink(Protocol):
    def publish(self, event: TelemetryEvent) -> bool:
        """Publish without mutating or delaying the simulation."""


@dataclass(slots=True)
class NonBlockingQueueSink:
    """Best-effort display transport; full or detached queues drop frames."""

    output_queue: Any
    published_count: int = 0
    dropped_count: int = 0

    def publish(self, event: TelemetryEvent) -> bool:
        try:
            self.output_queue.put_nowait(event.to_dict())
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            self.dropped_count += 1
            return False
        self.published_count += 1
        return True


TELEMETRY_COLUMNS = (
    "price",
    "return",
    "volatility_20d",
    "fundamental_value",
    "volume",
    "value_share",
    "trend_share",
    "noise_share",
    "cash_relative_error",
    "share_relative_error",
)


def write_telemetry(
    events: tuple[TelemetryEvent, ...],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write a lossless event table to DuckDB and Parquet."""
    destination = Path(output_dir)
    database_path = destination / "telemetry.duckdb"
    parquet_path = destination / "telemetry.parquet"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            """
            CREATE TABLE telemetry (
                run_id VARCHAR NOT NULL,
                sim_time INTEGER NOT NULL,
                sequence_no INTEGER NOT NULL,
                event_type VARCHAR NOT NULL,
                price DOUBLE NOT NULL,
                return DOUBLE NOT NULL,
                volatility_20d DOUBLE NOT NULL,
                fundamental_value DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                value_share DOUBLE NOT NULL,
                trend_share DOUBLE NOT NULL,
                noise_share DOUBLE NOT NULL,
                cash_error DOUBLE NOT NULL,
                share_error DOUBLE NOT NULL,
                payload_json VARCHAR NOT NULL,
                PRIMARY KEY (run_id, sequence_no)
            )
            """
        )
        rows = []
        for event in events:
            rows.append(
                (
                    event.run_id,
                    event.sim_time,
                    event.sequence_no,
                    event.event_type,
                    *(float(event.payload[column]) for column in TELEMETRY_COLUMNS),
                    json.dumps(
                        event.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
            )
        connection.executemany(
            """
            INSERT INTO telemetry VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )
        escaped_parquet_path = str(parquet_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT * FROM telemetry ORDER BY sequence_no
            ) TO '{escaped_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return database_path, parquet_path


def read_telemetry(
    run_dir: str | Path,
    *,
    after_sequence: int = -1,
    limit: int | None = None,
) -> tuple[TelemetryEvent, ...]:
    """Read a replay range without rerunning the market."""
    if after_sequence < -1:
        raise ValueError("after_sequence must be at least -1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source = Path(run_dir)
    parquet_path = source / "telemetry.parquet"
    database_path = source / "telemetry.duckdb"
    limit_clause = "" if limit is None else f" LIMIT {limit}"
    if parquet_path.is_file():
        connection = duckdb.connect()
        try:
            escaped_parquet_path = str(parquet_path).replace("'", "''")
            rows = connection.execute(
                f"""
                SELECT run_id, sim_time, sequence_no, event_type, payload_json
                FROM read_parquet('{escaped_parquet_path}')
                WHERE sequence_no > ?
                ORDER BY sequence_no{limit_clause}
                """,
                [after_sequence],
            ).fetchall()
        finally:
            connection.close()
    elif database_path.is_file():
        connection = duckdb.connect(str(database_path), read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT run_id, sim_time, sequence_no, event_type, payload_json
                FROM telemetry
                WHERE sequence_no > ?
                ORDER BY sequence_no{limit_clause}
                """,
                [after_sequence],
            ).fetchall()
        finally:
            connection.close()
    else:
        raise FileNotFoundError(f"no telemetry store found in {source}")
    return tuple(
        TelemetryEvent(
            run_id=row[0],
            sim_time=row[1],
            sequence_no=row[2],
            event_type=row[3],
            payload=json.loads(row[4]),
        )
        for row in rows
    )
