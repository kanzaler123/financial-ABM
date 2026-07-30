"""Validated and JSON-serializable contracts for the ABM harness."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date, datetime
from types import UnionType
from typing import Any, Literal, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
OrderSide = Literal["buy", "sell", "hold"]

SchemaT = TypeVar("SchemaT", bound="SchemaMixin")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _encode_json(value: Any) -> JsonValue:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_encode_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    return value


def _decode_json(value: JsonValue, expected_type: Any) -> Any:
    if expected_type is datetime:
        if not isinstance(value, str):
            raise TypeError("datetime values must be ISO-8601 strings")
        return datetime.fromisoformat(value)
    if expected_type is date:
        if not isinstance(value, str):
            raise TypeError("date values must be ISO-8601 strings")
        return date.fromisoformat(value)

    origin = get_origin(expected_type)
    arguments = get_args(expected_type)

    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("tuple fields must be represented by JSON arrays")
        item_type = arguments[0] if arguments else Any
        return tuple(_decode_json(item, item_type) for item in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("mapping fields must be represented by JSON objects")
        key_type, item_type = arguments
        if key_type is not str:
            raise TypeError("only string-keyed JSON mappings are supported")
        return {key: _decode_json(item, item_type) for key, item in value.items()}
    if origin in (Union, UnionType):
        if value is None and type(None) in arguments:
            return None
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _decode_json(value, candidate)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"value does not match {expected_type!r}")
    if origin is Literal:
        if value not in arguments:
            raise ValueError(f"expected one of {arguments!r}, got {value!r}")
        return value
    return value


class SchemaMixin:
    """Shared deterministic serialization for frozen dataclass contracts."""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            field.name: _encode_json(getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def from_dict(cls: type[SchemaT], data: Mapping[str, JsonValue]) -> SchemaT:
        hints = get_type_hints(cls)
        expected_names = {field.name for field in fields(cls)}
        unknown = set(data) - expected_names
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        values = {
            field.name: _decode_json(data[field.name], hints[field.name])
            for field in fields(cls)
            if field.name in data
        }
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MarketState(SchemaMixin):
    date: date
    asset_id: str
    price: float
    fundamental_value: float
    volume: float
    market_maker_inventory: float
    market_maker_cash: float

    def __post_init__(self) -> None:
        _require_non_empty(self.asset_id, "asset_id")
        for name in (
            "price",
            "fundamental_value",
            "volume",
            "market_maker_inventory",
            "market_maker_cash",
        ):
            _require_finite(getattr(self, name), name)
        if self.price <= 0 or self.fundamental_value <= 0:
            raise ValueError("price and fundamental_value must be positive")
        if self.volume < 0:
            raise ValueError("volume must not be negative")


@dataclass(frozen=True, slots=True)
class AgentState(SchemaMixin):
    agent_id: str
    agent_type: str
    cash: float
    positions: dict[str, float]
    beliefs: dict[str, JsonValue]
    memory_refs: tuple[str, ...]
    active: bool = True

    def __post_init__(self) -> None:
        _require_non_empty(self.agent_id, "agent_id")
        _require_non_empty(self.agent_type, "agent_type")
        _require_finite(self.cash, "cash")
        for asset_id, position in self.positions.items():
            _require_non_empty(asset_id, "positions key")
            _require_finite(position, f"positions[{asset_id!r}]")


@dataclass(frozen=True, slots=True)
class CompanyState(SchemaMixin):
    company_id: str
    cik: str
    assets: float
    cash_flow: float
    disclosure_state: str
    next_decision_date: date

    def __post_init__(self) -> None:
        _require_non_empty(self.company_id, "company_id")
        _require_non_empty(self.cik, "cik")
        _require_non_empty(self.disclosure_state, "disclosure_state")
        _require_finite(self.assets, "assets")
        _require_finite(self.cash_flow, "cash_flow")
        if self.assets < 0:
            raise ValueError("assets must not be negative")


@dataclass(frozen=True, slots=True)
class Message(SchemaMixin):
    message_id: str
    source_id: str
    target_scope: tuple[str, ...]
    event_time: datetime
    available_at: datetime
    content_type: str
    payload: dict[str, JsonValue]
    credibility: float
    parent_message_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.message_id, "message_id")
        _require_non_empty(self.source_id, "source_id")
        _require_non_empty(self.content_type, "content_type")
        if not self.target_scope:
            raise ValueError("target_scope must contain at least one target")
        if self.available_at < self.event_time:
            raise ValueError("available_at must not precede event_time")
        _require_finite(self.credibility, "credibility")
        if not 0 <= self.credibility <= 1:
            raise ValueError("credibility must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Edge(SchemaMixin):
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    trust: float
    delay_days: int
    valid_from: date
    valid_to: date | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.source_id, "source_id")
        _require_non_empty(self.target_id, "target_id")
        _require_non_empty(self.edge_type, "edge_type")
        _require_finite(self.weight, "weight")
        _require_finite(self.trust, "trust")
        if not 0 <= self.weight <= 1 or not 0 <= self.trust <= 1:
            raise ValueError("weight and trust must be between 0 and 1")
        if self.delay_days < 0:
            raise ValueError("delay_days must not be negative")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not precede valid_from")


@dataclass(frozen=True, slots=True)
class Order(SchemaMixin):
    agent_id: str
    asset_id: str
    side: OrderSide
    quantity: float
    submitted_at: datetime
    reason_code: str

    def __post_init__(self) -> None:
        _require_non_empty(self.agent_id, "agent_id")
        _require_non_empty(self.asset_id, "asset_id")
        _require_non_empty(self.reason_code, "reason_code")
        _require_finite(self.quantity, "quantity")
        if self.side not in ("buy", "sell", "hold"):
            raise ValueError("side must be buy, sell, or hold")
        if self.side == "hold" and self.quantity != 0:
            raise ValueError("hold orders must have zero quantity")
        if self.side in ("buy", "sell") and self.quantity <= 0:
            raise ValueError("buy and sell orders must have positive quantity")


@dataclass(frozen=True, slots=True)
class FactorEvent(SchemaMixin):
    factor_id: str
    event_time: datetime
    available_at: datetime
    source: str
    targets: tuple[str, ...]
    channel: str
    true_value: JsonValue
    observed_message: str
    uncertainty: float
    persistence: int

    def __post_init__(self) -> None:
        _require_non_empty(self.factor_id, "factor_id")
        _require_non_empty(self.source, "source")
        _require_non_empty(self.channel, "channel")
        if not self.targets:
            raise ValueError("targets must contain at least one target")
        if self.available_at < self.event_time:
            raise ValueError("available_at must not precede event_time")
        _require_finite(self.uncertainty, "uncertainty")
        if not 0 <= self.uncertainty <= 1:
            raise ValueError("uncertainty must be between 0 and 1")
        if self.persistence < 0:
            raise ValueError("persistence must not be negative")


@dataclass(frozen=True, slots=True)
class RunManifest(SchemaMixin):
    schema_version: str
    run_id: str
    code_revision: str
    config_sha256: str
    data_sha256: dict[str, str]
    random_seed: int
    agent_models: dict[str, str]
    prompt_sha256: dict[str, str]
    graph_version: str | None
    enabled_plugins: tuple[str, ...]
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.schema_version, "schema_version"),
            (self.run_id, "run_id"),
            (self.code_revision, "code_revision"),
            (self.config_sha256, "config_sha256"),
        ):
            _require_non_empty(value, name)
        if self.random_seed < 0:
            raise ValueError("random_seed must not be negative")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
