"""Validated configuration for the deterministic Stage 1 market."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

STRATEGY_NAMES = ("value", "trend", "noise")
FUNDAMENTAL_PROCESSES = ("gaussian", "student_t", "garch_t")


@dataclass(frozen=True, slots=True)
class Announcement:
    day: int
    fundamental_delta: float
    source: str

    def __post_init__(self) -> None:
        if self.day < 1:
            raise ValueError("announcement day uses one-based indexing and must be positive")
        if not math.isfinite(self.fundamental_delta):
            raise ValueError("fundamental_delta must be finite")
        if not self.source.strip():
            raise ValueError("announcement source must not be empty")


@dataclass(frozen=True, slots=True)
class Stage1Config:
    schema_version: str
    population_size: int
    trading_days: int
    seed: int
    asset_id: str
    initial_price: float
    initial_fundamental: float
    fundamental_process: str
    fundamental_drift: float
    fundamental_volatility: float
    fundamental_shock_df: float
    fundamental_arch: float
    fundamental_garch: float
    fundamental_common_correlation: float
    public_news_price_pass_through: float
    initial_agent_cash: float
    initial_agent_position: float
    initial_wealth_dispersion: float
    market_maker_cash: float
    market_maker_inventory: float
    transaction_cost_rate: float
    price_impact: float
    liquidity_scale: float
    max_log_return: float
    max_order_fraction: float
    max_short_leverage: float
    base_activity_rate: float
    activity_volatility_sensitivity: float
    liquidity_volatility_sensitivity: float
    liquidity_stress_threshold: float
    minimum_liquidity_fraction: float
    value_sensitivity: float
    trend_sensitivity: float
    trend_lookback: int
    noise_scale: float
    idiosyncratic_signal_scale: float
    common_signal_correlation: float
    learning_enabled: bool
    learning_interval: int
    learning_update_fraction: float
    learning_temperature: float
    risk_penalty: float
    strategy_shares: dict[str, float]
    announcements: tuple[Announcement, ...] = ()

    def __post_init__(self) -> None:
        if not self.schema_version.strip() or not self.asset_id.strip():
            raise ValueError("schema_version and asset_id must not be empty")
        if self.fundamental_process not in FUNDAMENTAL_PROCESSES:
            raise ValueError(
                "fundamental_process must be one of "
                f"{FUNDAMENTAL_PROCESSES}"
            )
        if self.population_size <= 0 or self.trading_days <= 0:
            raise ValueError("population_size and trading_days must be positive")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        positive_fields = (
            "initial_price",
            "initial_fundamental",
            "initial_agent_cash",
            "market_maker_cash",
            "market_maker_inventory",
            "liquidity_scale",
            "max_log_return",
            "learning_temperature",
            "liquidity_stress_threshold",
        )
        nonnegative_fields = (
            "initial_agent_position",
            "initial_wealth_dispersion",
            "fundamental_volatility",
            "fundamental_shock_df",
            "fundamental_arch",
            "fundamental_garch",
            "fundamental_common_correlation",
            "public_news_price_pass_through",
            "transaction_cost_rate",
            "price_impact",
            "max_order_fraction",
            "max_short_leverage",
            "base_activity_rate",
            "activity_volatility_sensitivity",
            "liquidity_volatility_sensitivity",
            "minimum_liquidity_fraction",
            "value_sensitivity",
            "trend_sensitivity",
            "noise_scale",
            "idiosyncratic_signal_scale",
            "common_signal_correlation",
            "risk_penalty",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.transaction_cost_rate >= 1:
            raise ValueError("transaction_cost_rate must be below one")
        if self.max_order_fraction > 1:
            raise ValueError("max_order_fraction must not exceed one")
        if self.max_short_leverage > 1:
            raise ValueError("max_short_leverage must not exceed one")
        if self.base_activity_rate > 1:
            raise ValueError("base_activity_rate must not exceed one")
        if self.minimum_liquidity_fraction > 1:
            raise ValueError("minimum_liquidity_fraction must not exceed one")
        if self.common_signal_correlation > 1:
            raise ValueError("common_signal_correlation must not exceed one")
        if not math.isfinite(self.fundamental_drift):
            raise ValueError("fundamental_drift must be finite")
        if self.public_news_price_pass_through > 1:
            raise ValueError(
                "public_news_price_pass_through must not exceed one"
            )
        if self.fundamental_common_correlation > 1:
            raise ValueError(
                "fundamental_common_correlation must not exceed one"
            )
        if self.fundamental_process == "gaussian":
            if (
                self.fundamental_shock_df != 0
                or self.fundamental_arch != 0
                or self.fundamental_garch != 0
            ):
                raise ValueError(
                    "gaussian fundamental_process requires zero df/ARCH/GARCH"
                )
        elif self.fundamental_process == "student_t":
            if self.fundamental_shock_df <= 4:
                raise ValueError(
                    "student_t fundamental_process requires df above four"
                )
            if self.fundamental_arch != 0 or self.fundamental_garch != 0:
                raise ValueError(
                    "student_t fundamental_process requires zero ARCH/GARCH"
                )
        elif self.fundamental_shock_df <= 4:
            raise ValueError(
                "garch_t fundamental_process requires df above four"
            )
        if self.fundamental_arch + self.fundamental_garch >= 1:
            raise ValueError(
                "fundamental_arch + fundamental_garch must be below one"
            )
        if not 0 < self.learning_update_fraction <= 1:
            raise ValueError("learning_update_fraction must be in (0, 1]")
        if self.trend_lookback < 1 or self.learning_interval < 1:
            raise ValueError("trend_lookback and learning_interval must be positive")
        if set(self.strategy_shares) != set(STRATEGY_NAMES):
            raise ValueError(f"strategy_shares must define exactly {STRATEGY_NAMES}")
        shares = tuple(self.strategy_shares[name] for name in STRATEGY_NAMES)
        if any(not math.isfinite(share) or share < 0 for share in shares):
            raise ValueError("strategy shares must be finite and nonnegative")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-12):
            raise ValueError("strategy shares must sum to one")
        if any(event.day > self.trading_days for event in self.announcements):
            raise ValueError("announcement day exceeds trading_days")
        announcement_days = [event.day for event in self.announcements]
        if len(announcement_days) != len(set(announcement_days)):
            raise ValueError("only one external announcement is allowed per day")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1Config":
        expected = {field.name for field in fields(cls)}
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing or unknown:
            raise ValueError(
                f"invalid Stage1Config keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        values = dict(payload)
        announcements = values["announcements"]
        if not isinstance(announcements, list):
            raise TypeError("announcements must be a JSON array")
        values["announcements"] = tuple(
            Announcement(
                day=int(item["day"]),
                fundamental_delta=float(item["fundamental_delta"]),
                source=str(item["source"]),
            )
            for item in announcements
        )
        values["strategy_shares"] = {
            str(name): float(share)
            for name, share in dict(values["strategy_shares"]).items()
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["announcements"] = [asdict(event) for event in self.announcements]
        return payload


def load_stage1_config(path: str | Path) -> Stage1Config:
    with Path(path).open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError("Stage 1 config root must be a JSON object")
    return Stage1Config.from_dict(payload)
