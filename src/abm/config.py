"""Validated configuration for the deterministic Stage 1 market."""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

STRATEGY_NAMES = ("value", "trend", "noise")
FUNDAMENTAL_PROCESSES = ("gaussian", "student_t", "garch_t")
PRICE_FORMATION_MODES = ("quasi_order_book",)


@dataclass(frozen=True, slots=True)
class Announcement:
    day: int
    fundamental_delta: float
    source: str

    def __post_init__(self) -> None:
        if type(self.day) is not int or self.day < 1:
            raise ValueError("announcement day must be a positive integer")
        if isinstance(self.fundamental_delta, bool) or not isinstance(
            self.fundamental_delta, (int, float)
        ):
            raise ValueError("fundamental_delta must be a finite number")
        if not math.isfinite(self.fundamental_delta):
            raise ValueError("fundamental_delta must be finite")
        if not isinstance(self.source, str) or not self.source.strip():
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
    price_formation: str = "quasi_order_book"
    target_position_fraction: float = 0.25
    value_target_adjustment: float = 0.05
    trend_target_adjustment: float = 0.12
    noise_target_adjustment: float = 0.20
    value_no_trade_band: float = 0.005
    initial_subjective_value_dispersion: float = 0.02
    information_response_dispersion: float = 0.10
    value_update_rate: float = 0.10
    trend_short_lookback_min: int = 5
    trend_short_lookback_max: int = 20
    trend_long_lookback_min: int = 40
    trend_long_lookback_max: int = 120
    signal_volatility_floor: float = 0.0025
    activity_signal_sensitivity: float = 1.5
    activity_rate_dispersion: float = 0.25
    activity_persistence: float = 0.95
    activity_shock_scale: float = 0.08
    common_signal_persistence: float = 0.80
    liquidity_need_scale: float = 0.05
    liquidity_need_persistence: float = 0.90
    permanent_impact_fraction: float = 0.45
    transient_impact_decay: float = 0.92
    order_flow_memory: float = 0.30
    base_spread_bps: float = 5.0
    spread_volatility_sensitivity: float = 8.0
    depth_resilience: float = 0.15
    volatility_ewma_decay: float = 0.80
    participation_pressure_exponent: float = 1.0
    demand_volatility_exponent: float = 0.0
    max_position_float_multiple: float = 5.0
    volatility_reference_decay: float = 0.99
    burn_in_days: int = 0

    def __post_init__(self) -> None:
        # Dataclass annotations do not validate JSON or direct constructors.
        # Reject coercions such as True -> 1 and "false" -> truthy before any
        # random allocation, range iteration, or simulation can take place.
        for item in fields(self):
            value = getattr(self, item.name)
            if item.type in (int, "int") and type(value) is not int:
                raise ValueError(f"{item.name} must be an integer")
            if item.type in (bool, "bool") and type(value) is not bool:
                raise ValueError(f"{item.name} must be a boolean")
            if item.type in (str, "str") and not isinstance(value, str):
                raise ValueError(f"{item.name} must be a string")
            if item.type in (float, "float") and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{item.name} must be a finite number")
        if not self.schema_version.strip() or not self.asset_id.strip():
            raise ValueError("schema_version and asset_id must not be empty")
        if self.fundamental_process not in FUNDAMENTAL_PROCESSES:
            raise ValueError(
                "fundamental_process must be one of "
                f"{FUNDAMENTAL_PROCESSES}"
            )
        if self.price_formation not in PRICE_FORMATION_MODES:
            raise ValueError(
                "price_formation must be one of "
                f"{PRICE_FORMATION_MODES}"
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
            "signal_volatility_floor",
            "base_spread_bps",
            "minimum_liquidity_fraction",
        )
        nonnegative_fields = (
            "initial_agent_position",
            "initial_wealth_dispersion",
            "fundamental_volatility",
            "fundamental_shock_df",
            "fundamental_arch",
            "fundamental_garch",
            "public_news_price_pass_through",
            "transaction_cost_rate",
            "price_impact",
            "max_order_fraction",
            "max_short_leverage",
            "base_activity_rate",
            "activity_volatility_sensitivity",
            "liquidity_volatility_sensitivity",
            "liquidity_stress_threshold",
            "value_sensitivity",
            "trend_sensitivity",
            "noise_scale",
            "idiosyncratic_signal_scale",
            "common_signal_correlation",
            "risk_penalty",
            "target_position_fraction",
            "value_target_adjustment",
            "trend_target_adjustment",
            "noise_target_adjustment",
            "value_no_trade_band",
            "initial_subjective_value_dispersion",
            "information_response_dispersion",
            "value_update_rate",
            "activity_signal_sensitivity",
            "activity_rate_dispersion",
            "activity_persistence",
            "activity_shock_scale",
            "common_signal_persistence",
            "liquidity_need_scale",
            "liquidity_need_persistence",
            "permanent_impact_fraction",
            "transient_impact_decay",
            "order_flow_memory",
            "spread_volatility_sensitivity",
            "depth_resilience",
            "volatility_ewma_decay",
            "participation_pressure_exponent",
            "demand_volatility_exponent",
            "max_position_float_multiple",
            "volatility_reference_decay",
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
        unit_interval_fields = (
            "target_position_fraction",
            "value_target_adjustment",
            "trend_target_adjustment",
            "noise_target_adjustment",
            "value_update_rate",
            "activity_persistence",
            "common_signal_persistence",
            "liquidity_need_persistence",
            "permanent_impact_fraction",
            "transient_impact_decay",
            "order_flow_memory",
            "depth_resilience",
        )
        for name in unit_interval_fields:
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed one")
        if not 0 < self.volatility_ewma_decay <= 1:
            raise ValueError("volatility_ewma_decay must be in (0, 1]")
        if self.participation_pressure_exponent > 3:
            raise ValueError(
                "participation_pressure_exponent must not exceed three"
            )
        if self.demand_volatility_exponent > 1.5:
            raise ValueError(
                "demand_volatility_exponent must not exceed 1.5"
            )
        if self.max_position_float_multiple > 100:
            raise ValueError(
                "max_position_float_multiple must not exceed one hundred"
            )
        if self.volatility_reference_decay > 1:
            raise ValueError("volatility_reference_decay must not exceed one")
        if not math.isfinite(self.fundamental_drift):
            raise ValueError("fundamental_drift must be finite")
        if self.public_news_price_pass_through > 1:
            raise ValueError(
                "public_news_price_pass_through must not exceed one"
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
        if self.learning_interval < 1:
            raise ValueError("learning_interval must be positive")
        if not (
            1
            <= self.trend_short_lookback_min
            <= self.trend_short_lookback_max
            < self.trend_long_lookback_min
            <= self.trend_long_lookback_max
        ):
            raise ValueError(
                "trend lookbacks must satisfy "
                "1 <= short_min <= short_max < long_min <= long_max"
            )
        if self.burn_in_days < 0 or self.burn_in_days >= self.trading_days:
            raise ValueError("burn_in_days must be in [0, trading_days)")
        if not isinstance(self.strategy_shares, dict):
            raise ValueError("strategy_shares must be an object")
        if set(self.strategy_shares) != set(STRATEGY_NAMES):
            raise ValueError(f"strategy_shares must define exactly {STRATEGY_NAMES}")
        shares = tuple(self.strategy_shares[name] for name in STRATEGY_NAMES)
        if any(
            isinstance(share, bool) or not isinstance(share, (int, float))
            or not math.isfinite(share) or share < 0
            for share in shares
        ):
            raise ValueError("strategy shares must be finite and nonnegative")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-12):
            raise ValueError("strategy shares must sum to one")
        if not isinstance(self.announcements, tuple) or any(
            not isinstance(event, Announcement) for event in self.announcements
        ):
            raise ValueError("announcements must be a tuple of Announcement objects")
        if any(event.day > self.trading_days for event in self.announcements):
            raise ValueError("announcement day exceeds trading_days")
        announcement_days = [event.day for event in self.announcements]
        if len(announcement_days) != len(set(announcement_days)):
            raise ValueError("only one external announcement is allowed per day")

    @property
    def position_cap(self) -> float:
        """Supply-bounded per-trader position limit (shares).

        Positions are additionally bounded by wealth leverage, but wealth
        grows with trading profits, so a supply-based cap keeps aggregate
        positions finite even when a strategy cohort chases a trend.
        """
        float_per_capita = (
            self.initial_agent_position
            + self.market_maker_inventory / self.population_size
        )
        return self.max_position_float_multiple * float_per_capita

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1Config":
        if not isinstance(payload, Mapping):
            raise TypeError("Stage1Config must be a JSON object")
        config_fields = fields(cls)
        expected = {field.name for field in config_fields}
        required = {
            field.name
            for field in config_fields
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing = required - set(payload)
        unknown = set(payload) - expected
        if missing or unknown:
            raise ValueError(
                f"invalid Stage1Config keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        values = dict(payload)
        announcements = values.get("announcements", [])
        if not isinstance(announcements, list):
            raise TypeError("announcements must be a JSON array")
        for item in announcements:
            if not isinstance(item, dict) or set(item) != {
                "day", "fundamental_delta", "source"
            }:
                raise ValueError("invalid announcement keys; expected day, fundamental_delta, source")
        values["announcements"] = tuple(Announcement(**item) for item in announcements)
        if not isinstance(values["strategy_shares"], dict):
            raise TypeError("strategy_shares must be a JSON object")
        values["strategy_shares"] = dict(values["strategy_shares"])
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
