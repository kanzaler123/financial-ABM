"""Validated configuration of the multi-asset market.

Rates and volatilities are annual unless a name says otherwise; the market
steps in trading days (``TRADING_DAYS_PER_YEAR`` per year).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

TRADING_DAYS_PER_YEAR = 252
IMITATION_RULES = ("type_mean", "pairwise")
LEARNING_CONTEXTS = ("exposure", "signal_tercile", "improvement_stock")


@dataclass(frozen=True, slots=True)
class MarketConfig:
    seed: int = 20261012
    n_investors: int = 1000
    n_firms: int = 200
    trading_days: int = 3500
    burn_in_days: int = 1000

    # Firms: size distribution and fundamentals.
    size_class_shares: tuple[float, float, float] = (0.1, 0.3, 0.6)
    intrinsic_value_log_sd: float = 1.0
    cash_flow_yield: float = 0.05
    fundamental_drift: float = 0.04
    fundamental_volatility: float = 0.159
    fundamental_common_correlation: float = 0.4

    # Investors.
    investor_wealth_log_sd: float = 1.0
    investor_float_share: float = 0.8
    # Warm start near the stationary state of the default market. Starting
    # from fundamental-only volatility (about 10% for the index) triggers an
    # immediate liquidity squeeze and a year-long crash during burn-in.
    initial_equity_share: float = 0.7
    initial_market_volatility: float = 0.18
    initial_asset_volatility: float = 0.27
    max_equity_share: float = 1.0
    risk_aversion_mean: float = 3.0
    risk_aversion_sd: float = 0.5
    risk_aversion_min: float = 0.5
    decision_interval_median: float = 20.0
    decision_interval_log_sd: float = 0.5
    decision_interval_max: int = 120
    # Demand must be inelastic enough for the price loop to be stable: the
    # daily feedback gain is roughly deciding share * rebalance_speed * float
    # * (1 + intensity * (yield + reversion)) / liquidity_fraction. Defaults
    # give per-asset share elasticities of about 1.4, in the range reported
    # by Koijen and Yogo (2019), and a gain far below one.
    choice_intensity_median: float = 2.0
    choice_intensity_log_sd: float = 0.5
    rebalance_speed: float = 0.5
    risk_free_rate: float = 0.025
    mispricing_reversion: float = 0.15
    variance_ewma_decay: float = 0.94
    # Equity/cash allocation uses a slower market-variance estimate: with the
    # fast 0.94 estimate one volatile day can cut every Merton share from ~1
    # to ~0.1 and trigger a market-wide fire sale that never recovers.
    allocation_variance_decay: float = 0.99

    # Price formation, per asset. Market makers move quotes with public
    # news about intrinsic value; without this, prices only follow value
    # through slow investor rebalancing and returns become strongly
    # autocorrelated (lag-1 autocorrelation of about 0.44 in testing).
    news_pass_through: float = 0.9
    price_impact: float = 1.0
    liquidity_fraction: float = 0.3
    liquidity_volatility_sensitivity: float = 3.03
    liquidity_stress_threshold: float = 0.8
    minimum_liquidity_fraction: float = 0.3
    depth_resilience: float = 0.15
    reference_variance_decay: float = 0.995
    # Only part of the flow impact is permanent; the rest follows the order
    # flow surprise and decays (the Stage 1 propagator form). Persistent order
    # flow would otherwise turn into persistent price trends.
    permanent_impact_fraction: float = 0.486
    transient_impact_decay: float = 0.974
    order_flow_memory: float = 0.3
    max_log_return: float = 0.3
    transaction_cost_rate: float = 0.001

    # Aggregate noise-trader account; noise_flow_scale = 0 switches it off.
    # Seven defaults (fundamental volatility, liquidity sensitivity, impact
    # permanence and decay, noise scale, correlation and volatility exponent)
    # come from an MSM fit to French 49 training moments; see
    # reports/disclosure_m1_20261011/. Fresh-seed J = 41: eight of nine
    # moments within 1.2 standard errors, but lag-1 return autocorrelation is
    # too negative (-0.18 vs -0.03, t = -6). noise_flow_scale sits on its
    # lower bound.
    noise_flow_scale: float = 0.002
    noise_common_correlation: float = 0.80
    noise_persistence: float = 0.5
    noise_inventory_reversion: float = 0.05
    noise_float_share: float = 0.05
    # Participation feedback (Stage 1's demand-volatility channel): noise
    # flow scales with (asset volatility / reference volatility) ** exponent,
    # the ratio capped at four. Zero switches the feedback off.
    noise_volatility_exponent: float = 1.77
    market_maker_cash_multiple: float = 1.0
    # Daily log-price skew per unit of market-maker inventory gap (target
    # minus inventory, as a fraction of shares outstanding). Without it the
    # market maker is an unlimited counterparty and drifts deep short.
    market_maker_inventory_skew: float = 0.01

    # Investor types. Impact investors (II) add taste * signal to utility,
    # where the signal uses the domain's fixed initial standard deviation;
    # taste is an annual-return equivalent per initial standard deviation. Zero
    # initial II share leaves the bare market unchanged.
    initial_impact_share: float = 0.0
    taste_median: float = 0.02
    taste_log_sd: float = 0.5
    # Proportional imitation every imitation_interval days (see game.py).
    # Intensity multiplies the difference in mean wealth growth over the
    # window; growth differences are fractions of a percent to a few percent,
    # so intensity must be of order 100 for switching to respond. Only a
    # fraction of investors reconsiders each time. "type_mean" compares the
    # two type means directly; "pairwise" first samples one other investor
    # and imitates only a partner of the other type (Schlag 1998).
    imitation_interval: int = 60
    imitation_intensity: float = 100.0
    imitation_update_fraction: float = 0.1
    imitation_rule: str = "type_mean"

    # Firms decide on the attribute every firm_decision_interval days.
    firm_decision_interval: int = 60
    firm_policy: str = "npv"
    firm_action_fractions: tuple[float, ...] = (.01,)
    # Q-learning only. The first state feature is the financing exposure share
    # ("exposure") or the tercile of the firm's own public signal among all
    # firms ("signal_tercile"), which carries delayed real improvements. With
    # a positive power w the step size of a table entry after n updates is
    # n ** -w (Even-Dar and Mansour 2003); zero keeps the constant rate.
    firm_learning_context: str = "exposure"
    firm_learning_rate_power: float = 0.0
    # Shared-grid rules only (npv_discrete, q_learning): a positive value makes
    # a candidate illegal when its resource cost exceeds this fraction of the
    # current market cap. Zero keeps the value-anchor rule alone.
    firm_cost_budget_fraction: float = 0.0
    # Fixed financing exposure, in units of initial market cap. These are
    # exploratory contract assumptions, not estimates from observed scores.
    financing_exposure_fraction: float = 0.5
    financing_spread: float = 0.05
    financing_repricing_interval: int = 60
    financing_horizon_days: int = 1260
    firm_discount_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.firm_policy not in ("npv", "npv_discrete", "q_learning"):
            raise ValueError("firm_policy must be 'npv', 'npv_discrete' or 'q_learning'")
        if self.imitation_rule not in IMITATION_RULES:
            raise ValueError(f"imitation_rule must be one of {IMITATION_RULES}")
        if self.firm_learning_context not in LEARNING_CONTEXTS:
            raise ValueError(f"firm_learning_context must be one of {LEARNING_CONTEXTS}")
        if not 0 <= self.firm_learning_rate_power <= 1:
            raise ValueError("firm_learning_rate_power must be in [0, 1]")
        if not 0 <= self.firm_cost_budget_fraction <= 1:
            raise ValueError("firm_cost_budget_fraction must be in [0, 1]")
        fractions = tuple(self.firm_action_fractions)
        if (not fractions or any(not math.isfinite(value) or not 0 < value <= 1 for value in fractions)
                or any(left >= right for left, right in zip(fractions, fractions[1:]))):
            raise ValueError("firm_action_fractions must be finite, strictly increasing and in (0,1]")
        object.__setattr__(self, "firm_action_fractions", fractions)
        for name in ("imitation_interval", "firm_decision_interval", "financing_repricing_interval", "financing_horizon_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("n_investors", "n_firms", "trading_days", "decision_interval_max"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.burn_in_days) is not int or not 0 <= self.burn_in_days < self.trading_days:
            raise ValueError("burn_in_days must be in [0, trading_days)")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        shares = self.size_class_shares
        if len(shares) != 3 or min(shares) <= 0 or abs(sum(shares) - 1.0) > 1e-9:
            raise ValueError("size_class_shares must be three positive shares summing to one")
        for name in (
            "fundamental_common_correlation", "news_pass_through", "investor_float_share", "initial_equity_share",
            "max_equity_share", "rebalance_speed", "variance_ewma_decay", "allocation_variance_decay",
            "liquidity_fraction", "minimum_liquidity_fraction", "depth_resilience",
            "reference_variance_decay", "noise_common_correlation", "noise_persistence",
            "permanent_impact_fraction", "transient_impact_decay", "order_flow_memory",
            "initial_impact_share", "imitation_update_fraction",
            "noise_inventory_reversion", "noise_float_share",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.investor_float_share + self.noise_float_share > 1:
            raise ValueError("investor and noise float shares must not exceed one")
        if self.liquidity_fraction <= 0 or self.depth_resilience <= 0:
            raise ValueError("liquidity_fraction and depth_resilience must be positive")
        for name in (
            "intrinsic_value_log_sd", "cash_flow_yield", "fundamental_volatility",
            "initial_market_volatility", "initial_asset_volatility",
            "investor_wealth_log_sd", "risk_aversion_sd", "decision_interval_log_sd",
            "choice_intensity_log_sd", "mispricing_reversion", "price_impact",
            "liquidity_volatility_sensitivity", "liquidity_stress_threshold",
            "transaction_cost_rate", "noise_flow_scale", "market_maker_cash_multiple",
            "market_maker_inventory_skew", "noise_volatility_exponent",
            "taste_median", "taste_log_sd", "imitation_intensity",
            "financing_exposure_fraction", "financing_spread", "firm_discount_rate",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("risk_aversion_min", "decision_interval_median", "choice_intensity_median", "max_log_return"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.transaction_cost_rate >= 1:
            raise ValueError("transaction_cost_rate must be below one")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["size_class_shares"] = list(self.size_class_shares)
        if self.firm_action_fractions == (.01,):
            payload.pop("firm_action_fractions")
        else:
            payload["firm_action_fractions"] = list(self.firm_action_fractions)
        # Defaults are omitted so that identities from before these options stay unchanged.
        if self.imitation_rule == "type_mean":
            payload.pop("imitation_rule")
        if self.firm_learning_context == "exposure":
            payload.pop("firm_learning_context")
        if self.firm_learning_rate_power == 0.0:
            payload.pop("firm_learning_rate_power")
        if self.firm_cost_budget_fraction == 0.0:
            payload.pop("firm_cost_budget_fraction")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown market config keys: {sorted(unknown)}")
        values = dict(payload)
        if "size_class_shares" in values:
            values["size_class_shares"] = tuple(float(value) for value in values["size_class_shares"])
        if "firm_action_fractions" in values:
            values["firm_action_fractions"] = tuple(float(value) for value in values["firm_action_fractions"])
        return cls(**values)


def load_market_config(path: str | Path) -> MarketConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("market config root must be a JSON object")
    return MarketConfig.from_dict(payload)
