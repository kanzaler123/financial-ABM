"""Array-backed trader population for deterministic batch simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import STRATEGY_NAMES, Stage1Config

FloatArray = NDArray[np.float64]
StrategyArray = NDArray[np.int8]

VALUE_STRATEGY = 0
TREND_STRATEGY = 1
NOISE_STRATEGY = 2
STRATEGY_CODES = np.array(
    [VALUE_STRATEGY, TREND_STRATEGY, NOISE_STRATEGY], dtype=np.int8
)


def strategy_counts(strategy_shares: dict[str, float], population_size: int) -> NDArray[np.int64]:
    """Convert proportions into exact integer counts using largest remainders."""
    raw = np.array(
        [strategy_shares[name] * population_size for name in STRATEGY_NAMES],
        dtype=np.float64,
    )
    counts = np.floor(raw).astype(np.int64)
    remainder = population_size - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


@dataclass(slots=True)
class TraderPopulation:
    cash: FloatArray
    positions: FloatArray
    strategies: StrategyArray
    risk_aversion: FloatArray
    subjective_values: FloatArray | None = None
    information_response_multipliers: FloatArray | None = None
    pending_information: FloatArray | None = None
    value_update_probabilities: FloatArray | None = None
    trend_short_lookbacks: NDArray[np.int64] | None = None
    trend_long_lookbacks: NDArray[np.int64] | None = None
    base_activity_rates: FloatArray | None = None
    liquidity_needs: FloatArray | None = None
    reference_positions: FloatArray | None = None
    desired_positions: FloatArray | None = None
    reference_wealth: FloatArray | None = None

    @classmethod
    def initialize(
        cls,
        config: Stage1Config,
        rng: np.random.Generator,
    ) -> "TraderPopulation":
        counts = strategy_counts(config.strategy_shares, config.population_size)
        strategies = np.repeat(STRATEGY_CODES, counts).astype(np.int8, copy=False)
        rng.shuffle(strategies)
        if config.initial_wealth_dispersion > 0:
            wealth_multipliers = rng.lognormal(
                mean=-0.5 * config.initial_wealth_dispersion**2,
                sigma=config.initial_wealth_dispersion,
                size=config.population_size,
            )
            # Preserve the configured aggregate endowment exactly while
            # representing the concentrated size distribution of real markets.
            wealth_multipliers /= float(wealth_multipliers.mean())
        else:
            wealth_multipliers = np.ones(
                config.population_size,
                dtype=np.float64,
            )
        initial_value_innovations = rng.standard_normal(config.population_size)
        subjective_values = config.initial_fundamental * np.exp(
            -0.5 * config.initial_subjective_value_dispersion**2
            + config.initial_subjective_value_dispersion
            * initial_value_innovations
        )
        response_innovations = rng.standard_normal(config.population_size)
        information_response_multipliers = np.clip(
            1.0
            + config.information_response_dispersion
            * response_innovations,
            0.25,
            1.75,
        )
        value_update_probabilities = np.clip(
            config.value_update_rate
            * rng.uniform(0.5, 1.5, size=config.population_size),
            0.0,
            1.0,
        )
        trend_short_lookbacks = rng.integers(
            config.trend_short_lookback_min,
            config.trend_short_lookback_max + 1,
            size=config.population_size,
            dtype=np.int64,
        )
        trend_long_lookbacks = rng.integers(
            config.trend_long_lookback_min,
            config.trend_long_lookback_max + 1,
            size=config.population_size,
            dtype=np.int64,
        )
        activity_multipliers = rng.lognormal(
            mean=-0.5 * config.activity_rate_dispersion**2,
            sigma=config.activity_rate_dispersion,
            size=config.population_size,
        )
        base_activity_rates = np.clip(
            config.base_activity_rate * activity_multipliers,
            1e-6,
            1.0 - 1e-6,
        )
        initial_positions = (
            config.initial_agent_position * wealth_multipliers
        ).astype(np.float64, copy=False)
        return cls(
            cash=(
                config.initial_agent_cash * wealth_multipliers
            ).astype(np.float64, copy=False),
            positions=initial_positions.copy(),
            strategies=strategies,
            # Stable individual heterogeneity prevents an entire strategy
            # cohort from behaving like one representative trader.
            risk_aversion=rng.uniform(0.5, 1.5, size=config.population_size),
            subjective_values=subjective_values.astype(
                np.float64, copy=False
            ),
            information_response_multipliers=(
                information_response_multipliers.astype(
                    np.float64, copy=False
                )
            ),
            pending_information=np.zeros(
                config.population_size, dtype=np.float64
            ),
            value_update_probabilities=value_update_probabilities.astype(
                np.float64, copy=False
            ),
            trend_short_lookbacks=trend_short_lookbacks,
            trend_long_lookbacks=trend_long_lookbacks,
            base_activity_rates=base_activity_rates.astype(
                np.float64, copy=False
            ),
            liquidity_needs=np.zeros(
                config.population_size, dtype=np.float64
            ),
            reference_positions=initial_positions.copy(),
            desired_positions=initial_positions.copy(),
            # Order sizing is anchored to the initial stake, so the wealth
            # random-walk level does not inject long memory into order flow.
            reference_wealth=(
                config.initial_agent_cash * wealth_multipliers
                + config.initial_agent_position
                * wealth_multipliers
                * config.initial_price
            ).astype(np.float64, copy=False),
        )

    @property
    def size(self) -> int:
        return int(self.cash.size)

    def wealth(self, price: float) -> FloatArray:
        return self.cash + self.positions * price

    def counts(self) -> NDArray[np.int64]:
        return np.bincount(self.strategies, minlength=len(STRATEGY_NAMES)).astype(
            np.int64,
            copy=False,
        )

    def validate(
        self,
        *,
        tolerance: float = 1e-9,
        allow_short: bool = False,
        price: float | None = None,
    ) -> None:
        if not (
            self.cash.shape
            == self.positions.shape
            == self.strategies.shape
            == self.risk_aversion.shape
        ):
            raise RuntimeError("population arrays must have identical shapes")
        if not np.all(np.isfinite(self.cash)) or not np.all(
            np.isfinite(self.positions)
        ):
            raise RuntimeError("population contains non-finite cash or positions")
        if float(self.cash.min()) < -tolerance:
            raise RuntimeError("population contains negative cash")
        if not allow_short and float(self.positions.min()) < -tolerance:
            raise RuntimeError("population contains illegal short positions")
        if price is not None and float(self.wealth(price).min()) < -tolerance:
            raise RuntimeError("population contains negative marked-to-market wealth")
        if not np.all(np.isin(self.strategies, STRATEGY_CODES)):
            raise RuntimeError("population contains an unknown strategy code")
        optional_arrays = (
            self.subjective_values,
            self.information_response_multipliers,
            self.pending_information,
            self.value_update_probabilities,
            self.trend_short_lookbacks,
            self.trend_long_lookbacks,
            self.base_activity_rates,
            self.liquidity_needs,
            self.reference_positions,
            self.desired_positions,
            self.reference_wealth,
        )
        for array in optional_arrays:
            if array is not None and array.shape != self.cash.shape:
                raise RuntimeError(
                    "optional population arrays must match the population"
                )
