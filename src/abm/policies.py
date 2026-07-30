"""Vectorized value, trend, and noise trading policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import STRATEGY_NAMES, Stage1Config
from .population import (
    NOISE_STRATEGY,
    TREND_STRATEGY,
    VALUE_STRATEGY,
    TraderPopulation,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarketObservation:
    price: float
    fundamental_value: float
    price_history: FloatArray


class BatchPolicy(Protocol):
    def act(
        self,
        observation: MarketObservation,
        population: TraderPopulation,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Return one signed order quantity per trader."""


@dataclass(frozen=True, slots=True)
class RuleBasedPolicy:
    config: Stage1Config

    def act(
        self,
        observation: MarketObservation,
        population: TraderPopulation,
        rng: np.random.Generator,
    ) -> FloatArray:
        price = observation.price
        wealth = population.wealth(price)
        capacity = self.config.max_order_fraction * wealth / price
        sensitivity_multiplier = 1.0 / population.risk_aversion

        value_signal = self.config.value_sensitivity * sensitivity_multiplier * (
            observation.fundamental_value - price
        ) / price
        available_history = observation.price_history.size - 1
        individual_lookbacks = np.clip(
            np.rint(
                self.config.trend_lookback * population.risk_aversion
            ).astype(np.int64),
            1,
            max(1, available_history),
        )
        if available_history == 0:
            lookback_prices = np.full(population.size, price, dtype=np.float64)
        else:
            lookback_prices = observation.price_history[
                observation.price_history.size - 1 - individual_lookbacks
            ]
        trend_signal = (
            self.config.trend_sensitivity
            * sensitivity_multiplier
            * (price / lookback_prices - 1.0)
        )
        # Draw for every trader every day so the random stream does not depend
        # on current strategy shares.
        independent_innovations = rng.standard_normal(population.size)
        common_by_strategy = rng.standard_normal(len(STRATEGY_NAMES))
        correlation = self.config.common_signal_correlation
        innovations = (
            np.sqrt(1.0 - correlation) * independent_innovations
            + np.sqrt(correlation)
            * common_by_strategy[population.strategies]
        )
        noise_signal = self.config.noise_scale * innovations
        idiosyncratic_signal = (
            self.config.idiosyncratic_signal_scale * innovations
        )

        signals = np.zeros(population.size, dtype=np.float64)
        value_mask = population.strategies == VALUE_STRATEGY
        trend_mask = population.strategies == TREND_STRATEGY
        signals[value_mask] = (
            value_signal[value_mask] + idiosyncratic_signal[value_mask]
        )
        signals[trend_mask] = (
            trend_signal[trend_mask] + idiosyncratic_signal[trend_mask]
        )
        noise_mask = population.strategies == NOISE_STRATEGY
        signals[noise_mask] = noise_signal[noise_mask]
        submitted = capacity * np.tanh(signals)

        if observation.price_history.size > 1:
            recent_prices = observation.price_history[-21:]
            recent_returns = np.diff(np.log(recent_prices))
            realized_volatility = float(np.std(recent_returns, ddof=0))
        else:
            realized_volatility = 0.0
        volatility_reference = max(
            self.config.fundamental_volatility,
            1e-6,
        )
        volatility_stress = realized_volatility / volatility_reference
        activity_rate = min(
            1.0,
            self.config.base_activity_rate
            * (
                1.0
                + self.config.activity_volatility_sensitivity
                * volatility_stress
            ),
        )
        # Daily order arrival is asynchronous. Volatility can raise
        # participation through attention and risk-management responses.
        active = rng.random(population.size) < activity_rate
        submitted = np.where(active, submitted, 0.0)

        # The price cap is included in the affordability bound, so a permitted
        # buy remains affordable at the eventual execution price.
        maximum_execution_price = price * np.exp(self.config.max_log_return)
        affordable_buys = population.cash / (
            maximum_execution_price * (1.0 + self.config.transaction_cost_rate)
        )
        maximum_short_positions = (
            self.config.max_short_leverage
            * wealth
            / maximum_execution_price
        )
        minimum_positions = -maximum_short_positions
        available_sales = population.positions - minimum_positions
        submitted = np.where(
            submitted > 0,
            np.minimum(submitted, affordable_buys),
            np.maximum(submitted, -available_sales),
        )
        return submitted.astype(np.float64, copy=False)
