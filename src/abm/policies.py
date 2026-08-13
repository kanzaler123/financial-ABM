"""Vectorized target-position policies with asynchronous order arrival."""

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
class PolicyDiagnostics:
    value_information_updates: int = 0
    mean_subjective_value: float = 0.0
    mean_absolute_pending_information: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketObservation:
    price: float
    fundamental_value: float
    price_history: FloatArray
    public_news_return: float = 0.0
    realized_volatility: float = 0.0
    spread: float = 0.0
    depth: float = 0.0


class BatchPolicy(Protocol):
    def act(
        self,
        observation: MarketObservation,
        population: TraderPopulation,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Return one signed order quantity per trader."""


@dataclass(slots=True)
class RuleBasedPolicy:
    config: Stage1Config
    activity_state: float = 0.0
    common_signal_state: FloatArray | None = None
    diagnostics: PolicyDiagnostics = PolicyDiagnostics()

    def act(
        self,
        observation: MarketObservation,
        population: TraderPopulation,
        rng: np.random.Generator,
    ) -> FloatArray:
        price = observation.price
        wealth = population.wealth(price)
        maximum_position = np.minimum(
            self.config.target_position_fraction * wealth / price,
            self.config.position_cap,
        )
        maximum_rebalance = (
            self.config.max_order_fraction * wealth / price
        )
        sensitivity_multiplier = 1.0 / population.risk_aversion

        if (
            population.subjective_values is None
            or population.information_response_multipliers is None
            or population.pending_information is None
            or population.value_update_probabilities is None
            or population.trend_short_lookbacks is None
            or population.trend_long_lookbacks is None
            or population.base_activity_rates is None
            or population.liquidity_needs is None
            or population.reference_positions is None
            or population.desired_positions is None
        ):
            raise RuntimeError(
                "policy state arrays are missing from the trader population"
            )

        value_mask = population.strategies == VALUE_STRATEGY
        population.pending_information[value_mask] += (
            observation.public_news_return
        )
        update_draws = rng.random(population.size)
        value_updates = value_mask & (
            update_draws < population.value_update_probabilities
        )
        population.subjective_values[value_updates] *= np.exp(
            population.information_response_multipliers[value_updates]
            * population.pending_information[value_updates]
        )
        population.pending_information[value_updates] = 0.0
        value_subjective_values = population.subjective_values[value_mask]
        value_pending_information = population.pending_information[value_mask]
        self.diagnostics = PolicyDiagnostics(
            value_information_updates=int(np.count_nonzero(value_updates)),
            mean_subjective_value=(
                float(value_subjective_values.mean())
                if value_subjective_values.size
                else 0.0
            ),
            mean_absolute_pending_information=(
                float(np.mean(np.abs(value_pending_information)))
                if value_pending_information.size
                else 0.0
            ),
        )
        relative_mispricing = (
            population.subjective_values - price
        ) / price
        value_signal = (
            self.config.value_sensitivity
            * sensitivity_multiplier
            * relative_mispricing
        )
        value_signal[
            np.abs(relative_mispricing) < self.config.value_no_trade_band
        ] = 0.0

        history = observation.price_history
        history_size = history.size
        short_lookbacks = np.minimum(
            population.trend_short_lookbacks,
            history_size,
        )
        long_lookbacks = np.minimum(
            population.trend_long_lookbacks,
            history_size,
        )
        cumulative_prices = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(history, dtype=np.float64),
            )
        )
        short_starts = history_size - short_lookbacks
        long_starts = history_size - long_lookbacks
        short_means = (
            cumulative_prices[-1] - cumulative_prices[short_starts]
        ) / short_lookbacks
        long_means = (
            cumulative_prices[-1] - cumulative_prices[long_starts]
        ) / long_lookbacks
        volatility_scale = max(
            observation.realized_volatility,
            self.config.signal_volatility_floor,
        )
        normalized_trend = np.log(short_means / long_means) / (
            volatility_scale
            * np.sqrt(np.maximum(long_lookbacks, 1))
        )
        trend_signal = (
            self.config.trend_sensitivity
            * sensitivity_multiplier
            * normalized_trend
        )

        volatility_reference = max(
            self.config.fundamental_volatility,
            self.config.signal_volatility_floor,
        )
        flow_volatility_ratio = (
            observation.realized_volatility / volatility_reference
        )
        flow_volatility_scale = flow_volatility_ratio ** (
            self.config.demand_volatility_exponent
        )
        population.liquidity_needs[:] = (
            self.config.liquidity_need_persistence
            * population.liquidity_needs
            + self.config.liquidity_need_scale
            * flow_volatility_scale
            * rng.standard_normal(population.size)
        )

        # Draw for every trader so the random stream is independent of current
        # strategy shares.
        independent_innovations = rng.standard_normal(population.size)
        if self.common_signal_state is None:
            self.common_signal_state = np.zeros(
                len(STRATEGY_NAMES), dtype=np.float64
            )
        persistence = self.config.common_signal_persistence
        self.common_signal_state = (
            persistence * self.common_signal_state
            + np.sqrt(1.0 - persistence**2)
            * rng.standard_normal(len(STRATEGY_NAMES))
        )
        common_by_strategy = self.common_signal_state
        correlation = self.config.common_signal_correlation
        innovations = (
            np.sqrt(1.0 - correlation) * independent_innovations
            + np.sqrt(correlation)
            * common_by_strategy[population.strategies]
        )
        noise_signal = (
            self.config.noise_scale * innovations * flow_volatility_scale
        )
        idiosyncratic_signal = (
            self.config.idiosyncratic_signal_scale
            * innovations
            * flow_volatility_scale
        )

        signals = np.zeros(population.size, dtype=np.float64)
        trend_mask = population.strategies == TREND_STRATEGY
        signals[value_mask] = (
            value_signal[value_mask] + idiosyncratic_signal[value_mask]
        )
        signals[trend_mask] = (
            trend_signal[trend_mask] + idiosyncratic_signal[trend_mask]
        )
        noise_mask = population.strategies == NOISE_STRATEGY
        signals[noise_mask] = noise_signal[noise_mask]
        signals += population.liquidity_needs

        raw_target_positions = np.clip(
            population.reference_positions
            + maximum_position * np.tanh(signals),
            -maximum_position,
            maximum_position,
        )
        no_trade_value_agents = value_mask & (
            np.abs(relative_mispricing) < self.config.value_no_trade_band
        )
        raw_target_positions[no_trade_value_agents] = population.positions[
            no_trade_value_agents
        ]
        adjustment_rates = np.empty(population.size, dtype=np.float64)
        adjustment_rates[value_mask] = self.config.value_target_adjustment
        adjustment_rates[trend_mask] = self.config.trend_target_adjustment
        adjustment_rates[noise_mask] = self.config.noise_target_adjustment
        population.desired_positions += adjustment_rates * (
            raw_target_positions - population.desired_positions
        )
        submitted = np.clip(
            population.desired_positions - population.positions,
            -maximum_rebalance,
            maximum_rebalance,
        )

        volatility_reference = max(
            self.config.fundamental_volatility,
            self.config.signal_volatility_floor,
        )
        volatility_stress = min(
            observation.realized_volatility / volatility_reference,
            5.0,
        )
        self.activity_state = (
            self.config.activity_persistence * self.activity_state
            + self.config.activity_shock_scale * float(rng.standard_normal())
        )
        base_activity = population.base_activity_rates
        base_logits = np.log(base_activity / (1.0 - base_activity))
        activity_logits = (
            base_logits
            + self.activity_state
            + self.config.activity_signal_sensitivity
            * np.abs(np.tanh(signals))
            + self.config.activity_volatility_sensitivity
            * volatility_stress
            + np.abs(population.liquidity_needs)
        )
        activity_rates = 1.0 / (1.0 + np.exp(-activity_logits))
        active = rng.random(population.size) < activity_rates
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
        minimum_positions = np.maximum(
            -maximum_short_positions,
            -self.config.position_cap,
        )
        available_sales = population.positions - minimum_positions
        submitted = np.where(
            submitted > 0,
            np.minimum(submitted, affordable_buys),
            np.maximum(submitted, -available_sales),
        )
        return submitted.astype(np.float64, copy=False)
