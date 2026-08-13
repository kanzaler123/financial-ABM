"""Aggregate quasi-order-book price formation for the Stage 1 market."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Stage1Config

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarketQuote:
    mid_price_before: float
    mid_price_after: float
    execution_price: float
    bid_price: float
    ask_price: float
    spread: float
    depth: float
    order_flow_imbalance: float
    order_flow_surprise: float
    public_news_impact: float
    permanent_impact: float
    transient_impact: float
    transient_impact_change: float
    demand_log_return: float
    price_cap_hit: bool


@dataclass(slots=True)
class QuasiOrderBook:
    """Stateful aggregate book with resilient depth and decaying impact.

    The book maps signed order flow into the mid price. A permanent component
    remains in the price, while a temporary component decays gradually rather
    than reversing completely on the following day.
    """

    config: Stage1Config
    mid_price: float
    depth: float
    transient_impact: float = 0.0
    expected_order_flow_imbalance: float = 0.0

    @classmethod
    def initialize(cls, config: Stage1Config) -> "QuasiOrderBook":
        return cls(
            config=config,
            mid_price=config.initial_price,
            depth=config.liquidity_scale,
        )

    def quote(
        self,
        *,
        submitted_orders: FloatArray,
        public_news_impact: float,
        realized_volatility: float,
    ) -> MarketQuote:
        if submitted_orders.ndim != 1:
            raise ValueError("submitted_orders must be one-dimensional")
        if not np.all(np.isfinite(submitted_orders)):
            raise ValueError("submitted_orders must be finite")
        if not np.isfinite(public_news_impact):
            raise ValueError("public_news_impact must be finite")
        if realized_volatility < 0 or not np.isfinite(
            realized_volatility
        ):
            raise ValueError(
                "realized_volatility must be finite and nonnegative"
            )

        volatility_reference = max(
            self.config.fundamental_volatility,
            self.config.signal_volatility_floor,
        )
        volatility_stress = max(
            realized_volatility / volatility_reference
            - self.config.liquidity_stress_threshold,
            0.0,
        )
        target_depth_fraction = max(
            self.config.minimum_liquidity_fraction,
            1.0
            / (
                1.0
                + self.config.liquidity_volatility_sensitivity
                * volatility_stress
            ),
        )
        gross_order_flow = float(np.abs(submitted_orders).sum())
        net_order_flow = float(submitted_orders.sum())
        base_depth = self.config.liquidity_scale
        target_depth = base_depth * target_depth_fraction
        self.depth += self.config.depth_resilience * (
            target_depth - self.depth
        )
        self.depth = max(
            self.depth,
            base_depth * self.config.minimum_liquidity_fraction,
            1e-12,
        )
        if gross_order_flow > 0:
            order_flow_imbalance = float(
                np.clip(net_order_flow / gross_order_flow, -1.0, 1.0)
            )
        else:
            order_flow_imbalance = 0.0
        order_flow_surprise = float(
            np.clip(
                order_flow_imbalance
                - self.expected_order_flow_imbalance,
                -2.0,
                2.0,
            )
        )
        self.expected_order_flow_imbalance = (
            self.config.order_flow_memory
            * self.expected_order_flow_imbalance
            + (1.0 - self.config.order_flow_memory)
            * order_flow_imbalance
        )
        participation_pressure = min(gross_order_flow / self.depth, 3.0)
        amplified_pressure = participation_pressure ** (
            self.config.participation_pressure_exponent
        )
        permanent_flow_impact = (
            self.config.price_impact
            * order_flow_imbalance
            * amplified_pressure
        )
        transient_flow_impact = (
            self.config.price_impact
            * order_flow_surprise
            * amplified_pressure
        )
        permanent_impact = (
            self.config.permanent_impact_fraction
            * permanent_flow_impact
        )
        transient_before = self.transient_impact
        self.transient_impact = (
            self.config.transient_impact_decay * transient_before
            + (1.0 - self.config.permanent_impact_fraction)
            * transient_flow_impact
        )
        transient_change = self.transient_impact - transient_before
        unconstrained_demand_return = permanent_impact + transient_change
        demand_log_return = float(
            np.clip(
                unconstrained_demand_return,
                -self.config.max_log_return,
                self.config.max_log_return,
            )
        )
        price_cap_hit = not np.isclose(
            demand_log_return,
            unconstrained_demand_return,
            rtol=0.0,
            atol=1e-15,
        )

        mid_price_before = self.mid_price
        self.mid_price *= float(
            np.exp(public_news_impact + demand_log_return)
        )
        depth_fraction = max(
            self.depth / base_depth,
            self.config.minimum_liquidity_fraction,
            1e-6,
        )
        spread_log = (
            self.config.base_spread_bps
            * 1e-4
            * (
                1.0
                + self.config.spread_volatility_sensitivity
                * realized_volatility
            )
            / depth_fraction
        )
        half_spread = 0.5 * spread_log
        bid_price = self.mid_price * float(np.exp(-half_spread))
        ask_price = self.mid_price * float(np.exp(half_spread))
        if net_order_flow > 0:
            execution_price = ask_price
        elif net_order_flow < 0:
            execution_price = bid_price
        else:
            execution_price = self.mid_price

        return MarketQuote(
            mid_price_before=mid_price_before,
            mid_price_after=self.mid_price,
            execution_price=execution_price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=ask_price - bid_price,
            depth=self.depth,
            order_flow_imbalance=order_flow_imbalance,
            order_flow_surprise=order_flow_surprise,
            public_news_impact=public_news_impact,
            permanent_impact=permanent_impact,
            transient_impact=self.transient_impact,
            transient_impact_change=transient_change,
            demand_log_return=demand_log_return,
            price_cap_hit=price_cap_hit,
        )
