"""Single-entry settlement against a market-maker counterparty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .population import TraderPopulation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SettlementResult:
    executed_orders: FloatArray
    transaction_costs: FloatArray
    market_maker_cash: float
    market_maker_inventory: float


@dataclass(frozen=True, slots=True)
class SettlementEngine:
    transaction_cost_rate: float

    def __post_init__(self) -> None:
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be in [0, 1)")

    def settle(
        self,
        *,
        population: TraderPopulation,
        submitted_orders: FloatArray,
        execution_price: float,
        market_maker_cash: float,
        market_maker_inventory: float,
        minimum_positions: FloatArray | None = None,
    ) -> SettlementResult:
        if submitted_orders.shape != population.cash.shape:
            raise ValueError("submitted_orders shape must match the population")
        if execution_price <= 0 or not np.isfinite(execution_price):
            raise ValueError("execution_price must be finite and positive")
        if market_maker_cash < 0:
            raise ValueError("market-maker cash must be nonnegative")
        if not np.isfinite(market_maker_inventory):
            raise ValueError("market-maker inventory must be finite")

        executed = submitted_orders.astype(np.float64, copy=True)
        if minimum_positions is None:
            minimum_positions = np.zeros_like(population.positions)
        if minimum_positions.shape != population.positions.shape:
            raise ValueError("minimum_positions shape must match the population")

        # Margin call: whatever the submitted direction, no position may end
        # the day below the leverage floor. This also clips sales that would
        # push a position through the floor.
        executed = np.maximum(
            executed,
            minimum_positions - population.positions,
        )
        buys = executed > 0
        sells = executed < 0

        affordable = population.cash / (
            execution_price * (1.0 + self.transaction_cost_rate)
        )
        executed[buys] = np.minimum(executed[buys], affordable[buys])
        available_sales = population.positions - minimum_positions
        executed[sells] = np.maximum(executed[sells], -available_sales[sells])
        if np.any(
            population.positions + executed < minimum_positions - 1e-9
        ):
            raise RuntimeError(
                "margin call exceeds available trader cash"
            )

        # If the counterparty runs out of cash for net purchases, scale the
        # whole batch pro rata. A single factor preserves the synchronous
        # batch and avoids one-sided rescaling invalidating the other
        # resource constraint. Net sales are absorbed by the market maker
        # short-selling, so inventory imposes no execution bound.
        market_maker_cash_delta = float(
            (
                executed * execution_price
                + np.abs(executed)
                * execution_price
                * self.transaction_cost_rate
            ).sum()
        )
        scale = 1.0
        if market_maker_cash_delta < -market_maker_cash:
            scale = min(scale, market_maker_cash / -market_maker_cash_delta)
        if scale < 1.0:
            executed *= scale

        transaction_costs = (
            np.abs(executed) * execution_price * self.transaction_cost_rate
        )
        trader_cash_delta = -executed * execution_price - transaction_costs
        population.cash += trader_cash_delta
        population.positions += executed

        market_maker_cash_after = market_maker_cash - float(trader_cash_delta.sum())
        market_maker_inventory_after = market_maker_inventory - float(executed.sum())

        # Remove harmless floating-point crumbs before invariant checks.
        population.cash[np.abs(population.cash) < 1e-12] = 0.0
        population.positions[np.abs(population.positions) < 1e-12] = 0.0
        if abs(market_maker_cash_after) < 1e-9:
            market_maker_cash_after = 0.0
        if abs(market_maker_inventory_after) < 1e-9:
            market_maker_inventory_after = 0.0

        return SettlementResult(
            executed_orders=executed,
            transaction_costs=transaction_costs,
            market_maker_cash=market_maker_cash_after,
            market_maker_inventory=market_maker_inventory_after,
        )
