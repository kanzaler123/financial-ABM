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
        if population.cash.ndim != 1 or population.positions.shape != population.cash.shape:
            raise ValueError("population cash and positions must be matching vectors")
        for name, values in (("cash", population.cash), ("positions", population.positions),
                             ("submitted_orders", submitted_orders)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if np.any(population.cash < 0):
            raise ValueError("population cash must be nonnegative")
        if submitted_orders.shape != population.cash.shape:
            raise ValueError("submitted_orders shape must match the population")
        if execution_price <= 0 or not np.isfinite(execution_price):
            raise ValueError("execution_price must be finite and positive")
        if market_maker_cash < 0 or not np.isfinite(market_maker_cash):
            raise ValueError("market-maker cash must be finite and nonnegative")
        if not np.isfinite(market_maker_inventory):
            raise ValueError("market-maker inventory must be finite")

        executed = submitted_orders.astype(np.float64, copy=True)
        if minimum_positions is None:
            minimum_positions = np.zeros_like(population.positions)
        if minimum_positions.shape != population.positions.shape:
            raise ValueError("minimum_positions shape must match the population")
        if not np.all(np.isfinite(minimum_positions)):
            raise ValueError("minimum_positions must be finite")

        # Margin call: whatever the submitted direction, no position may end
        # the day below the leverage floor. This also clips sales that would
        # push a position through the floor.
        executed = np.maximum(
            executed,
            minimum_positions - population.positions,
        )
        mandatory_covers = np.maximum(
            minimum_positions - population.positions,
            0.0,
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
        # voluntary part of the batch pro rata. Margin covers are excluded:
        # solvency enforcement takes precedence over counterparty rationing.
        # Net sales beyond cash are absorbed by short-selling, so inventory
        # imposes no execution bound.
        discretionary = executed - mandatory_covers
        discretionary_cash_delta = float(
            (
                discretionary * execution_price
                + np.abs(discretionary)
                * execution_price
                * self.transaction_cost_rate
            ).sum()
        )
        # Mandatory buybacks pay the counterparty too. Include their cash
        # proceeds before rationing voluntary net sales in this same batch.
        available_counterparty_cash = market_maker_cash + float(
            mandatory_covers.sum() * execution_price
            * (1.0 + self.transaction_cost_rate)
        )
        scale = 1.0
        if discretionary_cash_delta < -available_counterparty_cash:
            scale = available_counterparty_cash / -discretionary_cash_delta
        if scale < 1.0:
            executed = mandatory_covers + scale * discretionary

        transaction_costs = (
            np.abs(executed) * execution_price * self.transaction_cost_rate
        )
        trader_cash_delta = -executed * execution_price - transaction_costs
        cash_after = population.cash + trader_cash_delta
        positions_after = population.positions + executed
        market_maker_cash_after = market_maker_cash - float(trader_cash_delta.sum())
        market_maker_inventory_after = market_maker_inventory - float(executed.sum())
        # Compute and validate both sides before committing either account.
        if not (
            np.all(np.isfinite(cash_after))
            and np.all(np.isfinite(positions_after))
            and np.all(np.isfinite(transaction_costs))
            and np.isfinite(market_maker_cash_after)
            and np.isfinite(market_maker_inventory_after)
        ):
            raise ValueError("settlement produced non-finite balances")
        if np.any(cash_after < -1e-9) or market_maker_cash_after < -1e-9:
            raise RuntimeError("settlement would create negative cash")
        population.cash[:] = cash_after
        population.positions[:] = positions_after

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
