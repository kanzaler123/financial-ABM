import numpy as np

from abm.population import TraderPopulation
from abm.settlement import SettlementEngine


def test_settlement_conserves_cash_and_shares() -> None:
    population = TraderPopulation(
        cash=np.array([1000.0, 1000.0]),
        positions=np.array([5.0, 5.0]),
        strategies=np.array([0, 1], dtype=np.int8),
        risk_aversion=np.ones(2),
    )
    market_maker_cash = 10_000.0
    market_maker_inventory = 100.0
    cash_before = float(population.cash.sum()) + market_maker_cash
    shares_before = float(population.positions.sum()) + market_maker_inventory

    result = SettlementEngine(transaction_cost_rate=0.01).settle(
        population=population,
        submitted_orders=np.array([2.0, -3.0]),
        execution_price=100.0,
        market_maker_cash=market_maker_cash,
        market_maker_inventory=market_maker_inventory,
    )

    assert np.isclose(
        float(population.cash.sum()) + result.market_maker_cash,
        cash_before,
    )
    assert np.isclose(
        float(population.positions.sum()) + result.market_maker_inventory,
        shares_before,
    )
    assert np.all(population.cash >= 0)
    assert np.all(population.positions >= 0)


def test_settlement_caps_orders_at_counterparty_resources() -> None:
    population = TraderPopulation(
        cash=np.array([100_000.0, 100_000.0]),
        positions=np.array([100.0, 100.0]),
        strategies=np.array([0, 1], dtype=np.int8),
        risk_aversion=np.ones(2),
    )
    engine = SettlementEngine(transaction_cost_rate=0.0)

    inventory_limited = engine.settle(
        population=population,
        submitted_orders=np.array([100.0, 100.0]),
        execution_price=100.0,
        market_maker_cash=10_000.0,
        market_maker_inventory=10.0,
    )

    assert np.isclose(inventory_limited.executed_orders.sum(), 10.0)
    assert np.isclose(inventory_limited.market_maker_inventory, 0.0)

    cash_limited = engine.settle(
        population=TraderPopulation(
            cash=np.array([1000.0, 1000.0]),
            positions=np.array([100.0, 100.0]),
            strategies=np.array([0, 1], dtype=np.int8),
            risk_aversion=np.ones(2),
        ),
        submitted_orders=np.array([-100.0, -100.0]),
        execution_price=100.0,
        market_maker_cash=500.0,
        market_maker_inventory=100.0,
    )

    assert np.isclose(cash_limited.executed_orders.sum(), -5.0)
    assert np.isclose(cash_limited.market_maker_cash, 0.0)


def test_settlement_allows_only_bounded_short_positions() -> None:
    population = TraderPopulation(
        cash=np.array([1000.0]),
        positions=np.array([1.0]),
        strategies=np.array([0], dtype=np.int8),
        risk_aversion=np.ones(1),
    )

    result = SettlementEngine(transaction_cost_rate=0.0).settle(
        population=population,
        submitted_orders=np.array([-10.0]),
        execution_price=100.0,
        market_maker_cash=10_000.0,
        market_maker_inventory=100.0,
        minimum_positions=np.array([-2.0]),
    )

    assert result.executed_orders[0] == -3.0
    assert population.positions[0] == -2.0
