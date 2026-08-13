"""Core market tests: population, policies, settlement, microstructure, harness."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.config import load_stage1_config
from abm.harness import MarketHarness, run_stage1
from abm.microstructure import QuasiOrderBook
from abm.policies import MarketObservation, RuleBasedPolicy
from abm.population import (
    NOISE_STRATEGY,
    TREND_STRATEGY,
    VALUE_STRATEGY,
    TraderPopulation,
)
from abm.schemas import RunManifest
from abm.settlement import SettlementEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def _small_config():
    return replace(
        load_stage1_config(CONFIG_PATH),
        population_size=100,
        trading_days=10,
        burn_in_days=0,
        liquidity_scale=1600.0,
        announcements=(),
        learning_enabled=False,
    )


# --- Harness --------------------------------------------------------------


def test_same_seed_and_config_replay_exactly() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=200,
        trading_days=80,
        burn_in_days=0,
        announcements=(),
    )

    first = run_stage1(config)
    second = run_stage1(config)

    assert first.fingerprint() == second.fingerprint()
    assert np.array_equal(first.prices, second.prices)
    assert np.array_equal(first.final_agent_cash, second.final_agent_cash)
    assert np.array_equal(first.final_strategies, second.final_strategies)


def test_disabling_learning_recovers_fixed_rule_baseline_exactly() -> None:
    base = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=200,
        trading_days=59,
        burn_in_days=0,
        announcements=(),
    )

    disabled = run_stage1(replace(base, learning_enabled=False))
    not_yet_updated = run_stage1(replace(base, learning_enabled=True))

    assert disabled.fingerprint() == not_yet_updated.fingerprint()
    assert np.array_equal(disabled.prices, not_yet_updated.prices)
    assert np.array_equal(
        disabled.final_strategies, not_yet_updated.final_strategies
    )


def test_duplicate_or_out_of_order_settlement_is_rejected() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=20,
        trading_days=2,
        burn_in_days=0,
        announcements=(),
    )
    harness = MarketHarness(config)
    harness.step(0)

    with pytest.raises(RuntimeError, match="settled twice"):
        harness.step(0)


def test_logit_updates_are_traceable_to_strategy_fitness() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=300,
        trading_days=120,
        burn_in_days=0,
        learning_enabled=True,
        announcements=(),
    )
    result = run_stage1(config)

    assert [audit.day for audit in result.learning_audits] == [60, 120]
    for audit in result.learning_audits:
        assert sum(audit.counts_before) == config.population_size
        assert sum(audit.counts_after) == config.population_size
        assert np.isclose(sum(audit.choice_probabilities), 1.0)
        best_fitness = int(np.argmax(audit.mean_fitness_by_strategy))
        best_probability = int(np.argmax(audit.choice_probabilities))
        assert best_probability == best_fitness
        assert audit.updated_agents == round(
            config.population_size * config.learning_update_fraction
        )


def test_full_stage1_acceptance_run_meets_invariants() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=1000,
        trading_days=250,
        burn_in_days=0,
        liquidity_scale=16000.0,
        learning_enabled=False,
    )
    result = run_stage1(config)

    assert result.prices.shape == (251,)
    assert len(result.daily_audits) == 250
    assert len({audit.settlement_id for audit in result.daily_audits}) == 250
    assert np.all(np.isfinite(result.prices))
    assert np.all(result.prices > 0)
    assert np.all(result.final_agent_cash >= 0)
    assert np.all(
        result.final_agent_positions
        >= -config.max_short_leverage
        * (
            result.final_agent_cash
            + result.final_agent_positions * result.prices[-1]
        )
        / result.prices[-1]
        - 1e-8
    )
    assert np.max(np.abs(result.total_cash - result.total_cash[0])) < 1e-6
    assert np.max(np.abs(result.total_shares - result.total_shares[0])) < 1e-8
    assert result.strategy_counts.shape == (251, 3)
    assert np.all(result.strategy_counts.sum(axis=1) == 1000)
    assert not result.learning_audits
    assert all(np.isfinite(result.spreads))
    assert np.all(result.spreads > 0)
    assert np.all(result.depths > 0)
    assert np.all(np.abs(result.order_flow_imbalance) <= 1.0)
    assert not any(audit.price_cap_hit for audit in result.daily_audits)


def test_public_news_updates_quotes_and_agent_demand() -> None:
    base = load_stage1_config(CONFIG_PATH)
    config = replace(
        base,
        population_size=200,
        trading_days=60,
        burn_in_days=0,
        fundamental_drift=0.0,
        fundamental_volatility=0.0,
        announcements=(base.announcements[0],),
    )
    result = run_stage1(config)
    announcement_audit = result.daily_audits[59]

    assert announcement_audit.public_news_return != 0.0
    assert np.isclose(
        announcement_audit.public_news_impact,
        config.public_news_price_pass_through
        * announcement_audit.public_news_return,
    )
    assert np.isclose(
        np.log(
            announcement_audit.mid_price_after
            / announcement_audit.price_before
        ),
        announcement_audit.public_news_impact
        + announcement_audit.demand_log_return,
    )


def test_zero_public_news_channel_does_not_reveal_fundamental_level() -> None:
    base = load_stage1_config(CONFIG_PATH)
    config = replace(
        base,
        population_size=100,
        trading_days=3,
        burn_in_days=0,
        public_news_price_pass_through=0.0,
        value_sensitivity=0.0,
        trend_sensitivity=0.0,
        noise_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        liquidity_need_scale=0.0,
        initial_agent_position=0.0,
        announcements=(),
    )
    result = run_stage1(
        config,
        fundamental_innovations=np.full(config.trading_days, 4.0),
    )

    assert not np.array_equal(result.fundamentals, result.prices)
    assert np.all(result.public_news_impacts == 0.0)
    assert np.allclose(result.prices, config.initial_price)


def test_learning_disabled_keeps_fixed_strategy_population_for_250_days() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=1000,
        trading_days=250,
        burn_in_days=0,
        liquidity_scale=16000.0,
        learning_enabled=False,
    )
    result = run_stage1(config)

    assert not result.learning_audits
    assert np.all(result.strategy_counts == result.strategy_counts[0])


def test_run_outputs_are_auditable_and_not_overwritten(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=20,
        trading_days=5,
        burn_in_days=0,
        announcements=(),
    )
    result = run_stage1(config)
    output = tmp_path / "run"

    result.write(output, config)

    assert (output / "state_arrays.npz").is_file()
    manifest = RunManifest.from_dict(
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.random_seed == config.seed
    assert len((output / "daily_audit.jsonl").read_text().splitlines()) == 5
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["fingerprint"] == result.fingerprint()
    with pytest.raises(FileExistsError):
        result.write(output, config)


# --- Settlement -----------------------------------------------------------


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


def test_settlement_absorbs_net_purchases_by_short_selling() -> None:
    population = TraderPopulation(
        cash=np.array([100_000.0, 100_000.0]),
        positions=np.array([100.0, 100.0]),
        strategies=np.array([0, 1], dtype=np.int8),
        risk_aversion=np.ones(2),
    )
    engine = SettlementEngine(transaction_cost_rate=0.0)

    filled = engine.settle(
        population=population,
        submitted_orders=np.array([100.0, 100.0]),
        execution_price=100.0,
        market_maker_cash=10_000.0,
        market_maker_inventory=10.0,
    )

    # Inventory does not bound execution: the market maker covers net
    # purchases by short-selling, so the full batch executes.
    assert np.isclose(filled.executed_orders.sum(), 200.0)
    assert np.isclose(filled.market_maker_inventory, -190.0)
    assert np.isclose(filled.market_maker_cash, 30_000.0)

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

    # Net sales still exhaust the market maker's cash pro rata.
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


# --- Quasi order book -----------------------------------------------------


def test_quasi_order_book_records_quotes_and_learns_persistent_flow() -> None:
    config = _small_config()
    book = QuasiOrderBook.initialize(config)
    orders = np.full(config.population_size, 1.0)

    first = book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.01,
    )
    second = book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.01,
    )

    assert first.bid_price < first.mid_price_after < first.ask_price
    assert first.spread > 0
    assert first.depth > 0
    assert first.order_flow_imbalance == 1.0
    assert abs(second.order_flow_surprise) < abs(first.order_flow_surprise)


def test_persistent_informed_flow_retains_permanent_price_discovery() -> None:
    config = replace(
        _small_config(),
        public_news_price_pass_through=0.0,
        permanent_impact_fraction=1.0,
        order_flow_memory=0.0,
    )
    book = QuasiOrderBook.initialize(config)
    orders = np.full(config.population_size, 1.0)

    first = book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.0,
    )
    second = book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.0,
    )

    assert second.order_flow_surprise == 0.0
    assert first.permanent_impact > 0.0
    assert second.permanent_impact > 0.0
    assert second.mid_price_after > first.mid_price_after


def test_transient_surprise_impact_decays_without_new_flow() -> None:
    config = replace(
        _small_config(),
        permanent_impact_fraction=0.0,
        transient_impact_decay=0.5,
    )
    book = QuasiOrderBook.initialize(config)
    first = book.quote(
        submitted_orders=np.full(config.population_size, 1.0),
        public_news_impact=0.0,
        realized_volatility=0.0,
    )
    second = book.quote(
        submitted_orders=np.zeros(config.population_size),
        public_news_impact=0.0,
        realized_volatility=0.0,
    )

    assert first.transient_impact > 0.0
    assert 0.0 < second.transient_impact < first.transient_impact
    assert second.transient_impact_change < 0.0


def test_book_has_no_hidden_fundamental_level_anchor() -> None:
    config = replace(_small_config(), price_impact=0.0)
    orders = np.zeros(config.population_size, dtype=np.float64)
    low_fundamental_book = QuasiOrderBook.initialize(config)
    high_fundamental_book = QuasiOrderBook.initialize(config)

    low_quote = low_fundamental_book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.0,
    )
    high_quote = high_fundamental_book.quote(
        submitted_orders=orders,
        public_news_impact=0.0,
        realized_volatility=0.0,
    )

    assert low_quote.mid_price_after == config.initial_price
    assert high_quote.mid_price_after == low_quote.mid_price_after
    assert low_quote.public_news_impact == 0.0


def test_public_news_channel_is_explicit_and_independent_of_orders() -> None:
    config = replace(_small_config(), price_impact=0.0)
    book = QuasiOrderBook.initialize(config)
    news_impact = 0.025

    quote = book.quote(
        submitted_orders=np.zeros(config.population_size),
        public_news_impact=news_impact,
        realized_volatility=0.0,
    )

    assert np.isclose(
        np.log(quote.mid_price_after / quote.mid_price_before),
        news_impact,
    )
    assert quote.demand_log_return == 0.0


def test_policy_submits_incremental_target_position_rebalancing() -> None:
    config = replace(
        _small_config(),
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        base_activity_rate=1.0,
        activity_volatility_sensitivity=0.0,
        activity_signal_sensitivity=0.0,
        activity_persistence=0.0,
        activity_shock_scale=0.0,
        liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        common_signal_correlation=0.0,
        initial_subjective_value_dispersion=0.0,
        information_response_dispersion=0.0,
        value_update_rate=1.0,
        value_target_adjustment=0.1,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    assert population.value_update_probabilities is not None
    assert population.base_activity_rates is not None
    population.value_update_probabilities[:] = 1.0
    population.base_activity_rates[:] = 1.0 - 1e-12
    policy = RuleBasedPolicy(config)
    observation = MarketObservation(
        price=100.0,
        fundamental_value=110.0,
        price_history=np.full(10, 100.0),
        public_news_return=float(np.log(1.10)),
        realized_volatility=0.01,
    )

    first = policy.act(observation, population, np.random.default_rng(2))
    population.positions += first
    second = policy.act(observation, population, np.random.default_rng(3))

    assert np.all(first >= 0)
    assert np.all(second >= 0)
    assert np.count_nonzero(first) >= 95
    assert np.count_nonzero(second) >= 95
    assert float(second.mean()) < float(first.mean())


# --- Agent policies -------------------------------------------------------


def test_value_agents_accumulate_and_consume_public_information() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=100,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        initial_subjective_value_dispersion=0.0,
        information_response_dispersion=0.0,
        value_update_rate=0.0,
        base_activity_rate=1.0,
        activity_signal_sensitivity=0.0,
        activity_volatility_sensitivity=0.0,
        activity_persistence=0.0,
        activity_shock_scale=0.0,
        liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        common_signal_correlation=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(10))
    assert population.value_update_probabilities is not None
    assert population.pending_information is not None
    assert population.subjective_values is not None
    population.value_update_probabilities[:50] = 1.0
    population.value_update_probabilities[50:] = 0.0
    policy = RuleBasedPolicy(config)
    news = float(np.log(1.05))
    observation = MarketObservation(
        price=100.0,
        fundamental_value=105.0,
        price_history=np.full(20, 100.0),
        public_news_return=news,
    )

    policy.act(observation, population, np.random.default_rng(11))

    assert np.allclose(population.subjective_values[:50], 105.0)
    assert np.allclose(population.pending_information[:50], 0.0)
    assert np.allclose(population.subjective_values[50:], 100.0)
    assert np.allclose(population.pending_information[50:], news)
    assert policy.diagnostics.value_information_updates == 50

    population.value_update_probabilities[:] = 1.0
    no_news = replace(observation, fundamental_value=105.0, public_news_return=0.0)
    policy.act(no_news, population, np.random.default_rng(12))

    assert np.allclose(population.subjective_values, 105.0)
    assert np.allclose(population.pending_information, 0.0)
    assert policy.diagnostics.value_information_updates == 100


def test_fundamental_level_is_not_directly_visible_to_value_agents() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=100,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        initial_subjective_value_dispersion=0.0,
        information_response_dispersion=0.0,
        value_update_rate=1.0,
    )
    first = TraderPopulation.initialize(config, np.random.default_rng(20))
    second = TraderPopulation.initialize(config, np.random.default_rng(20))
    low = MarketObservation(
        price=100.0,
        fundamental_value=100.0,
        price_history=np.full(20, 100.0),
        public_news_return=0.0,
    )
    high = replace(low, fundamental_value=1000.0)

    low_orders = RuleBasedPolicy(config).act(
        low, first, np.random.default_rng(21)
    )
    high_orders = RuleBasedPolicy(config).act(
        high, second, np.random.default_rng(21)
    )

    assert np.array_equal(low_orders, high_orders)
    assert np.array_equal(first.subjective_values, second.subjective_values)


def test_three_rule_policies_create_distinct_order_distributions() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=300,
        strategy_shares={"value": 1 / 3, "trend": 1 / 3, "noise": 1 / 3},
        trend_sensitivity=2.0,
    )
    seed_sequence = np.random.SeedSequence(config.seed)
    population_seed, policy_seed = seed_sequence.spawn(2)
    population = TraderPopulation.initialize(
        config, np.random.default_rng(population_seed)
    )
    observation = MarketObservation(
        price=100.0,
        fundamental_value=105.0,
        price_history=np.array([95.0, 100.0]),
        public_news_return=float(np.log(1.05)),
        realized_volatility=0.01,
    )

    orders = RuleBasedPolicy(config).act(
        observation,
        population,
        np.random.default_rng(policy_seed),
    )

    value_orders = orders[population.strategies == VALUE_STRATEGY]
    trend_orders = orders[population.strategies == TREND_STRATEGY]
    noise_orders = orders[population.strategies == NOISE_STRATEGY]
    assert 0 < np.count_nonzero(orders) < config.population_size
    assert np.all(np.isfinite(orders))
    assert np.max(np.abs(orders)) <= (
        config.max_order_fraction
        * np.max(population.wealth(observation.price))
        / observation.price
        + 1e-12
    )
    assert value_orders.mean() > noise_orders.mean()
    assert trend_orders.std() > 0
    assert not np.isclose(value_orders.mean(), trend_orders.mean())
    assert noise_orders.std() > 0
    assert np.any(noise_orders < 0) and np.any(noise_orders > 0)
