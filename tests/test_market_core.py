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


def test_margin_covers_are_exempt_from_counterparty_rationing() -> None:
    population = TraderPopulation(
        cash=np.array([20_000.0, 20_000.0]),
        positions=np.array([-100.0, 100.0]),
        strategies=np.array([0, 1], dtype=np.int8),
        risk_aversion=np.ones(2),
    )
    engine = SettlementEngine(transaction_cost_rate=0.0)

    # Trader 0 needs a forced cover of +60 shares while trader 1 sells 100.
    # The market maker only has 500 cash, which would ration the batch, but
    # the margin cover must execute in full regardless.
    result = engine.settle(
        population=population,
        submitted_orders=np.array([0.0, -100.0]),
        execution_price=100.0,
        market_maker_cash=500.0,
        market_maker_inventory=1000.0,
        minimum_positions=np.array([-40.0, -10.0]),
    )

    assert result.executed_orders[0] == 60.0
    assert population.positions[0] == -40.0
    # The voluntary sale is rationed by the counterparty cash.
    assert result.executed_orders[1] <= -5.0
    assert result.market_maker_cash >= 0.0


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


def test_notional_reference_preserves_signals_and_nominal_order_scale() -> None:
    base = replace(
        _small_config(),
        population_size=3,
        strategy_shares={"value": 0.0, "trend": 0.0, "noise": 1.0},
        noise_scale=0.0,
        liquidity_need_scale=0.0,
        liquidity_need_persistence=1.0,
        noise_target_adjustment=1.0,
        max_order_fraction=1.0,
    )
    nominal_orders = []
    rng_states = []
    signals = np.array([-0.4, 0.0, 0.4])
    for price in (100.0, 200.0, 600.0):
        for reference_basis in ("shares", "notional"):
            for depth_basis in ("shares", "notional"):
                config = replace(
                    base,
                    reference_position_basis=reference_basis,
                    liquidity_depth_basis=depth_basis,
                )
                population = TraderPopulation.initialize(
                    config, np.random.default_rng(1)
                )
                population.positions *= config.initial_price / price
                population.desired_positions[:] = population.positions
                population.liquidity_needs[:] = signals
                population.base_activity_rates[:] = 1.0 - 1e-12
                observation = MarketObservation(
                    price=price,
                    fundamental_value=price,
                    price_history=np.full(20, price),
                    realized_volatility=0.01,
                )
                rng = np.random.default_rng(2)
                orders = RuleBasedPolicy(config).act(
                    observation, population, rng
                )
                rng_states.append(rng.bit_generator.state)
                maximum = (
                    config.target_position_fraction
                    * population.reference_wealth / price
                )
                assert np.all(maximum < config.position_cap)
                assert np.all(population.reference_positions == 10.0)
                assert np.all(population.cash == config.initial_agent_cash)
                if reference_basis == "notional":
                    expected = (
                        population.reference_positions
                        * config.initial_price / price
                        + maximum * np.tanh(signals)
                    )
                    assert np.allclose(population.desired_positions, expected)
                    assert np.allclose(
                        orders, expected - population.positions
                    )
                    nominal_orders.append(orders * price)
                    assert np.all(np.diff(orders) > 0)
                elif price == 600.0:
                    assert np.all(population.reference_positions >= 2 * maximum)
                    assert np.allclose(population.desired_positions, maximum)
                    assert np.allclose(orders, orders[0])

    assert all(state == rng_states[0] for state in rng_states)
    assert all(np.allclose(orders, nominal_orders[0]) for orders in nominal_orders)


def test_notional_depth_preserves_impact_and_recovers_across_price_levels() -> None:
    config = replace(
        _small_config(),
        liquidity_depth_basis="notional",
        liquidity_volatility_sensitivity=1.0,
        liquidity_stress_threshold=0.8,
        minimum_liquidity_fraction=0.4,
    )
    books = [QuasiOrderBook.initialize(config) for _ in range(2)]
    books[1].mid_price *= 6.0
    fraction = 1.0
    for volatility in (0.02, 0.0, 0.0):
        fraction += config.depth_resilience * (
            (0.4 if volatility else 1.0) - fraction
        )
        quotes = []
        for book in books:
            price = book.mid_price
            quote = book.quote(
                submitted_orders=np.array([2500.0, -1000.0, 500.0]) / price,
                public_news_impact=0.01,
                realized_volatility=volatility,
                realized_volatility_reference=0.01,
            )
            assert np.isclose(
                quote.depth * price,
                config.liquidity_scale * config.initial_price * fraction,
            )
            quotes.append(quote)
        low, high = quotes
        assert np.isclose(high.mid_price_after, low.mid_price_after * 6.0)
        assert np.isclose(high.demand_log_return, low.demand_log_return)
        assert np.isclose(high.permanent_impact, low.permanent_impact)
        assert np.isclose(high.transient_impact, low.transient_impact)
        assert np.isclose(high.spread / high.mid_price_after,
                          low.spread / low.mid_price_after)


@pytest.mark.parametrize("basis", ["shares", "notional"])
def test_price_cap_keeps_transient_state_and_complete_price_identity(basis) -> None:
    config = replace(
        _small_config(),
        liquidity_depth_basis=basis,
        liquidity_volatility_sensitivity=0.0,
        minimum_liquidity_fraction=1.0,
        price_impact=0.8,
        permanent_impact_fraction=0.25,
        transient_impact_decay=0.5,
        order_flow_memory=0.0,
        max_log_return=0.3,
    )
    book = QuasiOrderBook.initialize(config)
    transient = 0.0
    previous_imbalance = 0.0
    for pressure, news in ((3.0, 0.02), (0.0, -0.01), (-3.0, 0.03),
                           (0.0, 0.0), (0.02, -0.02), (-0.02, 0.01)):
        base_depth = config.liquidity_scale
        if basis == "notional":
            base_depth *= config.initial_price / book.mid_price
        quote = book.quote(
            submitted_orders=np.array([pressure * base_depth]),
            public_news_impact=news,
            realized_volatility=0.0,
        )
        imbalance = float(np.sign(pressure))
        surprise = imbalance - previous_imbalance
        expected_transient = (
            config.transient_impact_decay * transient
            + (1.0 - config.permanent_impact_fraction)
            * config.price_impact * surprise * abs(pressure)
        )
        assert np.isclose(quote.order_flow_surprise, surprise)
        assert np.isclose(quote.transient_impact, expected_transient)
        assert np.isclose(quote.transient_impact_change,
                          expected_transient - transient)
        raw = quote.permanent_impact + quote.transient_impact_change
        cap_adjustment = quote.demand_log_return - raw
        assert np.isclose(quote.demand_log_return, np.clip(raw, -0.3, 0.3))
        assert quote.price_cap_hit == (abs(raw) > config.max_log_return)
        assert np.isclose(
            np.log(quote.mid_price_after / quote.mid_price_before),
            news + quote.permanent_impact
            + quote.transient_impact_change + cap_adjustment,
        )
        if abs(pressure) == 3.0:
            assert quote.price_cap_hit
        if pressure == 0.0:
            assert np.isclose(quote.transient_impact, transient * 0.5)
            assert quote.transient_impact != 0.0
        transient = expected_transient
        previous_imbalance = imbalance


@pytest.mark.parametrize(
    ("learning", "fingerprint"),
    [(False, "3102a3020e2decc12cad4a60b0b51c39963e4dfe83ba041aae88dd61b4444e7e"),
     (True, "41ed29300137693c1c96aeb45f8509f78612733eb51ea7cf1f24980fa89dd1fb")],
)
def test_basis_switches_preserve_legacy_replay_and_notional_determinism(
    learning, fingerprint
) -> None:
    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v6.json"),
        population_size=100,
        trading_days=250,
        burn_in_days=0,
        announcements=(),
        learning_enabled=learning,
    )
    import hashlib

    assert config.reference_position_basis == config.liquidity_depth_basis == "shares"
    assert config.value_order_constraint == "none"
    result = run_stage1(config)
    assert result.fingerprint() == run_stage1(config).fingerprint()
    fixture = json.loads((
        PROJECT_ROOT / "reports" / "stage1_iter1_r2_20260926" / "baseline"
        / f"legacy_logit_{str(learning).lower()}.json"
    ).read_text(encoding="utf-8"))
    assert fixture["fingerprint"] == fingerprint
    for name, expected_hash in fixture["array_sha256"].items():
        array = np.ascontiguousarray(getattr(result, name))
        if array.dtype.kind == "f":
            array = np.round(array, 6)
            expected_hash = fixture["array_sha256_6dp"][name]
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected_hash
    audits = [
        {name: int(round(value * 10000)) if isinstance(value, float) else value
         for name, value in audit.to_dict().items() if name != "settlement_id"}
        for audit in result.daily_audits
    ]
    audit_bytes = json.dumps(audits, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(audit_bytes).hexdigest() == fixture["daily_audit_sha256_4dp_without_settlement_id"]
    assert len({audit.settlement_id for audit in result.daily_audits}) == config.trading_days
    notional = replace(
        config,
        reference_position_basis="notional",
        liquidity_depth_basis="notional",
    )
    assert run_stage1(notional).fingerprint() == run_stage1(notional).fingerprint()


@pytest.mark.parametrize("price", [80.0, 125.0])
def test_notional_center_rebalances_without_new_signal(price: float) -> None:
    config = replace(
        _small_config(),
        population_size=1,
        strategy_shares={"value": 0.0, "trend": 0.0, "noise": 1.0},
        reference_position_basis="notional",
        liquidity_depth_basis="notional",
        noise_scale=0.0,
        liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        activity_shock_scale=0.0,
        liquidity_volatility_sensitivity=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.base_activity_rates[:] = 1.0 - 1e-12
    policy = RuleBasedPolicy(config)
    observation = MarketObservation(
        price=config.initial_price,
        fundamental_value=config.initial_fundamental,
        price_history=np.full(20, config.initial_price),
        public_news_return=0.0,
        realized_volatility=0.01,
    )
    assert np.array_equal(
        policy.act(observation, population, np.random.default_rng(2)), [0.0]
    )
    orders = policy.act(
        replace(observation, price=price, price_history=np.full(20, price)),
        population,
        np.random.default_rng(3),
    )
    expected = config.noise_target_adjustment * (
        config.initial_agent_position * config.initial_price / price
        - config.initial_agent_position
    )
    np.testing.assert_allclose(orders, [expected], rtol=0.0, atol=1e-14)
    assert expected * (price - config.initial_price) < 0.0
    assert np.all(population.positions + orders > 0.0)
    book = QuasiOrderBook.initialize(config)
    book.mid_price = price
    quote = book.quote(
        submitted_orders=orders, public_news_impact=0.0,
        realized_volatility=observation.realized_volatility,
    )
    signed_pressure = expected / (config.liquidity_scale * config.initial_price / price)
    permanent = config.permanent_impact_fraction * config.price_impact * signed_pressure
    transient = (1.0 - config.permanent_impact_fraction) * config.price_impact * signed_pressure
    assert quote.permanent_impact == pytest.approx(permanent)
    assert quote.transient_impact == pytest.approx(transient)
    assert quote.transient_impact_change == pytest.approx(transient)
    assert quote.demand_log_return == pytest.approx(permanent + transient)
    assert np.sign(quote.permanent_impact) == np.sign(expected)
    assert np.sign(quote.transient_impact) == np.sign(expected)
    assert not quote.price_cap_hit


def test_reached_value_target_has_no_orders_despite_persistent_mispricing() -> None:
    config = replace(
        _small_config(),
        population_size=1,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        reference_position_basis="notional",
        value_sensitivity=1.0,
        value_update_rate=0.0,
        value_no_trade_band=0.005,
        noise_scale=0.0,
        liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        activity_shock_scale=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.base_activity_rates[:] = 1.0 - 1e-12
    population.risk_aversion[:] = 1.0
    population.subjective_values[:] = 110.0
    price = 100.0
    limit = config.target_position_fraction * population.reference_wealth / price
    target = population.reference_positions + limit * np.tanh(0.1)
    assert np.all((target > 0.0) & (target < limit))
    assert np.all(limit < config.position_cap)
    population.positions[:] = target
    population.desired_positions[:] = target
    observation = MarketObservation(
        price=price, fundamental_value=110.0,
        price_history=np.full(20, price), public_news_return=0.0,
        realized_volatility=0.01,
    )
    policy = RuleBasedPolicy(config)
    for seed in (2, 3, 4):
        orders = policy.act(observation, population, np.random.default_rng(seed))
        np.testing.assert_array_equal(orders, [0.0])
        np.testing.assert_array_equal(population.desired_positions, target)
        assert np.all((population.subjective_values - price) / price == 0.1)
        assert np.all(population.cash > target * price)


def test_no_trade_band_retains_voluntary_target_gap_without_margin_covers() -> None:
    config = replace(
        _small_config(),
        population_size=2,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        reference_position_basis="notional",
        value_target_adjustment=0.25,
        value_update_rate=0.0,
        value_no_trade_band=0.005,
        noise_scale=0.0,
        liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0,
        activity_shock_scale=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.base_activity_rates[:] = 1.0 - 1e-12
    population.subjective_values[:] = 100.2
    gap = np.array([2.0, -2.0])
    population.desired_positions[:] = population.positions + gap
    observation = MarketObservation(
        price=100.0, fundamental_value=100.2,
        price_history=np.full(20, 100.0), public_news_return=0.0,
        realized_volatility=0.01,
    )
    assert np.all(
        np.abs(population.subjective_values / observation.price - 1.0)
        < config.value_no_trade_band
    )
    orders = RuleBasedPolicy(config).act(
        observation, population, np.random.default_rng(2)
    )
    expected = (1.0 - config.value_target_adjustment) * gap
    np.testing.assert_array_equal(population.desired_positions - population.positions, expected)
    np.testing.assert_array_equal(orders, expected)
    floor_price = observation.price * np.exp(
        config.max_log_return + 4.0 * config.fundamental_volatility
    )
    minimum_positions = np.maximum(
        -config.max_short_leverage * population.wealth(observation.price) / floor_price,
        -config.position_cap,
    )
    assert np.all(population.positions > minimum_positions)
    assert np.all(population.positions + orders > minimum_positions)
    settlement = SettlementEngine(config.transaction_cost_rate).settle(
        population=population, submitted_orders=orders,
        execution_price=observation.price,
        market_maker_cash=config.market_maker_cash,
        market_maker_inventory=config.market_maker_inventory,
        minimum_positions=minimum_positions,
    )
    np.testing.assert_array_equal(settlement.executed_orders, expected)


@pytest.mark.parametrize(
    ("strategy", "mispricing", "direction", "sensitivity", "active", "removed"),
    [
        ("value", 0.1, -1, 1.0, True, True),
        ("value", -0.1, 1, 1.0, True, True),
        ("value", 0.1, 1, 1.0, True, False),
        ("value", -0.1, -1, 1.0, True, False),
        ("value", 0.0, 1, 1.0, True, False),
        ("value", 0.0, -1, 1.0, True, False),
        ("value", 0.002, -1, 1.0, True, False),
        ("value", -0.002, 1, 1.0, True, False),
        ("value", 0.1, -1, 0.0, True, False),
        ("value", -0.1, 1, 0.0, True, False),
        ("trend", 0.1, -1, 1.0, True, False),
        ("noise", -0.1, 1, 1.0, True, False),
        ("value", 0.1, -1, 1.0, False, False),
    ],
)
def test_value_direction_constraint_changes_only_opposing_active_orders(
    strategy, mispricing, direction, sensitivity, active, removed
) -> None:
    from copy import deepcopy
    from dataclasses import fields

    config = replace(
        _small_config(), population_size=1,
        strategy_shares={name: float(name == strategy) for name in ("value", "trend", "noise")},
        value_order_constraint="none", value_sensitivity=sensitivity,
        value_target_adjustment=0.25, value_no_trade_band=0.005,
        value_update_rate=0.0, noise_scale=0.0, liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0, activity_shock_scale=0.0,
        activity_signal_sensitivity=0.0, activity_volatility_sensitivity=0.0,
    )
    original = TraderPopulation.initialize(config, np.random.default_rng(1))
    original.subjective_values[:] = 100.0 * (1.0 + mispricing)
    original.risk_aversion[:] = 1.0
    original.desired_positions[:] = original.positions + 10.0 * direction
    original.base_activity_rates[:] = 1.0 - 1e-12 if active else 1e-12
    observation = MarketObservation(
        price=100.0, fundamental_value=100.0, price_history=np.full(20, 100.0),
        public_news_return=0.0, realized_volatility=0.01,
    )
    populations = [deepcopy(original) for _ in range(3)]
    rngs = [np.random.default_rng(2) for _ in range(3)]
    enabled = replace(config, value_order_constraint="valuation_direction")
    policies = [RuleBasedPolicy(config, instrument_orders=True),
                RuleBasedPolicy(enabled, instrument_orders=True), RuleBasedPolicy(enabled)]
    before, after, unobserved = [
        policy.act(observation, population, rng)
        for policy, population, rng in zip(policies, populations, rngs, strict=True)
    ]
    assert np.sign(before[0]) == (direction if active else 0)
    np.testing.assert_array_equal(after, np.zeros(1) if removed else before)
    np.testing.assert_array_equal(after, unobserved)
    for field in fields(original):
        for population in populations[1:]:
            np.testing.assert_equal(getattr(population, field.name), getattr(populations[0], field.name))
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state == rngs[2].bit_generator.state
    assert policies[0].activity_state == policies[1].activity_state == policies[2].activity_state
    np.testing.assert_array_equal(policies[0].common_signal_state, policies[1].common_signal_state)
    assert policies[0].diagnostics == policies[1].diagnostics == policies[2].diagnostics
    assert policies[2].order_diagnostics is None
    snapshots = policies[1].order_diagnostics
    expected_signal = sensitivity * mispricing if abs(mispricing) >= config.value_no_trade_band else 0.0
    assert snapshots.value_signal[0] == pytest.approx(expected_signal)
    np.testing.assert_array_equal(snapshots.before_constraint, before)
    np.testing.assert_array_equal(snapshots.after_constraint, after)
    np.testing.assert_array_equal(snapshots.after_risk, after)
    assert np.all(np.abs(snapshots.after_constraint) <= np.abs(snapshots.before_constraint))
    for field in fields(snapshots):
        array = getattr(snapshots, field.name)
        assert not array.flags.writeable
        assert not np.shares_memory(array, after)
        with pytest.raises(ValueError):
            array[0] = 123.0


def test_value_direction_constraint_keeps_cash_margin_and_settlement_bounds() -> None:
    config = replace(
        _small_config(), population_size=3,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_order_constraint="valuation_direction", value_sensitivity=1.0,
        value_target_adjustment=0.25, value_update_rate=0.0,
        noise_scale=0.0, liquidity_need_scale=0.0, idiosyncratic_signal_scale=0.0,
        activity_shock_scale=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.cash[:] = [1.0, 1000.0, 10000.0]
    population.positions[:] = [0.0, -0.5, -30.0]
    population.desired_positions[:] = [10.0, -20.0, -20.0]
    population.subjective_values[:] = [110.0, 90.0, 90.0]
    population.risk_aversion[:] = 1.0
    population.base_activity_rates[:] = 1.0 - 1e-12
    observation = MarketObservation(
        price=100.0, fundamental_value=100.0, price_history=np.full(20, 100.0),
        public_news_return=0.0, realized_volatility=0.01,
    )
    initial_cash = population.cash.sum() + config.market_maker_cash
    initial_shares = population.positions.sum() + config.market_maker_inventory
    affordable = population.cash / (100.0 * np.exp(config.max_log_return) * (1.0 + config.transaction_cost_rate))
    minimum = np.maximum(
        -config.max_short_leverage * population.wealth(100.0)
        / (100.0 * np.exp(config.max_log_return + 4.0 * config.fundamental_volatility)),
        -config.position_cap,
    )
    minimum_orders = minimum - population.positions
    policy = RuleBasedPolicy(config, instrument_orders=True)
    orders = policy.act(observation, population, np.random.default_rng(2))
    snapshots = policy.order_diagnostics
    assert orders[0] == pytest.approx(affordable[0])
    assert orders[1] == pytest.approx(minimum_orders[1])
    assert snapshots.value_signal[2] < 0.0
    assert snapshots.before_constraint[2] > 0.0
    assert snapshots.after_constraint[2] == 0.0
    assert orders[2] == pytest.approx(minimum_orders[2])
    assert orders[2] > 0.0
    np.testing.assert_allclose(snapshots.forced_cover_orders, [0.0, 0.0, orders[2]])
    assert np.all(np.abs(snapshots.after_constraint) <= np.abs(snapshots.before_constraint))
    assert np.all(population.positions + orders >= minimum - 1e-12)
    settlement = SettlementEngine(config.transaction_cost_rate).settle(
        population=population, submitted_orders=orders, execution_price=100.0,
        market_maker_cash=config.market_maker_cash,
        market_maker_inventory=config.market_maker_inventory, minimum_positions=minimum,
    )
    np.testing.assert_allclose(settlement.executed_orders, orders)
    assert np.all(population.cash >= 0.0)
    assert np.all(population.positions >= minimum - 1e-12)
    assert np.all(np.abs(population.positions) <= config.position_cap)
    assert population.cash.sum() + settlement.market_maker_cash == pytest.approx(initial_cash, rel=0.0, abs=1e-7)
    assert population.positions.sum() + settlement.market_maker_inventory == pytest.approx(initial_shares, rel=0.0, abs=1e-9)


def test_value_direction_constraint_does_not_observe_true_fundamental() -> None:
    from copy import deepcopy

    config = replace(
        _small_config(), population_size=3,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_order_constraint="valuation_direction", value_sensitivity=1.0,
        value_target_adjustment=0.25, value_update_rate=0.0,
        noise_scale=0.0, liquidity_need_scale=0.0, idiosyncratic_signal_scale=0.0,
        activity_shock_scale=0.0,
    )
    first = TraderPopulation.initialize(config, np.random.default_rng(1))
    first.subjective_values[:] = [110.0, 90.0, 100.2]
    first.desired_positions[:] = [12.0, 8.0, 12.0]
    first.base_activity_rates[:] = 1.0 - 1e-12
    second = deepcopy(first)
    observation = MarketObservation(
        price=100.0, fundamental_value=100.0, price_history=np.full(20, 100.0),
        public_news_return=0.0, realized_volatility=0.01,
    )
    orders = RuleBasedPolicy(config).act(observation, first, np.random.default_rng(2))
    changed = RuleBasedPolicy(config).act(
        replace(observation, fundamental_value=1000.0), second, np.random.default_rng(2)
    )
    assert np.any(orders > 0.0) and np.any(orders < 0.0)
    np.testing.assert_array_equal(orders, changed)
    np.testing.assert_array_equal(first.desired_positions, second.desired_positions)
    np.testing.assert_array_equal(first.subjective_values, second.subjective_values)


@pytest.mark.parametrize("learning", [False, True])
def test_value_direction_constraint_replays_deterministically(learning) -> None:
    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v8.json"),
        population_size=100, trading_days=120, burn_in_days=0,
        learning_enabled=learning, announcements=(),
    )
    assert config.value_order_constraint == "valuation_direction"
    first = run_stage1(config)
    second = run_stage1(config)
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize("bypass", ["value_sensitivity_zero", "trend_only", "noise_only"])
def test_value_direction_constraint_bypass_preserves_complete_path(bypass) -> None:
    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v8.json"),
        population_size=100, trading_days=120, burn_in_days=0,
        learning_enabled=False, announcements=(),
    )
    if bypass == "value_sensitivity_zero":
        config = replace(config, value_sensitivity=0.0)
    else:
        strategy = bypass.removesuffix("_only")
        config = replace(config, strategy_shares={
            name: float(name == strategy) for name in ("value", "trend", "noise")
        })
    constrained = run_stage1(config)
    original = run_stage1(replace(config, value_order_constraint="none"))
    assert constrained.fingerprint() == original.fingerprint()


@pytest.mark.parametrize(
    ("limit", "maximum", "position", "desired_order", "expected"),
    [
        (5.0, 2.0, 1.0, 1.25, 1.25),
        (5.0, 2.0, -1.0, -1.25, -1.25),
        (5.0, 0.2, 1.0, 1.0, 0.2),
        (5.0, 0.2, -1.0, -1.0, -0.2),
        (5.0, 2.0, 4.0, 2.0, 1.0),
        (5.0, 2.0, -4.0, -2.0, -1.0),
        (5.0, 2.0, 4.0, -1.0, -1.0),
        (5.0, 2.0, 0.0, 0.0, 0.0),
        (5.0, 2.0, 6.0, 1.0, -1.0),
        (5.0, 2.0, -6.0, -1.0, 1.0),
        (5.0, 2.0, 6.0, 0.0, -1.0),
        (5.0, 2.0, -6.0, 0.0, 1.0),
        (5.0, 2.0, 7.0, 0.0, -2.0),
        (5.0, 2.0, -7.0, 0.0, 2.0),
        (5.0, 2.0, 8.0, 0.0, -2.0),
        (5.0, 2.0, -8.0, 0.0, 2.0),
        (5.0, 0.0, 8.0, 0.0, 0.0),
        (5.0, 0.0, -8.0, 0.0, 0.0),
        (0.0, 2.0, 1.0, 0.0, -1.0),
        (0.0, 2.0, -1.0, 0.0, 1.0),
        (0.0, 2.0, 3.0, 0.0, -2.0),
        (0.0, 2.0, -3.0, 0.0, 2.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0, 0.0, 0.0),
    ],
)
def test_value_budget_projects_orders_and_contracts_symmetric_excess(
    limit, maximum, position, desired_order, expected
) -> None:
    config = replace(
        _small_config(), population_size=1,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_inventory_control="budget_priority", value_target_adjustment=0.0,
        target_position_fraction=limit / 10.0, max_order_fraction=maximum / 10.0,
        value_update_rate=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.cash[:] = 1_000_000.0
    population.reference_wealth[:] = 1000.0
    population.positions[:] = position
    population.desired_positions[:] = position + desired_order
    population.subjective_values[:] = 100.0
    population.base_activity_rates[:] = 1.0 - 1e-12
    observation = MarketObservation(100.0, 100.0, np.full(20, 100.0))
    policy = RuleBasedPolicy(config, instrument_orders=True)
    orders = policy.act(observation, population, np.random.default_rng(2))
    snapshots = policy.order_diagnostics
    assert snapshots.budget_limit[0] == limit
    assert snapshots.maximum_rebalance[0] == maximum
    assert snapshots.after_constraint[0] == np.clip(desired_order, -maximum, maximum)
    assert snapshots.after_budget[0] == expected
    assert orders[0] == expected
    assert snapshots.budget_adjustment[0] == expected - snapshots.after_constraint[0]
    assert abs(expected) <= maximum
    excess_before = max(abs(position) - limit, 0.0)
    excess_after = max(abs(position + orders[0]) - limit, 0.0)
    assert excess_after <= max(excess_before - maximum, 0.0)
    assert population.positions[0] == position
    assert population.cash[0] == 1_000_000.0
    assert population.desired_positions[0] == position + desired_order


@pytest.mark.parametrize("sensitivity", [0.0, 1.0])
def test_value_budget_switches_preserve_state_rng_and_information_boundary(sensitivity) -> None:
    from copy import deepcopy
    from dataclasses import fields

    config = replace(
        _small_config(), population_size=10,
        value_sensitivity=sensitivity, value_target_adjustment=0.0,
        trend_target_adjustment=0.0, noise_target_adjustment=0.0,
        target_position_fraction=0.5, max_order_fraction=0.2,
        value_update_rate=0.0, noise_scale=0.0, liquidity_need_scale=0.0,
        idiosyncratic_signal_scale=0.0, activity_shock_scale=0.0,
        activity_signal_sensitivity=0.0, activity_volatility_sensitivity=0.0,
    )
    original = TraderPopulation.initialize(config, np.random.default_rng(1))
    original.strategies[:] = [VALUE_STRATEGY] * 8 + [TREND_STRATEGY, NOISE_STRATEGY]
    original.cash[:] = 1_000_000.0
    original.reference_wealth[:] = 1000.0
    original.positions[:] = [8.0, -8.0, 4.0, -4.0, 8.0, -8.0, 4.0, -4.0, 8.0, -8.0]
    original.desired_positions[:] = original.positions + [-1.0, 1.0, 1.0, -1.0, 2.0, -2.0, 2.0, -2.0, 1.0, -1.0]
    original.subjective_values[:] = [110.0, 90.0] * 5
    original.risk_aversion[:] = 1.0
    original.base_activity_rates[:] = [1.0 - 1e-12] * 4 + [1e-12] * 6
    observation = MarketObservation(100.0, 100.0, np.full(20, 100.0))
    rng_states = []
    policies = []
    for direction in ("none", "valuation_direction"):
        for budget in ("none", "budget_priority"):
            selected = replace(config, value_order_constraint=direction,
                               value_inventory_control=budget)
            expected_direction = np.array([-1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if direction == "valuation_direction" and sensitivity:
                expected_direction[:2] = 0.0
            expected = (
                np.array([-2.0, 2.0, 1.0, -1.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
                if budget == "budget_priority" else expected_direction
            )
            for instrument, fundamental in ((True, 100.0), (False, 100.0), (True, 1000.0)):
                population = deepcopy(original)
                rng = np.random.default_rng(2)
                policy = RuleBasedPolicy(selected, instrument_orders=instrument)
                orders = policy.act(replace(observation, fundamental_value=fundamental), population, rng)
                np.testing.assert_array_equal(orders, expected)
                for field in fields(original):
                    np.testing.assert_equal(getattr(population, field.name), getattr(original, field.name))
                if instrument:
                    snapshots = policy.order_diagnostics
                    np.testing.assert_array_equal(snapshots.after_constraint, expected_direction)
                    np.testing.assert_array_equal(snapshots.after_budget, expected)
                    np.testing.assert_array_equal(snapshots.budget_adjustment, expected - expected_direction)
                    np.testing.assert_array_equal(snapshots.after_risk, expected)
                    np.testing.assert_array_equal(snapshots.forced_cover_orders, np.zeros(10))
                    np.testing.assert_array_equal(snapshots.budget_limit, np.full(10, 5.0))
                    np.testing.assert_array_equal(snapshots.maximum_rebalance, np.full(10, 2.0))
                    for field in fields(snapshots):
                        array = getattr(snapshots, field.name)
                        assert not array.flags.writeable
                        assert not np.shares_memory(array, orders)
                        assert not np.shares_memory(array, population.positions)
                        with pytest.raises(ValueError):
                            array[0] = 123.0
                else:
                    assert policy.order_diagnostics is None
                rng_states.append(rng.bit_generator.state)
                policies.append(policy)
    assert all(state == rng_states[0] for state in rng_states)
    assert all(policy.activity_state == policies[0].activity_state for policy in policies)
    assert all(policy.diagnostics == policies[0].diagnostics for policy in policies)
    for policy in policies[1:]:
        np.testing.assert_array_equal(policy.common_signal_state, policies[0].common_signal_state)


def test_value_budget_keeps_legacy_diagnostic_constructor() -> None:
    from abm.policies import PolicyOrderDiagnostics

    snapshots = PolicyOrderDiagnostics(*(np.zeros(1) for _ in range(5)))
    assert snapshots.after_budget is None
    assert snapshots.budget_adjustment is None
    assert snapshots.budget_limit is None
    assert snapshots.maximum_rebalance is None


def test_value_budget_uses_supply_cap_when_tighter_than_nominal_limit() -> None:
    config = replace(
        _small_config(), population_size=1,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_inventory_control="budget_priority", value_target_adjustment=0.0,
        target_position_fraction=1.0, max_order_fraction=0.2,
        initial_agent_position=0.0, market_maker_inventory=10.0,
        max_position_float_multiple=0.5,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.reference_wealth[:] = 1000.0
    population.positions[:] = 6.0
    population.desired_positions[:] = 6.0
    policy = RuleBasedPolicy(config, instrument_orders=True)
    orders = policy.act(MarketObservation(100.0, 100.0, np.full(20, 100.0)),
                        population, np.random.default_rng(2))
    assert config.target_position_fraction * population.reference_wealth[0] / 100.0 == 10.0
    assert policy.order_diagnostics.budget_limit[0] == config.position_cap == 5.0
    np.testing.assert_array_equal(orders, [-1.0])


def test_value_budget_keeps_cash_bounds_and_margin_priority() -> None:
    config = replace(
        _small_config(), population_size=3,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_order_constraint="valuation_direction", value_inventory_control="budget_priority",
        value_sensitivity=1.0, value_target_adjustment=0.0, value_update_rate=0.0,
        target_position_fraction=0.5, max_order_fraction=0.2,
        max_short_leverage=0.1, max_log_return=0.1,
        fundamental_volatility=0.0, transaction_cost_rate=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    floor_price = 100.0 * np.exp(config.max_log_return)
    population.positions[:] = [4.0, -8.0, -4.0]
    population.cash[:] = [1.0, 800.0 + floor_price / 0.1, 400.0 + floor_price / 0.1]
    population.reference_wealth[:] = 1000.0
    population.desired_positions[:] = population.positions + [2.0, 2.0, -2.0]
    population.subjective_values[:] = [110.0, 90.0, 90.0]
    population.base_activity_rates[:] = 1.0 - 1e-12
    minimum = -config.max_short_leverage * population.wealth(100.0) / floor_price
    initial_cash = population.cash.sum() + config.market_maker_cash
    initial_shares = population.positions.sum() + config.market_maker_inventory
    policy = RuleBasedPolicy(config, instrument_orders=True)
    orders = policy.act(MarketObservation(100.0, 100.0, np.full(20, 100.0)),
                        population, np.random.default_rng(2))
    snapshots = policy.order_diagnostics
    np.testing.assert_array_equal(snapshots.after_constraint, [2.0, 0.0, -2.0])
    np.testing.assert_array_equal(snapshots.after_budget, [1.0, 2.0, -1.0])
    np.testing.assert_allclose(orders, [1.0 / floor_price, 2.0, 3.0])
    np.testing.assert_allclose(snapshots.forced_cover_orders, [0.0, 0.0, 3.0])
    settlement = SettlementEngine(0.0).settle(
        population=population, submitted_orders=orders, execution_price=100.0,
        market_maker_cash=config.market_maker_cash,
        market_maker_inventory=config.market_maker_inventory, minimum_positions=minimum,
    )
    np.testing.assert_allclose(settlement.executed_orders, [1.0 / floor_price, 7.0, 3.0])
    assert np.all(population.positions >= minimum - 1e-12)
    assert np.all(population.cash >= 0.0)
    assert population.cash.sum() + settlement.market_maker_cash == pytest.approx(initial_cash, rel=0.0, abs=1e-7)
    assert population.positions.sum() + settlement.market_maker_inventory == pytest.approx(initial_shares, rel=0.0, abs=1e-9)


def test_value_budget_orders_receive_no_counterparty_rationing_exemption() -> None:
    config = replace(
        _small_config(), population_size=3,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_inventory_control="budget_priority", value_target_adjustment=0.0,
        target_position_fraction=0.5, max_order_fraction=0.2, transaction_cost_rate=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.cash[:] = 1_000_000.0
    population.positions[:] = [8.0, 8.0, -8.0]
    population.desired_positions[:] = population.positions
    population.reference_wealth[:] = 1000.0
    initial_cash = population.cash.sum() + 50.0
    initial_shares = population.positions.sum() + config.market_maker_inventory
    policy = RuleBasedPolicy(config, instrument_orders=True)
    orders = policy.act(MarketObservation(100.0, 100.0, np.full(20, 100.0)),
                        population, np.random.default_rng(2))
    np.testing.assert_array_equal(orders, [-2.0, -2.0, 2.0])
    settlement = SettlementEngine(0.0).settle(
        population=population, submitted_orders=orders, execution_price=100.0,
        market_maker_cash=50.0, market_maker_inventory=config.market_maker_inventory,
        minimum_positions=np.full(3, -100.0),
    )
    np.testing.assert_array_equal(settlement.executed_orders, [-0.5, -0.5, 0.5])
    np.testing.assert_array_equal(population.positions, [7.5, 7.5, -7.5])
    assert np.all(np.abs(population.positions) > policy.order_diagnostics.budget_limit)
    assert np.all(population.cash >= 0.0)
    assert settlement.market_maker_cash == 0.0
    assert population.cash.sum() + settlement.market_maker_cash == initial_cash
    assert population.positions.sum() + settlement.market_maker_inventory == initial_shares


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_value_budget_recovers_in_finite_steps_with_sufficient_execution(sign) -> None:
    config = replace(
        _small_config(), population_size=1,
        strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
        value_order_constraint="valuation_direction", value_inventory_control="budget_priority",
        value_target_adjustment=0.0, value_update_rate=0.0,
        target_position_fraction=0.5, max_order_fraction=0.2, transaction_cost_rate=0.0,
    )
    population = TraderPopulation.initialize(config, np.random.default_rng(1))
    population.cash[:] = 1_000_000.0
    population.positions[:] = 12.0 * sign
    population.desired_positions[:] = population.positions
    population.reference_wealth[:] = 1000.0
    population.subjective_values[:] = 100.0
    population.base_activity_rates[:] = 1e-12
    policy = RuleBasedPolicy(config, instrument_orders=True)
    maker_cash, maker_inventory = config.market_maker_cash, config.market_maker_inventory
    initial_cash = population.cash.sum() + maker_cash
    initial_shares = population.positions.sum() + maker_inventory
    for day, expected_position in enumerate([10.0, 8.0, 6.0, 5.0, 5.0]):
        orders = policy.act(MarketObservation(100.0, 100.0, np.full(20, 100.0)),
                            population, np.random.default_rng(day + 2))
        np.testing.assert_array_equal(policy.order_diagnostics.after_constraint, [0.0])
        settlement = SettlementEngine(0.0).settle(
            population=population, submitted_orders=orders, execution_price=100.0,
            market_maker_cash=maker_cash, market_maker_inventory=maker_inventory,
            minimum_positions=np.array([-100.0]),
        )
        maker_cash, maker_inventory = settlement.market_maker_cash, settlement.market_maker_inventory
        np.testing.assert_array_equal(settlement.executed_orders, orders)
        assert population.positions[0] == expected_position * sign
        assert population.desired_positions[0] == 12.0 * sign
        assert population.cash.sum() + maker_cash == initial_cash
        assert population.positions.sum() + maker_inventory == initial_shares


@pytest.mark.parametrize("strategy", ["trend", "noise"])
def test_value_budget_preserves_nonvalue_complete_paths(strategy) -> None:
    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v9.json"),
        population_size=100, trading_days=120, burn_in_days=0,
        learning_enabled=False, announcements=(),
        strategy_shares={name: float(name == strategy) for name in ("value", "trend", "noise")},
    )
    enabled = run_stage1(config)
    disabled = run_stage1(replace(config, value_inventory_control="none"))
    assert enabled.fingerprint() == disabled.fingerprint()


@pytest.mark.parametrize("learning", [False, True])
def test_value_budget_replays_deterministically(learning) -> None:
    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v9.json"),
        population_size=100, trading_days=120, burn_in_days=0,
        learning_enabled=learning, announcements=(),
    )
    assert run_stage1(config).fingerprint() == run_stage1(config).fingerprint()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_net_share_permanent_impact_cancels_round_trip_across_quotes(sign) -> None:
    config = replace(
        _small_config(), permanent_impact_basis="net_shares",
        liquidity_depth_basis="notional", permanent_impact_fraction=1.0,
        liquidity_volatility_sensitivity=1.0, minimum_liquidity_fraction=0.4,
    )
    book = QuasiOrderBook.initialize(config)
    first = book.quote(
        submitted_orders=np.array([200.0 * sign]),
        public_news_impact=float(np.log(2.0)), realized_volatility=0.04,
        realized_volatility_reference=0.01,
    )
    second = book.quote(
        submitted_orders=np.array([-200.0 * sign]),
        public_news_impact=-float(np.log(2.0)), realized_volatility=0.001,
        realized_volatility_reference=0.01,
    )
    assert first.depth != second.depth
    assert first.permanent_impact == -second.permanent_impact
    assert second.mid_price_after == pytest.approx(config.initial_price, rel=1e-14)
    assert not first.price_cap_hit and not second.price_cap_hit


def test_net_share_permanent_impact_depends_on_net_not_gross_flow() -> None:
    config = replace(
        _small_config(), permanent_impact_basis="net_shares",
        participation_pressure_exponent=2.0,
    )
    quotes = [
        QuasiOrderBook.initialize(config).quote(
            submitted_orders=orders, public_news_impact=0.0,
            realized_volatility=0.02,
        )
        for orders in (np.array([200.0, -100.0]), np.array([100.0, 0.0]))
    ]
    assert quotes[0].permanent_impact == quotes[1].permanent_impact
    assert quotes[0].transient_impact != quotes[1].transient_impact


def test_net_share_impact_preserves_transient_liquidity_feedback() -> None:
    config = replace(_small_config(), liquidity_depth_basis="notional")
    quotes = [
        QuasiOrderBook.initialize(replace(config, permanent_impact_basis=basis)).quote(
            submitted_orders=np.array([200.0, -100.0]),
            public_news_impact=0.01, realized_volatility=0.04,
            realized_volatility_reference=0.01,
        )
        for basis in ("participation", "net_shares")
    ]
    for field in ("depth", "order_flow_imbalance", "order_flow_surprise",
                  "transient_impact", "transient_impact_change"):
        assert getattr(quotes[0], field) == getattr(quotes[1], field)


def test_net_share_impact_is_invariant_to_share_unit_scaling() -> None:
    config = replace(_small_config(), permanent_impact_basis="net_shares")
    quotes = [
        QuasiOrderBook.initialize(replace(
            config, liquidity_scale=config.liquidity_scale * scale,
        )).quote(
            submitted_orders=np.array([200.0, -100.0]) * scale,
            public_news_impact=0.0, realized_volatility=0.02,
        )
        for scale in (1.0, 10.0)
    ]
    assert quotes[0].permanent_impact == quotes[1].permanent_impact
    assert quotes[0].mid_price_after == pytest.approx(quotes[1].mid_price_after)
    zero = QuasiOrderBook.initialize(config).quote(
        submitted_orders=np.array([100.0, -100.0]), public_news_impact=0.0,
        realized_volatility=0.02,
    )
    assert zero.permanent_impact == 0.0
