"""Stage D options: decreasing step size and the own-signal tercile state (research contract section 19)."""

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import training
from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.learning import batch_update, context_states, load_frozen, save_frozen
from abm.tabular import CompanyQLearner


def _context(**overrides):
    values = dict(financing_exposure=np.full(6, .5), prices=np.array([.4, .9, 1.2, 2.0, 1.0, .6]),
                  inflation=np.zeros(6), marginal_cost=np.full(6, .05), real_cost_scale=.05,
                  max_change_per_period=.25, pending_due_days=np.full(6, -1.0),
                  pending_cash_amount=np.zeros(6), decision_days=60)
    values.update(overrides)
    return context_states(**values)


def test_defaults_leave_serialized_config_unchanged():
    payload = MarketConfig().to_dict()
    assert "firm_learning_context" not in payload and "firm_learning_rate_power" not in payload
    config = MarketConfig(firm_learning_context="signal_tercile", firm_learning_rate_power=.85)
    restored = MarketConfig.from_dict(config.to_dict())
    assert restored.firm_learning_context == "signal_tercile" and restored.firm_learning_rate_power == .85


@pytest.mark.parametrize("field,value", [("firm_learning_context", "level"), ("firm_learning_rate_power", 1.5),
                                         ("firm_learning_rate_power", -.1)])
def test_invalid_learning_options_are_rejected(field, value):
    with pytest.raises(ValueError):
        MarketConfig(**{field: value})


def test_power_one_is_the_running_mean_and_power_zero_is_unchanged():
    rewards = [.3, -.1, .7, .2]
    learner = CompanyQLearner(groups=6, actions=3)
    learner.learning_rate_power = 1.0
    constant = CompanyQLearner(groups=6, actions=3)
    expected = constant.values[0, 0, 1]
    for reward in rewards:
        for table in (learner, constant):
            batch_update(table, np.array([0]), np.array([0]), np.array([0]), np.array([1]), np.array([reward]),
                         np.array([0]), np.array([0]), np.ones((1, 3), dtype=bool), elapsed_days=60,
                         annual_discount_rate=.05, terminated=np.array([True]))
        expected = expected + .1 * (reward - expected)
    assert learner.values[0, 0, 1] == pytest.approx(np.mean(rewards), abs=1e-15)
    assert constant.values[0, 0, 1] == expected


def test_batched_step_uses_count_after_the_batch():
    learner = CompanyQLearner(groups=6, actions=3)
    learner.learning_rate_power = .85
    learner.counts[0, 0, 1] = 4
    learner.values[0, 0, 1] = 1.0
    batch_update(learner, np.array([0, 1]), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
                 np.array([1, 1]), np.array([.0, .2]), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
                 np.ones((2, 3), dtype=bool), elapsed_days=60, annual_discount_rate=.05, terminated=np.ones(2, dtype=bool))
    step = 2 * 6 ** -.85
    assert learner.values[0, 0, 1] == pytest.approx(1.0 + step * (.1 - 1.0), rel=1e-14)
    assert learner.counts[0, 0, 1] == 6


def test_signal_tercile_replaces_only_the_first_feature():
    exposure = _context()
    signal = np.array([-1.0, 2.0, .5, .1, -.3, 1.0])
    tercile = _context(own_signal=signal)
    assert np.array_equal(exposure % 27, tercile % 27)
    ranks = np.argsort(np.argsort(signal))
    assert np.array_equal(tercile // 27, ranks // 2)
    with pytest.raises(ValueError):
        _context(own_signal=signal[:5])


def test_frozen_table_keeps_the_power(tmp_path):
    learner = CompanyQLearner(groups=6, actions=7)
    learner.learning_rate_power = .85
    save_frozen(tmp_path / "model.npz", learner, {"purpose": "unit_test"})
    restored, _ = load_frozen(tmp_path / "model.npz")
    assert restored.learning_rate_power == .85
    plain = CompanyQLearner(groups=6, actions=7)
    save_frozen(tmp_path / "plain.npz", plain, {"purpose": "unit_test"})
    assert load_frozen(tmp_path / "plain.npz")[0].learning_rate_power == 0.0


def _small(**overrides):
    values = dict(n_investors=30, n_firms=9, trading_days=360, burn_in_days=0, initial_impact_share=.2,
                  firm_policy="q_learning", firm_action_fractions=(.04, .2, 1.0))
    return MarketConfig(**{**values, **overrides})


def test_market_rejects_a_learner_with_another_power():
    domain = load_domain("configs/domains/esg.json")
    learner = CompanyQLearner(groups=6, actions=7, epsilon=1.0)
    with pytest.raises(ValueError, match="power"):
        DisclosureMarket(_small(firm_learning_rate_power=.85), domain, learner=learner, training=True)


def test_training_world_with_both_options_is_deterministic():
    domain = load_domain("configs/domains/esg.json")
    config = _small(firm_learning_rate_power=.85, firm_learning_context="signal_tercile")
    fingerprints, tables = [], []
    for _ in range(2):
        learner = CompanyQLearner(groups=6, actions=7, epsilon=1.0)
        learner.learning_rate_power = .85
        result = DisclosureMarket(config, domain, learner=learner, training=True).run()
        fingerprints.append(result.fingerprint())
        tables.append(learner.values.copy())
    assert fingerprints[0] == fingerprints[1] and np.array_equal(tables[0], tables[1])
    assert set(np.unique(result.learning_states[1:] // 27)) <= {0, 1, 2}
    default = DisclosureMarket(replace(config, firm_learning_context="exposure"), domain,
                               learner=_power(CompanyQLearner(groups=6, actions=7, epsilon=1.0)), training=True).run()
    assert not np.array_equal(default.learning_states, result.learning_states)


def _power(learner):
    learner.learning_rate_power = .85
    return learner


def test_training_budget_records_the_step_rule(tmp_path):
    domain = load_domain("configs/domains/esg.json")
    config = _small(firm_learning_rate_power=.85, n_firms=6, trading_days=180)
    summary = training.train_and_evaluate(config, domain, None, tmp_path / "out", training_seeds=[1],
                                          validation_seeds=[2])
    assert summary["budget"]["learning_rate_power"] == .85
    frozen, _ = load_frozen(tmp_path / "out" / "frozen_model.npz")
    assert frozen.learning_rate_power == .85
    assert summary["status"] == "complete"


def test_cost_budget_is_omitted_by_default_and_validated():
    assert "firm_cost_budget_fraction" not in MarketConfig().to_dict()
    assert MarketConfig.from_dict(MarketConfig(firm_cost_budget_fraction=.01).to_dict()).firm_cost_budget_fraction == .01
    with pytest.raises(ValueError):
        MarketConfig(firm_cost_budget_fraction=-.01)


@pytest.mark.parametrize("policy", ["q_learning", "npv_discrete"])
def test_cost_budget_bounds_every_shared_grid_choice(policy):
    domain = load_domain("configs/domains/esg.json")
    budget = 2e-4
    config = _small(firm_policy=policy, firm_cost_budget_fraction=budget)
    kwargs = {}
    if policy == "q_learning":
        kwargs = {"learner": CompanyQLearner(groups=6, actions=7, epsilon=1.0), "training": True}
    market = DisclosureMarket(config, domain, **kwargs)
    result = market.run()
    assert result.cost_fraction.max() <= budget
    assert result.valid_actions[1:, :, 2].all()
    if policy == "q_learning":
        unconstrained = DisclosureMarket(replace(config, firm_cost_budget_fraction=0.0), domain,
                                         learner=CompanyQLearner(groups=6, actions=7, epsilon=1.0), training=True).run()
        assert unconstrained.cost_fraction.max() > budget
