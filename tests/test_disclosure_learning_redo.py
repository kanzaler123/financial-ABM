"""Stage D.3 options: the improvement-stock state and the training entry's validation rules and archive (contract section 23)."""

import json

import numpy as np
import pytest

from abm.disclosure import training
from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import IMPROVE, DisclosureMarket
from abm.disclosure.learning import context_states
from abm.tabular import CompanyQLearner


def _context(**overrides):
    values = dict(financing_exposure=np.full(6, .5), prices=np.array([.4, .9, 1.2, 2.0, 1.0, .6]),
                  inflation=np.array([0, .1, .3, 0, 0, .2]), marginal_cost=np.array([.02, .05, .2, .05, .05, .05]),
                  real_cost_scale=.05, max_change_per_period=.25, pending_due_days=np.full(6, -1.0),
                  pending_cash_amount=np.zeros(6), decision_days=60,
                  own_signal=np.array([-1.0, 2.0, .5, .1, -.3, 1.0]))
    values.update(overrides)
    return context_states(**values)


def test_config_accepts_and_serializes_the_new_context():
    config = MarketConfig(firm_learning_context="improvement_stock")
    assert MarketConfig.from_dict(config.to_dict()).firm_learning_context == "improvement_stock"
    assert "firm_learning_context" not in MarketConfig().to_dict()


def test_improvement_stock_replaces_only_the_cost_tier():
    tercile = _context()
    stock = np.array([0, 1e-13, .01, .2499999, .25, 3.0])
    states = _context(improvement_stock=stock)
    assert np.array_equal(states // 9, tercile // 9)
    assert np.array_equal(states % 3, tercile % 3)
    assert np.array_equal(states // 3 % 3, [0, 0, 1, 1, 2, 2])
    assert not np.array_equal(tercile // 3 % 3, states // 3 % 3)


@pytest.mark.parametrize("stock", [np.zeros(5), np.array([0, 0, -.1, 0, 0, 0])])
def test_invalid_improvement_stock_is_rejected(stock):
    with pytest.raises(ValueError):
        _context(improvement_stock=stock)


def _small(**overrides):
    values = dict(n_investors=30, n_firms=9, trading_days=360, burn_in_days=0, initial_impact_share=.2,
                  firm_policy="q_learning", firm_action_fractions=(.04, .2, 1.0), firm_learning_rate_power=.85,
                  firm_learning_context="improvement_stock")
    return MarketConfig(**{**values, **overrides})


def _learner():
    learner = CompanyQLearner(groups=6, actions=7, epsilon=1.0)
    learner.learning_rate_power = .85
    return learner


def test_market_state_tracks_committed_real_improvement():
    domain = load_domain("configs/domains/esg.json")
    config = _small()
    results = [DisclosureMarket(config, domain, learner=_learner(), training=True).run() for _ in range(2)]
    result = results[0]
    assert results[0].fingerprint() == results[1].fingerprint()
    committed = np.where(result.actions == IMPROVE, result.amounts, 0.0)
    stock = np.vstack([np.zeros((1, committed.shape[1])), np.cumsum(committed, axis=0)[:-1]])
    expected = np.digitize(stock, (1e-12, domain.max_change_per_period))
    periods = slice(1, None)
    assert np.array_equal(result.learning_states[periods] // 3 % 3, expected[periods])
    assert len(np.unique(expected[periods])) > 1
    tercile = DisclosureMarket(_small(firm_learning_context="signal_tercile"), domain,
                               learner=_learner(), training=True).run()
    # The first decision has no history, so only the third feature can differ there.
    first = 1
    assert np.array_equal(tercile.learning_states[first] // 9, result.learning_states[first] // 9)
    assert np.array_equal(tercile.learning_states[first] % 3, result.learning_states[first] % 3)


def test_training_entry_validation_rules_and_compact_archive(tmp_path):
    domain = load_domain("configs/domains/esg.json")
    config = _small(n_firms=6, trading_days=180)
    summary = training.train_and_evaluate(config, domain, None, tmp_path / "out", training_seeds=[1],
                                          validation_seeds=[2], validation_rules=["q_learning"], archive="compact")
    assert summary["status"] == "complete"
    assert summary["budget"]["validation_rules"] == ["q_learning"] and summary["budget"]["archive"] == "compact"
    assert sorted(path.name for path in (tmp_path / "out" / "validation" / "seed_2").iterdir()) == ["q_learning"]
    for folder in (tmp_path / "out" / "training" / "seed_1", tmp_path / "out" / "validation" / "seed_2" / "q_learning"):
        with np.load(folder / "result.npz") as result:
            assert "true_goodness" not in result.files and "learning_states" in result.files
        assert "compact_archive" in json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert summary["validation"]["coverage"]["status"] in ("PASS", "FAIL")


def test_training_entry_defaults_keep_the_budget(tmp_path):
    domain = load_domain("configs/domains/esg.json")
    config = _small(n_firms=6, trading_days=180, firm_learning_context="signal_tercile")
    summary = training.train_and_evaluate(config, domain, None, tmp_path / "out", training_seeds=[1], validation_seeds=[2])
    assert summary["budget"]["validation_rules"] == ["q_learning", "npv_discrete", "npv"]
    assert "archive" not in summary["budget"]
    assert len([row for row in summary["records"] if row["phase"] == "validation"]) == 3


@pytest.mark.parametrize("rules,archive", [(["npv"], "full"), (["npv", "q_learning"], "full"),
                                           (["q_learning"], "small")])
def test_training_entry_rejects_bad_options(tmp_path, rules, archive):
    domain = load_domain("configs/domains/esg.json")
    with pytest.raises(ValueError):
        training.train_and_evaluate(_small(n_firms=6, trading_days=180), domain, None, tmp_path / "out",
                                    training_seeds=[1], validation_seeds=[2], validation_rules=rules, archive=archive)
    assert not (tmp_path / "out").exists()
