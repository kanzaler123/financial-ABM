"""Tests for bounded-rationality firm policies: diff_q and peer_imitation (Track 4)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import MAINTAIN, DisclosureMarket
from abm.disclosure.policy import PolicySchedule

ESG = load_domain("configs/domains/esg.json")
WASH = replace(
    ESG,
    baseline_fixed_fine=4.56e-5,
    baseline_penalty=0.0,
    baseline_audit_probability=0.01,
    baseline_transparency=0.1,
    natural_decay=0.2,
)
ASSURE_16 = dict(
    assurance_scope="all",
    assurer_selection="assigned",
    assurer_liability=0.00036491926429311675,
    assurance_fee=5.430346194838047e-07,
    assurer_effort_cost=2.172138477935219e-06,
)


def _small_cfg(policy: str, **overrides) -> MarketConfig:
    params = dict(
        n_investors=50,
        n_firms=30,
        trading_days=600,
        burn_in_days=180,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=62001,
        intrinsic_value_log_sd=0.915,
        financing_exposure_fraction=0.5,
        financing_spread=0.05,
        financing_repricing_interval=60,
        financing_horizon_days=1260,
        firm_discount_rate=0.05,
        firm_policy=policy,
        firm_action_fractions=(0.04, 0.2, 1.0),
        firm_cost_budget_fraction=0.01,
    )
    params.update(overrides)
    return MarketConfig(**params)


def test_bounded_config_defaults_omitted_and_validated() -> None:
    plain = MarketConfig()
    payload = plain.to_dict()
    for key in (
        "firm_learning_alpha",
        "firm_learning_epsilon",
        "firm_imitation_intensity",
        "firm_imitation_update_fraction",
        "firm_imitation_exploration",
    ):
        assert key not in payload

    for pol in ("diff_q", "peer_imitation"):
        cfg = _small_cfg(pol, firm_learning_alpha=0.2, firm_imitation_exploration=0.08)
        d = cfg.to_dict()
        assert d["firm_policy"] == pol
        assert d["firm_learning_alpha"] == 0.2
        assert d["firm_imitation_exploration"] == 0.08
        assert MarketConfig.from_dict(d) == cfg

    with pytest.raises(ValueError, match="firm_learning_alpha"):
        MarketConfig(firm_learning_alpha=0.0)
    with pytest.raises(ValueError, match="firm_learning_epsilon"):
        MarketConfig(firm_learning_epsilon=1.5)
    with pytest.raises(ValueError, match="firm_imitation_update_fraction"):
        MarketConfig(firm_imitation_update_fraction=-0.1)


@pytest.mark.parametrize("policy", ["diff_q", "peer_imitation"])
def test_bounded_policies_run_across_all_enforcement_regimes_with_preswitch_parity(policy: str) -> None:
    cfg = _small_cfg(policy)
    base = DisclosureMarket(cfg, WASH, PolicySchedule.constant(WASH)).run()
    regimes = [
        {"transparency": 0.6},
        {"fixed_fine": 4.0 * 4.56e-5},
        {"audit_allocation": "value"},
        {"audit_allocation": "stackelberg_eq"},
        ASSURE_16,
        {**ASSURE_16, "assurer_selection": "firm"},
    ]
    for chg in regimes:
        sched = PolicySchedule.constant(WASH).with_intervention(181, **chg)
        mkt = DisclosureMarket(cfg, WASH, sched)
        res = mkt.run()
        # Bit-for-bit identical pre-switch history through day 180
        np.testing.assert_array_equal(res.true_goodness[:181], base.true_goodness[:181])
        np.testing.assert_array_equal(res.inflation[:181], base.inflation[:181])
        np.testing.assert_array_equal(res.market.prices[:181], base.market.prices[:181])
        assert np.isfinite(res.true_goodness).all()
        assert np.isfinite(res.inflation).all()
        # Variance reduction: difference-reward SD is orders of magnitude below raw 190 bp noise
        assert mkt.bounded_tracker is not None
        assert len(mkt.bounded_tracker.settled_returns) > 0
        reward_sd_bp = float(np.std(mkt.bounded_tracker.settled_returns) * 1e4)
        assert reward_sd_bp < 50.0


def test_diff_q_achieves_full_coverage_and_maintains_under_zero_taste() -> None:
    """With zero impact taste (B = 0), diff_q learns MAINTAIN as optimal (Q <= 0 for costly actions)."""
    cfg = _small_cfg(
        "diff_q",
        n_firms=60,
        trading_days=900,
        initial_impact_share=0.0,
        firm_learning_epsilon=0.05,
        firm_action_fractions=(0.04, 0.2, 0.5),
        firm_cost_budget_fraction=0.05,
    )
    mkt = DisclosureMarket(cfg, WASH, PolicySchedule.constant(WASH))
    res = mkt.run()
    assert mkt.diff_learner is not None
    # Every clean-state cell across all three tiers is visited
    assert (mkt.diff_learner.counts[:, 0, :] > 0).all()
    # In the final third of periods, MAINTAIN dominates (> 80% of decisions despite 5% exploration)
    tail_actions = res.actions[10:]
    assert float((tail_actions == MAINTAIN).mean()) > 0.80
