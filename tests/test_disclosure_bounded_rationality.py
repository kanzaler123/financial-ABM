"""Tests for zero-leakage empirical-belief bounded rationality (Phase 3 / L2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.bounded import EmpiricalBeliefLearner
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


def _small_cfg(**overrides) -> MarketConfig:
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
        firm_policy="empirical_belief",
    )
    params.update(overrides)
    return MarketConfig(**params)


def test_empirical_belief_config_defaults_omitted_and_validated() -> None:
    plain = MarketConfig()
    assert "firm_belief_decay" not in plain.to_dict()

    cfg = _small_cfg(firm_belief_decay=0.25)
    d = cfg.to_dict()
    assert d["firm_policy"] == "empirical_belief"
    assert d["firm_belief_decay"] == 0.25
    assert MarketConfig.from_dict(d) == cfg

    with pytest.raises(ValueError, match="firm_belief_decay"):
        MarketConfig(firm_belief_decay=0.0)
    with pytest.raises(ValueError, match="firm_belief_decay"):
        MarketConfig(firm_belief_decay=1.5)


def test_empirical_belief_strict_zero_leakage_and_thompson_selection() -> None:
    """Beliefs update ONLY from realized audits/catches/checks (never peeking at true policy params)."""
    learner = EmpiricalBeliefLearner(n_tiers=3, n_providers=4, decay=0.2)
    sc = np.array([0] * 10 + [1] * 10 + [2] * 20, dtype=np.int64)
    zeros = np.zeros(40, dtype=np.float64)

    # 1. With zero catches, emp_fixed_fine_money stays 0 regardless of unobserved statutory fine
    sel = np.zeros(40, dtype=bool)
    sel[:3] = True  # 30% audit rate in large tier, 0% in mid/small
    learner.observe_audits(
        size_class=sc,
        selected=sel,
        caught=np.zeros(40, dtype=bool),
        realized_fixed_fine_money=zeros,
        realized_proportional_rate=zeros,
    )
    assert learner.emp_fixed_fine_money == 0.0
    assert learner.tier_audit_probability[0] > learner.tier_audit_probability[2]

    # 2. First catch reveals statutory fine; later 4x catch smoothly raises belief
    caught = np.zeros(40, dtype=bool)
    caught[0] = True
    learner.observe_audits(
        size_class=sc,
        selected=sel,
        caught=caught,
        realized_fixed_fine_money=np.where(caught, 1e-4, 0.0),
        realized_proportional_rate=zeros,
    )
    assert learner.emp_fixed_fine_money == pytest.approx(1e-4)
    learner.observe_audits(
        size_class=sc,
        selected=sel,
        caught=caught,
        realized_fixed_fine_money=np.where(caught, 4e-4, 0.0),
        realized_proportional_rate=zeros,
    )
    assert learner.emp_fixed_fine_money == pytest.approx(0.8 * 1e-4 + 0.2 * 4e-4)

    # 3. Assurance beliefs learn strict vs lax providers purely from checks; Thompson sampling prefers lax
    client_of = np.repeat(np.arange(4, dtype=np.int64), 10)
    checked_inf = np.ones(40, dtype=bool)
    corrected = np.zeros(40, dtype=bool)
    # Provider 0 catches 0/10 (lax); Providers 1-3 catch 8/10 (strict)
    corrected[10:18] = True
    corrected[20:28] = True
    corrected[30:38] = True
    for _ in range(3):
        learner.observe_assurance_period(
            checked_inflated=checked_inf, corrected=corrected, client_of=client_of,
        )
    means = learner.provider_quality_mean
    assert means[0] < 0.05
    assert np.all(means[1:] > 0.65)

    rng = np.random.default_rng(7)
    chosen = learner.choose_providers_thompson(
        mandated=np.ones(40, dtype=bool),
        current_providers=client_of,
        value_fn=lambda a: 1.0 - a,  # strictly prefers lower check probability
        rng=rng,
    )
    # Overwhelming majority of inflating firms select empirically laxest provider 0
    assert float((chosen == 0).mean()) > 0.90


def test_empirical_belief_runs_across_regimes_with_preswitch_parity() -> None:
    cfg = _small_cfg()
    base = DisclosureMarket(cfg, WASH, PolicySchedule.constant(WASH)).run()
    regimes = [
        {"transparency": 0.6},
        {"fixed_fine": 4.0 * 4.56e-5},
        {"audit_allocation": "value"},
        {"audit_allocation": "stackelberg_obs"},
        ASSURE_16,
        {**ASSURE_16, "assurer_selection": "firm"},
        {**ASSURE_16, "assurer_selection": "firm", "assurer_smoothing": 0.7},
    ]
    for chg in regimes:
        sched = PolicySchedule.constant(WASH).with_intervention(181, **chg)
        mkt = DisclosureMarket(cfg, WASH, sched)
        res = mkt.run()
        np.testing.assert_array_equal(res.true_goodness[:181], base.true_goodness[:181])
        np.testing.assert_array_equal(res.inflation[:181], base.inflation[:181])
        np.testing.assert_array_equal(res.market.prices[:181], base.market.prices[:181])
        assert np.isfinite(res.true_goodness).all()
        assert np.isfinite(res.inflation).all()
        assert mkt.empirical_belief is not None
        assert mkt.empirical_belief.audit_updates > 0


def test_empirical_belief_maintains_under_zero_taste() -> None:
    cfg = _small_cfg(initial_impact_share=0.0)
    res = DisclosureMarket(cfg, WASH, PolicySchedule.constant(WASH)).run()
    assert np.all(res.actions == MAINTAIN)


def test_empirical_belief_baseline_prior_does_not_leak_intervention() -> None:
    """Market initializes priors from domain.baseline_* only; post-switch regime is never peeked."""
    cfg = _small_cfg(trading_days=240, burn_in_days=180)
    sched = PolicySchedule.constant(WASH).with_intervention(
        181,
        fixed_fine=4.0 * WASH.baseline_fixed_fine,
        audit_probability=0.0,  # zero post-switch audits -> zero post-switch catches -> cannot learn 4x
    )
    mkt = DisclosureMarket(cfg, WASH, sched)
    mkt.run()
    assert mkt.empirical_belief is not None
    # Because zero catches occurred after day 181, perceived fine stays at baseline 1x, never jumping to 4x
    assert mkt.empirical_belief.emp_fixed_fine_money == pytest.approx(WASH.baseline_fixed_fine)


def test_empirical_belief_voluntary_assurance_rejected() -> None:
    cfg = _small_cfg()
    sched = PolicySchedule.constant(WASH).with_intervention(
        181, **{**ASSURE_16, "assurance_scope": "voluntary"},
    )
    with pytest.raises(ValueError, match="voluntary assurance"):
        DisclosureMarket(cfg, WASH, sched)


def test_empirical_belief_archives_cleanly_via_runner(tmp_path) -> None:
    from abm.disclosure.runner import run_and_archive

    cfg = _small_cfg(trading_days=240, burn_in_days=180)
    manifest = run_and_archive(
        cfg, WASH, tmp_path / "emp_run", PolicySchedule.constant(WASH),
        purpose="development", archive="compact",
    )
    assert manifest["status"] == "complete"
    assert (tmp_path / "emp_run" / "metrics.json").exists()
    assert not (tmp_path / "emp_run" / "learning.json").exists()


def test_empirical_belief_learns_public_risk_bands_without_peeking() -> None:
    """EmpiricalBeliefLearner updates per-band audit rates strictly from realized audits in each public band."""
    learner = EmpiricalBeliefLearner(n_tiers=3, n_providers=4, decay=0.2)
    sc = np.zeros(30, dtype=np.int64)
    bands = np.array([0] * 20 + [1] * 6 + [2] * 4, dtype=np.int64)
    sel = np.zeros(30, dtype=bool)
    sel[26:] = True  # all 4 high-risk band firms audited; 0 low/mid audited
    zeros = np.zeros(30, dtype=np.float64)
    learner.observe_audits(
        size_class=sc, selected=sel, caught=np.zeros(30, dtype=bool),
        realized_fixed_fine_money=zeros, realized_proportional_rate=zeros,
        public_band=bands, n_bands=3,
    )
    q_band = learner.band_audit_probabilities(bands, 3)
    assert q_band[29] > q_band[20] > q_band[0] or q_band[29] > q_band[0] + 0.10

    cfg = _small_cfg(trading_days=360, burn_in_days=120)
    sched = PolicySchedule.constant(WASH).with_intervention(
        121, audit_allocation="stackelberg_obs", audit_commitment_buckets=3,
    )
    mkt = DisclosureMarket(cfg, WASH, sched)
    res = mkt.run()
    assert np.isfinite(res.true_goodness).all()
    assert mkt.empirical_belief is not None and mkt.empirical_belief.band_alpha is not None
    assert mkt.empirical_belief.band_alpha.size == 3
