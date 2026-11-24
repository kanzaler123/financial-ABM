"""Tests for partial-observability Bayesian assurance provider updating (Track 5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
WASH = replace(
    ESG,
    baseline_fixed_fine=4.56e-5,
    baseline_penalty=0.0,
    baseline_audit_probability=0.01,
    baseline_transparency=0.1,
    natural_decay=0.2,
)
REGIME = PolicyRegime(audit_probability=0.01, penalty=0.0, transparency=0.1, subsidy=0.0)
ASSURED = dict(
    assurance_scope="all",
    assurer_effort_cost=2.172138477935219e-06,
    assurer_liability=0.00036491926429311675,
    assurance_fee=5.430346194838047e-07,
)


def test_partial_obs_policy_defaults_omitted_and_validated() -> None:
    payload = REGIME.to_dict()
    assert not {"assurer_observation", "assurer_prior_share", "assurer_prior_strength"} & set(payload)

    noisy = replace(REGIME, **ASSURED, assurer_observation="detected_bayes", assurer_prior_share=0.25, assurer_prior_strength=3.0)
    d = noisy.to_dict()
    assert d["assurer_observation"] == "detected_bayes"
    assert d["assurer_prior_share"] == 0.25
    assert d["assurer_prior_strength"] == 3.0
    assert PolicyRegime(**d) == noisy

    # Exact default is omitted even when assurance_scope != "none"
    exact = replace(REGIME, **ASSURED)
    assert "assurer_observation" not in exact.to_dict()

    with pytest.raises(ValueError, match="assurer_observation"):
        replace(exact, assurer_observation="oracle")
    with pytest.raises(ValueError, match="assurer_prior_share"):
        replace(exact, assurer_prior_share=1.5)
    with pytest.raises(ValueError, match="assurer_prior_strength"):
        replace(exact, assurer_prior_strength=0.0)
    with pytest.raises(ValueError, match="need an assurance_scope"):
        replace(REGIME, assurer_observation="detected_bayes")


def test_detected_bayes_updates_solely_from_observed_events_not_latent_inflation() -> None:
    """In year 1 and later years, detected_bayes matches exact empirical-Bayes formula from corrections + liability."""
    cfg = MarketConfig(
        n_investors=50,
        n_firms=40,
        trading_days=780,
        burn_in_days=180,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=63001,
        intrinsic_value_log_sd=0.915,
        financing_exposure_fraction=0.5,
        financing_spread=0.05,
        financing_repricing_interval=60,
        financing_horizon_days=1260,
        firm_discount_rate=0.05,
    )
    prior = 0.22
    strength = 2.5
    sched = PolicySchedule.constant(WASH).with_intervention(
        181,
        **ASSURED,
        assurer_selection="assigned",
        assurer_observation="detected_bayes",
        assurer_prior_share=prior,
        assurer_prior_strength=strength,
    )
    base = DisclosureMarket(cfg, WASH, PolicySchedule.constant(WASH)).run()
    res = DisclosureMarket(cfg, WASH, sched).run()

    # Pre-switch history bit-identical to baseline
    np.testing.assert_array_equal(res.true_goodness[:181], base.true_goodness[:181])
    np.testing.assert_array_equal(res.inflation[:181], base.inflation[:181])

    first = 240 // 60  # period 4: first assurance year
    lam = sched.intervention.assurer_slope
    # Year 1 uses prior share directly (zero access to latent inflation!)
    np.testing.assert_allclose(res.assurer_inflation_share[first], prior, atol=1e-15)
    np.testing.assert_allclose(res.assurer_quality[first], min(1.0, lam * prior), atol=1e-15)

    # Verify year 2 exact empirical-Bayes update from year 1's realized corrections + liability notices
    second = first + 4
    year1 = slice(first, second)
    assign1 = res.assurer_assignment[year1]
    corr1 = res.assurance_corrections[year1]
    liab1 = res.assurer_liability_events[slice(first + 1, second + 1)].sum(axis=0)
    a1 = res.assurer_quality[first]
    qbar = WASH.baseline_audit_probability
    p_obs = a1 + (1.0 - a1) * qbar
    for k in range(4):
        n_k = int((assign1 == k).sum())
        c_k = int((corr1 & (assign1 == k)).sum())
        e_k = int(liab1[k])
        expected_s = np.clip((strength * prior + c_k + e_k) / (strength + n_k * p_obs[k]), 0.0, 1.0)
        assert res.assurer_inflation_share[second, k] == pytest.approx(expected_s, abs=1e-12)
        assert res.assurer_quality[second, k] == pytest.approx(min(1.0, lam * expected_s), abs=1e-12)


def test_detected_bayes_supports_firm_choice_and_preserves_escape_incentive() -> None:
    """Firm-chosen assurance runs cleanly under partial observability with bounded qualities in [0, 1]."""
    cfg = MarketConfig(
        n_investors=50,
        n_firms=40,
        trading_days=780,
        burn_in_days=180,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=63002,
        intrinsic_value_log_sd=0.915,
        financing_exposure_fraction=0.5,
        financing_spread=0.05,
        financing_repricing_interval=60,
        financing_horizon_days=1260,
        firm_discount_rate=0.05,
    )
    sched = PolicySchedule.constant(WASH).with_intervention(
        181,
        **ASSURED,
        assurer_selection="firm",
        assurer_observation="detected_bayes",
    )
    res = DisclosureMarket(cfg, WASH, sched).run()
    assert np.all((res.assurer_quality >= 0.0) & (res.assurer_quality <= 1.0))
    assert np.all((res.assurer_inflation_share >= 0.0) & (res.assurer_inflation_share <= 1.0))
    assert np.isfinite(res.true_goodness).all()
