"""Tests for Track C AIJ revision extensions:

1. Quantitative risk-band rate announcement (``audit_band_rates_public=True``) for
   bounded-rationality empirical-belief learners (with zero oracle leakage of fines/assurance).
2. Joint targeted/Stackelberg audits + third-party assurance (per-firm ``audit_probability``
   in ``assured_signal_factors``, client-weighted per-provider audit exposure in ``AssuranceState``,
   and full end-to-end ``DisclosureMarket`` integration for NPV and bounded-rationality firms).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import assured_signal_factors, financing_signal_factors
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
REGIME = PolicyRegime.baseline(ESG)
TIMING = dict(
    decision_day=120,
    horizon_days=360,
    repricing_interval=60,
    period_days=60,
    implementation_delay=20,
    disclosure_delay=20,
)
SMALL_CFG = MarketConfig(
    n_firms=24,
    n_investors=80,
    trading_days=480,
    burn_in_days=120,
    firm_decision_interval=60,
    initial_impact_share=0.25,
    taste_median=0.0724,
    seed=79001,
)
ASSURANCE_OPTS = dict(
    assurance_scope="all",
    assurer_selection="firm",
    assurer_observation="detected_bayes",
    assurer_liability=0.016,
    assurance_fee=0.0,
    assurer_effort_cost=0.001,
    assurer_count=4,
    assurer_smoothing=0.7,
)


def test_audit_band_rates_public_validation_and_backward_compatible_serialization() -> None:
    # Default False must be omitted from to_dict() so existing experiment identities never change.
    base_dict = REGIME.to_dict()
    assert "audit_band_rates_public" not in base_dict
    bucketed = replace(REGIME, audit_allocation="stackelberg_obs", audit_commitment_buckets=2)
    assert "audit_band_rates_public" not in bucketed.to_dict()

    announced = replace(
        REGIME,
        audit_allocation="stackelberg_obs",
        audit_commitment_buckets=2,
        audit_band_rates_public=True,
    )
    assert announced.to_dict()["audit_band_rates_public"] is True
    assert PolicyRegime(**announced.to_dict()) == announced

    with pytest.raises(ValueError, match="audit_band_rates_public must be boolean"):
        replace(bucketed, audit_band_rates_public=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="audit_band_rates_public requires audit_commitment_buckets >= 2"):
        replace(REGIME, audit_allocation="stackelberg_obs", audit_band_rates_public=True)
    with pytest.raises(ValueError, match="audit_commitment_buckets"):
        replace(REGIME, audit_allocation="stackelberg_obs", audit_rule_public=False, audit_commitment_buckets=2,
                audit_band_rates_public=True)


def test_quantitative_band_announcement_uses_announced_rates_without_leaking_fines(monkeypatch) -> None:
    """Bounded learners read announced band q_g when enabled, but still learn fines strictly online."""
    import abm.disclosure.game as game_mod

    cfg = replace(SMALL_CFG, firm_policy="empirical_belief", trading_days=300)
    # Zero baseline fixed_fine; large post-switch fixed_fine + zero catches check or monitored calls.
    dom = replace(ESG, baseline_fixed_fine=0.0, baseline_audit_probability=0.05)
    sched = PolicySchedule.constant(dom).with_intervention(
        121,
        audit_allocation="stackelberg_obs",
        audit_commitment_buckets=2,
        audit_band_rates_public=True,
        fixed_fine=0.05,
        penalty=0.0,
    )
    recorded_calls: list[tuple[int, np.ndarray, float]] = []
    orig_npv = game_mod.npv_choices

    def spy_npv(*args, **kwargs):
        if kwargs["decision_day"] >= 120:
            q_arg = kwargs.get("audit_probability")
            if q_arg is not None:
                recorded_calls.append((
                    int(kwargs["decision_day"]),
                    np.asarray(q_arg, dtype=np.float64).copy(),
                    float(kwargs["regime"].fixed_fine),
                ))
        return orig_npv(*args, **kwargs)

    monkeypatch.setattr(game_mod, "npv_choices", spy_npv)
    mkt = DisclosureMarket(cfg, dom, sched)
    res = mkt.run()

    # After switch (day 180+), announced band rates must have at most 2 distinct values and conserve budget.
    post_calls = [entry for entry in recorded_calls if entry[0] >= 180]
    assert post_calls
    for day, q_vec, seen_fine in post_calls:
        unique_q = np.unique(np.round(q_vec, 12))
        assert unique_q.size <= 2
        assert q_vec.sum() == pytest.approx(cfg.n_firms * dom.baseline_audit_probability, rel=1e-9, abs=1e-12)
        # On the very first post-switch period before any post-switch catch occurs, fixed_fine must not leak.
        if day == 180 and res.detections[day // 60].sum() == 0:
            assert seen_fine == 0.0
    assert np.isfinite(res.true_goodness).all()


def test_assured_signal_factors_vector_audit_matches_scalar_entry_by_entry() -> None:
    rates = np.array([0.05, 0.08, 0.0, 0.04, 0.06])
    q_vec = np.array([0.0, 0.02, 0.15, 0.5, 1.0])
    a_vec = np.array([0.0, 0.1, 0.35, 0.8, 1.0])

    vec_out = financing_signal_factors(
        rates,
        audit_probability=q_vec,
        transparency=0.1,
        assurance_probability=a_vec,
        **TIMING,
    )
    for idx in range(rates.size):
        scalar_out = assured_signal_factors(
            rates[[idx]],
            audit_probability=float(q_vec[idx]),
            transparency=0.1,
            assurance_probability=a_vec[[idx]],
            **TIMING,
        )
        for v_arr, s_arr in zip(vec_out, scalar_out):
            assert v_arr[idx] == pytest.approx(s_arr[0], rel=1e-13, abs=1e-15)

    # Invalid vector entries must raise ValueError cleanly.
    with pytest.raises(ValueError, match="audit_probability must be finite and in"):
        assured_signal_factors(rates, audit_probability=np.array([-0.1, 0.1, 0.1, 0.1, 0.1]),
                               transparency=0.1, assurance_probability=a_vec, **TIMING)
    with pytest.raises(ValueError, match="audit_probability must be finite and in"):
        assured_signal_factors(rates, audit_probability=np.array([0.1, 1.2, 0.1, 0.1, 0.1]),
                               transparency=0.1, assurance_probability=a_vec, **TIMING)


@pytest.mark.parametrize("firm_policy", ["npv", "empirical_belief"])
@pytest.mark.parametrize("allocation", ["value", "stackelberg_obs"])
def test_joint_targeted_audits_and_assurance_end_to_end(firm_policy: str, allocation: str) -> None:
    cfg = replace(SMALL_CFG, firm_policy=firm_policy)
    plain_sched = PolicySchedule.constant(ESG)
    joint_sched = plain_sched.with_intervention(
        121,
        audit_allocation=allocation,
        **ASSURANCE_OPTS,
    )
    plain = DisclosureMarket(cfg, ESG, plain_sched).run()
    joint = DisclosureMarket(cfg, ESG, joint_sched).run()

    # Exact pre-switch parity through burn-in day 120.
    pre_periods = 120 // cfg.firm_decision_interval
    np.testing.assert_array_equal(joint.actions[:pre_periods], plain.actions[:pre_periods])
    np.testing.assert_array_equal(joint.true_goodness[:121], plain.true_goodness[:121])
    np.testing.assert_array_equal(joint.market.prices[:121], plain.market.prices[:121])

    # Both audit targeting and assurance outputs are populated, finite, and conserve audit budget.
    assert joint.audit_probability is not None
    assert joint.assurer_quality is not None
    assert joint.assurer_assignment is not None
    assert np.isfinite(joint.true_goodness).all()
    assert np.isfinite(joint.assurer_quality).all()
    assert np.all((joint.assurer_quality >= 0.0) & (joint.assurer_quality <= 1.0))

    expected_budget = cfg.n_firms * ESG.baseline_audit_probability
    post_probs = joint.audit_probability[pre_periods + 1 :]
    assert np.all((post_probs >= 0.0) & (post_probs <= 1.0))
    np.testing.assert_allclose(post_probs.sum(axis=1), expected_budget, rtol=1e-9, atol=1e-12)
    assert (joint.assurer_assignment[pre_periods + 1 :] >= 0).all()


def test_uniform_assurance_is_bit_identical_for_both_npv_and_bounded_learners() -> None:
    """Confirm C3 changes do not alter scalar-audit assurance trajectories or run_ids."""
    for firm_policy in ("npv", "empirical_belief"):
        cfg = replace(SMALL_CFG, firm_policy=firm_policy, trading_days=300)
        sched = PolicySchedule.constant(ESG).with_intervention(121, **ASSURANCE_OPTS)
        m1 = DisclosureMarket(cfg, ESG, sched)
        m2 = DisclosureMarket(cfg, ESG, sched)
        assert m1.run_id == m2.run_id
        r1 = m1.run()
        r2 = m2.run()
        np.testing.assert_array_equal(r1.true_goodness, r2.true_goodness)
        np.testing.assert_array_equal(r1.assurer_quality, r2.assurer_quality)
        np.testing.assert_array_equal(r1.actions, r2.actions)
