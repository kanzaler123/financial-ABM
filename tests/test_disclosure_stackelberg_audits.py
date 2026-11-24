"""Tests for Stackelberg optimal budget-constrained audit allocation (Proposition 5 / Track 3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import IMPROVE, INFLATE, DisclosureMarket, npv_choices
from abm.disclosure.policy import (
    STACKELBERG_ALLOCATIONS,
    PolicyRegime,
    PolicySchedule,
    audit_probabilities,
    bayesian_stackelberg_audit_probabilities,
    firm_deterrence_thresholds,
    stackelberg_audit_probabilities,
)

ESG = load_domain("configs/domains/esg.json")
WASH_DOMAIN = replace(
    ESG,
    baseline_fixed_fine=4.56e-5,
    baseline_penalty=0.0,
    baseline_audit_probability=0.01,
    baseline_transparency=0.1,
    natural_decay=0.2,
)
REGIME = PolicyRegime(audit_probability=0.01, penalty=0.0, transparency=0.1, subsidy=0.0, fixed_fine=4.56e-5)
TIMING = dict(
    discount_rate=0.05,
    decision_day=1260,
    horizon_days=1260,
    repricing_interval=60,
    period_days=60,
    implementation_delay=120,
    disclosure_delay=20,
)


def test_stackelberg_policy_serialization_and_validation() -> None:
    for alloc in STACKELBERG_ALLOCATIONS:
        reg = replace(REGIME, audit_allocation=alloc)
        payload = reg.to_dict()
        assert payload["audit_allocation"] == alloc
        assert "audit_jump_threshold" not in payload
        assert PolicyRegime(**payload) == reg
        with pytest.raises(ValueError, match="stackelberg_audit_probabilities"):
            audit_probabilities(reg, np.ones(10))


def test_deterrence_thresholds_match_npv_choices_exactly() -> None:
    """For every interior firm, q = q_det + eps deters (IMPROVE) and q = q_det - eps inflates (INFLATE)."""
    rng = np.random.default_rng(7)
    n = 30
    prices = rng.lognormal(0.0, 0.9, n)
    prices /= prices.mean()
    benefit = np.full(n, 4.4e-4)
    cost = 0.013 * rng.lognormal(0.0, 0.4, n)
    pi = np.where(np.arange(n) < 8, 0.0, 1.0)

    q_det = firm_deterrence_thresholds(
        prices=prices,
        benefit=benefit,
        marginal_cost=cost,
        new_case_prob=pi,
        regime=REGIME,
        wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
        max_change_per_period=WASH_DOMAIN.max_change_per_period,
        **TIMING,
    )
    assert np.all(np.isfinite(q_det))
    assert np.all((q_det >= 0.0) & (q_det <= 0.11))
    # Uncapped open-case firms share the upper threshold q* ~ 0.104924; capped open-case firms need <= q*
    assert np.all(q_det[:8] <= 0.10493)

    eps = 1e-4
    q_above = np.minimum(1.0, q_det + eps)
    q_below = np.maximum(0.0, q_det - eps)
    new_case_fine = (REGIME.fixed_fine / prices) * pi

    choices_above = npv_choices(
        benefit=benefit,
        cost_of_capital=np.full(n, 0.05),
        marginal_cost=cost,
        regime=REGIME,
        domain=WASH_DOMAIN,
        period_days=60,
        decision_day=1260,
        new_case_fine=new_case_fine,
        audit_probability=q_above,
    )
    choices_below = npv_choices(
        benefit=benefit,
        cost_of_capital=np.full(n, 0.05),
        marginal_cost=cost,
        regime=REGIME,
        domain=WASH_DOMAIN,
        period_days=60,
        decision_day=1260,
        new_case_fine=new_case_fine,
        audit_probability=q_below,
    )
    assert np.all(choices_above.actions == IMPROVE)
    positive = q_det > eps
    assert positive.sum() >= 10
    assert np.all(choices_below.actions[positive] == INFLATE)


def test_stackelberg_budget_conservation_and_knapsack_optimality() -> None:
    """Budget sum(q) == N * audit_probability holds strictly; stackelberg_eq is exact on 0-1 knapsack."""
    rng = np.random.default_rng(19)
    n = 14
    prices = rng.lognormal(0.0, 0.9, n)
    prices /= prices.mean()
    benefit = np.full(n, 4.4e-4)
    cost = 0.013 * rng.lognormal(0.0, 0.3, n)
    pi = np.where(np.arange(n) < 4, 0.0, 1.0)

    for mean_q in (0.002, 0.01, 0.05, 0.25):
        for alloc in STACKELBERG_ALLOCATIONS:
            reg = replace(REGIME, audit_probability=mean_q, audit_allocation=alloc)
            q = stackelberg_audit_probabilities(
                regime=reg,
                prices=prices,
                benefit=benefit,
                marginal_cost=cost,
                new_case_prob=pi,
                wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
                **TIMING,
            )
            assert q.shape == (n,)
            assert np.all((q >= 0.0) & (q <= 1.0))
            assert q.sum() == pytest.approx(n * mean_q, abs=1e-11)

    # Brute-force check of equal-weight knapsack optimality at mean_q = 0.015 (budget = 0.21)
    reg_eq = replace(REGIME, audit_probability=0.015, audit_allocation="stackelberg_eq")
    q_det = firm_deterrence_thresholds(
        prices=prices,
        benefit=benefit,
        marginal_cost=cost,
        new_case_prob=pi,
        regime=reg_eq,
        wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
        **TIMING,
    )
    q_eq = stackelberg_audit_probabilities(
        regime=reg_eq,
        prices=prices,
        benefit=benefit,
        marginal_cost=cost,
        new_case_prob=pi,
        wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
        deterrence_margin=0.0,
        **TIMING,
    )
    budget = n * 0.015
    best_bruteforce = 0
    for mask in range(1 << n):
        sel = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
        if q_det[sel].sum() <= budget + 1e-12:
            best_bruteforce = max(best_bruteforce, int(sel.sum()))
    deterred_eq = int(np.sum(q_eq + 1e-12 >= q_det))
    assert deterred_eq == best_bruteforce


def test_stackelberg_market_integration_preswitch_parity_and_budget() -> None:
    """Pre-switch histories stay bit-identical to B; post-switch budget is strictly conserved."""
    cfg = MarketConfig(
        n_investors=60,
        n_firms=24,
        trading_days=480,
        burn_in_days=180,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=61001,
        intrinsic_value_log_sd=0.915,
        financing_exposure_fraction=0.5,
        financing_spread=0.05,
        financing_repricing_interval=60,
        financing_horizon_days=1260,
        firm_discount_rate=0.05,
    )
    base = DisclosureMarket(cfg, WASH_DOMAIN, PolicySchedule.constant(WASH_DOMAIN)).run()
    for alloc in STACKELBERG_ALLOCATIONS:
        sched = PolicySchedule.constant(WASH_DOMAIN).with_intervention(181, audit_allocation=alloc)
        res = DisclosureMarket(cfg, WASH_DOMAIN, sched).run()
        # Bit-for-bit identical through switch day 180
        np.testing.assert_array_equal(res.true_goodness[:181], base.true_goodness[:181])
        np.testing.assert_array_equal(res.inflation[:181], base.inflation[:181])
        np.testing.assert_array_equal(res.market.prices[:181], base.market.prices[:181])
        # Exact budget conservation on every audit date
        assert res.audit_probability is not None
        budget = cfg.n_firms * WASH_DOMAIN.baseline_audit_probability
        for period in range(1, res.audit_probability.shape[0]):
            assert res.audit_probability[period].sum() == pytest.approx(budget, abs=1e-11)
            assert np.all((res.audit_probability[period] >= 0.0) & (res.audit_probability[period] <= 1.0))

    # Hidden Stackelberg runs cleanly and differs from public commitment
    sched_hid = PolicySchedule.constant(WASH_DOMAIN).with_intervention(
        181, audit_allocation="stackelberg_eq", audit_rule_public=False
    )
    res_hid = DisclosureMarket(cfg, WASH_DOMAIN, sched_hid).run()
    np.testing.assert_array_equal(res_hid.true_goodness[:181], base.true_goodness[:181])


def test_bayesian_observable_stackelberg_budget_and_invariance() -> None:
    """Observable-only Bayesian Stackelberg conserves budget and reverses value-targeting on giant firms."""
    rng = np.random.default_rng(101)
    n = 100
    prices = np.sort(rng.lognormal(0.0, 0.915, n))
    prices /= prices.mean()
    size_class = np.where(np.arange(n) >= 90, 0, np.where(np.arange(n) >= 60, 1, 2))
    mult = np.asarray(WASH_DOMAIN.size_cost_multipliers)[size_class]
    # Post-burn-in cost scale (g ~ -0.9 SD -> exp(-1.35) * 0.05 ~ 0.013)
    base_cost = 0.013 * mult

    q = bayesian_stackelberg_audit_probabilities(
        regime=REGIME,
        prices=prices,
        benefit_median=4.4e-4,
        base_cost=base_cost,
        cost_log_sd_eff=1.3,
        wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
        **TIMING,
    )
    assert q.shape == (n,)
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert q.sum() == pytest.approx(n * REGIME.audit_probability, abs=1e-11)
    # Anti-V under tight post-burn-in budget: small firms get far higher audit rate than giant firms
    assert q[size_class == 2].mean() > q[size_class == 0].mean()

    # At higher baseline cost, small firms saturate first and mid firms lead (inverted-U vs giant firms)
    q_hi = bayesian_stackelberg_audit_probabilities(
        regime=REGIME,
        prices=prices,
        benefit_median=4.4e-4,
        base_cost=0.08 * mult,
        cost_log_sd_eff=1.3,
        wash_cost_ratio=WASH_DOMAIN.wash_cost_ratio,
        **TIMING,
    )
    assert q_hi.sum() == pytest.approx(n * REGIME.audit_probability, abs=1e-11)
    assert q_hi[size_class == 1].mean() > q_hi[size_class == 0].mean()
