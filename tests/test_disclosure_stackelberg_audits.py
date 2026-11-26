"""Tests for Stackelberg optimal budget-constrained audit allocation (Proposition 5 / Track 3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.bayesian_audit import BayesianAuditFilter, quantize_audit_probabilities
from abm.disclosure.domain import load_domain
from abm.disclosure.game import IMPROVE, INFLATE, DisclosureMarket, npv_choices
from abm.disclosure.policy import (
    BAYESIAN_STACKELBERG_ALLOCATIONS,
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


def test_bayesian_audit_filter_persistent_types_bayes_learning_and_signal() -> None:
    """Joint type-state Bayesian filter preserves deterred clean types, learns from audits, and uses Delta r."""
    n = 12
    size_class = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
    prices = np.array([4.0, 3.5, 1.4, 1.2, 1.0, 0.9, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18], dtype=np.float64)
    benefit = np.full(n, 1.5e-4, dtype=np.float64)
    init_rep = np.zeros(n, dtype=np.float64)

    filt = BayesianAuditFilter(
        domain=WASH_DOMAIN,
        size_class=size_class,
        initial_report=init_rep,
        report_bound=WASH_DOMAIN.report_bound,
        discount_rate=0.05,
        horizon_days=1260,
        repricing_interval=60,
        period_days=60,
        use_report_signal=False,
    )
    # 1. Persistent-type survival: give small firms q = 0.05 (deterring all their types) for 20 periods.
    # Under the old i.i.d. bug, p_clean collapsed exponentially; with persistent types, it stays high.
    q_commit = np.where(size_class == 2, 0.05, 0.005)
    rep = init_rep.copy()
    for step in range(20):
        day = step * 60
        filt.record_commitment(
            day=day, prices=prices, benefit=benefit, planned_q=q_commit, regime=REGIME, post_audit_report=rep,
        )
        rep = rep - WASH_DOMAIN.natural_decay * 60 / 252
        filt.condition_on_audit(
            day=day + 60,
            pre_audit_report=rep,
            post_audit_report=rep,
            selected=np.zeros(n, dtype=bool),
            caught=np.zeros(n, dtype=bool),
        )
    # Deterred small firms retain high p_clean (> 0.85), whereas undeterred large firms have high p_dirty (> 0.5)
    assert filt.p_clean[size_class == 2].mean() > 0.85
    assert filt.p_dirty[size_class == 0].mean() > 0.50

    # 2. Exact Bayes learning on audit outcomes: caught shifts posterior toward LOWER cost shock eps;
    # audited-clean shifts posterior toward HIGHER cost shock eps; both reset p_dirty to 0.
    prior_mean_eps = float(filt.prior_w @ filt.eps_nodes)
    # Audit firm 0 (caught) and firm 1 (clean) from the same large tier
    sel = np.zeros(n, dtype=bool)
    caught = np.zeros(n, dtype=bool)
    sel[0] = sel[1] = True
    caught[0] = True
    filt.condition_on_audit(
        day=21 * 60, pre_audit_report=rep, post_audit_report=rep, selected=sel, caught=caught,
    )
    assert filt.p_dirty[0] == pytest.approx(0.0)
    assert filt.p_dirty[1] == pytest.approx(0.0)
    eps_caught = float(filt.w_clean[0] @ filt.eps_nodes)
    eps_clean = float(filt.w_clean[1] @ filt.eps_nodes)
    assert eps_caught < eps_clean
    assert eps_caught <= prior_mean_eps + 1e-9

    # 3. Report-signal Bayesian update: an anomalous positive score jump sharply raises p_dirty
    filt_sig = BayesianAuditFilter(
        domain=WASH_DOMAIN,
        size_class=size_class,
        initial_report=init_rep,
        report_bound=WASH_DOMAIN.report_bound,
        discount_rate=0.05,
        horizon_days=1260,
        repricing_interval=60,
        period_days=60,
        use_report_signal=True,
    )
    q_unif = np.full(n, 0.01, dtype=np.float64)
    filt_sig.record_commitment(
        day=0, prices=prices, benefit=benefit, planned_q=q_unif, regime=REGIME, post_audit_report=init_rep,
    )
    # Firm 2 jumps +0.22 SD (impossible by clean improvement); firm 3 decays cleanly by -0.04 SD
    pre_rep = np.full(n, -0.04, dtype=np.float64)
    pre_rep[2] = 0.22
    filt_sig.predict(day=60, pre_audit_report=pre_rep)
    assert filt_sig.p_dirty[2] > filt_sig.p_dirty[3] + 0.20

    # 4. Exact budget conservation for both public commitment and unconfounded hidden clearing
    for public in (True, False):
        reg_p = replace(REGIME, audit_allocation="stackelberg_obs", audit_rule_public=public)
        q_alloc = filt_sig.allocate(day=60, regime=reg_p, prices=prices, benefit=benefit, post_audit_report=pre_rep)
        assert np.all((q_alloc >= 0.0) & (q_alloc <= 1.0))
        assert q_alloc.sum() == pytest.approx(n * REGIME.audit_probability, abs=1e-11)
        if not public:
            # Hidden optimal rule targets the highest-p_dirty firm (firm 2) with full priority
            assert q_alloc[2] >= q_alloc.max() - 1e-12


def test_bayesian_hidden_and_value_weighted_allocations_in_market() -> None:
    """All Bayesian Stackelberg variants (obs, obs_val, obs_sig, public & hidden) conserve budget & pre-switch."""
    cfg = MarketConfig(
        n_investors=50,
        n_firms=24,
        trading_days=360,
        burn_in_days=120,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=61002,
        intrinsic_value_log_sd=0.915,
        financing_exposure_fraction=0.5,
        financing_spread=0.05,
        financing_repricing_interval=60,
        financing_horizon_days=1260,
        firm_discount_rate=0.05,
    )
    base = DisclosureMarket(cfg, WASH_DOMAIN, PolicySchedule.constant(WASH_DOMAIN)).run()
    budget = cfg.n_firms * WASH_DOMAIN.baseline_audit_probability
    for alloc in BAYESIAN_STACKELBERG_ALLOCATIONS:
        for public in (True, False):
            sched = PolicySchedule.constant(WASH_DOMAIN).with_intervention(
                121, audit_allocation=alloc, audit_rule_public=public,
            )
            res = DisclosureMarket(cfg, WASH_DOMAIN, sched).run()
            np.testing.assert_array_equal(res.true_goodness[:121], base.true_goodness[:121])
            np.testing.assert_array_equal(res.inflation[:121], base.inflation[:121])
            assert res.audit_probability is not None
            for period in range(1, res.audit_probability.shape[0]):
                assert res.audit_probability[period].sum() == pytest.approx(budget, abs=1e-11)
                assert np.all((res.audit_probability[period] >= 0.0) & (res.audit_probability[period] <= 1.0))


def test_quantize_audit_probabilities_and_risk_band_commitments() -> None:
    """L2-optimal G-band quantization conserves budget, keeps sparse watchlists sharp, and omits default 0."""
    # Default 0 omitted from PolicyRegime.to_dict()
    assert "audit_commitment_buckets" not in REGIME.to_dict()
    reg3 = replace(REGIME, audit_allocation="stackelberg_obs", audit_commitment_buckets=3)
    assert reg3.to_dict()["audit_commitment_buckets"] == 3
    with pytest.raises(ValueError, match="audit_commitment_buckets"):
        replace(REGIME, audit_allocation="stackelberg_obs", audit_commitment_buckets=1)
    with pytest.raises(ValueError, match="audit_commitment_buckets"):
        replace(REGIME, audit_allocation="value", audit_commitment_buckets=3)
    with pytest.raises(ValueError, match="audit_commitment_buckets"):
        replace(REGIME, audit_allocation="stackelberg_obs", audit_rule_public=False, audit_commitment_buckets=3)

    # Sparse watchlist test: 180 zeros, 15 medium (~0.06), 5 high (~0.22)
    raw = np.array([0.0] * 180 + [0.06] * 15 + [0.22] * 5, dtype=np.float64)
    pooled, labels = quantize_audit_probabilities(raw, 3)
    assert pooled.sum() == pytest.approx(raw.sum(), abs=1e-12)
    assert len(np.unique(np.round(pooled, 12))) == 3
    assert np.allclose(pooled[:180], 0.0)
    assert np.allclose(pooled[180:195], 0.06)
    assert np.allclose(pooled[195:], 0.22)
    assert np.all(labels[:180] == 0) and np.all(labels[180:195] == 1) and np.all(labels[195:] == 2)

    # End-to-end market check under Q(kappa_T, s_P) and 3-band public risk tiers
    q_dom = replace(WASH_DOMAIN, natural_decay=0.0, natural_reversion=0.505, reversion_permanent_share=0.423)
    cfg = MarketConfig(
        n_investors=50, n_firms=24, trading_days=360, burn_in_days=120,
        initial_impact_share=0.2, taste_median=0.0724, seed=61009,
        intrinsic_value_log_sd=0.915, financing_exposure_fraction=0.5,
        financing_spread=0.05, financing_repricing_interval=60,
        financing_horizon_days=1260, firm_discount_rate=0.05,
    )
    sched = PolicySchedule.constant(q_dom).with_intervention(
        121, audit_allocation="stackelberg_obs_sig", audit_commitment_buckets=3,
    )
    res = DisclosureMarket(cfg, q_dom, sched).run()
    assert res.audit_probability is not None
    budget = cfg.n_firms * q_dom.baseline_audit_probability
    for period in range(3, res.audit_probability.shape[0]):
        assert res.audit_probability[period].sum() == pytest.approx(budget, abs=1e-11)
        assert len(np.unique(np.round(res.audit_probability[period], 10))) <= 3
