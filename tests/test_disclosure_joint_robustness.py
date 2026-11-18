"""The contract 24 joint robustness test: reverting true values, one fixed fine per inflated report and firm-level
wash cost ratios switched on together. The three options already exist; these tests check their combination."""

from dataclasses import replace

import numpy as np

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_signal_factors
from abm.disclosure.game import INFLATE, IMPROVE, DisclosureMarket, npv_choices, real_persistence
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
JOINT = replace(ESG, natural_decay=0., natural_reversion=.32, wash_cost_log_sd=np.log(2))
SMALL = MarketConfig(n_investors=60, n_firms=6, trading_days=130, burn_in_days=10, initial_impact_share=.5)


def factors(persistence, transparency):
    return financing_signal_factors(
        np.full(1, .05), decision_day=0, horizon_days=1260, repricing_interval=60, period_days=60,
        implementation_delay=ESG.implementation_delay_days, disclosure_delay=ESG.disclosure_delay_days,
        audit_probability=.01, transparency=transparency, real_persistence=persistence)


def test_firm_thresholds_follow_the_closed_form_with_all_three_options():
    persistence = real_persistence(JOINT, 200, 1260)
    real, _, wash, detection = factors(persistence, .1)
    ratios = np.array([.05, .1, .2, .4])
    benefit, cost, fine = np.full(4, 2e-4), np.full(4, .05), np.full(4, 2e-6)
    # Proposition 2 with A_r(kappa), w_i and pi = 1 (every new report carries its own fine).
    thresholds = 1 - np.sqrt(ratios) * np.sqrt(real[0]**2 + 4 * cost * fine * detection[0] / benefit**2) / wash[0]
    common = dict(benefit=benefit, marginal_cost=cost, cost_of_capital=np.full(4, .05), domain=JOINT, period_days=60,
                  new_case_fine=fine, wash_cost_ratio=ratios, real_persistence=persistence)
    for index, threshold in enumerate(thresholds):
        for transparency, action in ((threshold - 1e-3, INFLATE), (threshold + 1e-3, IMPROVE)):
            regime = PolicyRegime(audit_probability=.01, penalty=0., transparency=float(transparency), subsidy=0.)
            choice = npv_choices(regime=regime, **common)
            assert choice.actions[index] == action
            assert choice.amounts[index] < JOINT.max_change_per_period


def test_information_washers_are_a_subset_of_fine_washers():
    """Contract 24.6 T1: for the same state, washing under I (tau 0.6, fine f) implies washing under F (0.1, 4f).

    The derivation covers interior choices only. At the per-period cap the wash value is linear in its benefit, so
    F's extra benefit can fall short of its three extra fines; those are the only exceptions."""
    rng = np.random.default_rng(24)
    n = 20000
    persistence = real_persistence(JOINT, 200, 1260)
    common = dict(benefit=10 ** rng.uniform(-4.5, -2.5, n), marginal_cost=10 ** rng.uniform(-2, 0, n),
                  cost_of_capital=np.full(n, .05), domain=JOINT, period_days=60,
                  wash_cost_ratio=.2 * rng.lognormal(-np.log(2)**2 / 2, np.log(2), n), real_persistence=persistence)
    fine = 10 ** rng.uniform(-7, -3, n)
    base = PolicyRegime(audit_probability=.01, penalty=0., transparency=.1, subsidy=0.)
    washes = {}
    for name, transparency, multiple in (("B", .1, 1), ("I", .6, 1), ("F", .1, 4)):
        choice = npv_choices(regime=replace(base, transparency=transparency), new_case_fine=multiple * fine, **common)
        washes[name] = choice.actions == INFLATE
        if name == "I":
            capped = choice.amounts >= JOINT.max_change_per_period
    assert washes["I"].sum() > 100 and washes["F"].sum() > washes["I"].sum()
    exceptions = washes["I"] & ~washes["F"]
    assert not np.any(exceptions & ~capped)
    assert exceptions.sum() < .01 * washes["I"].sum()
    assert not np.any(washes["F"] & ~washes["B"])


def test_market_runs_with_all_three_options_and_charges_whole_fines():
    config = replace(SMALL, trading_days=400, burn_in_days=10)
    domain = replace(JOINT, baseline_fixed_fine=1e-6, baseline_penalty=0.)
    regime = replace(PolicyRegime.baseline(domain), fixed_fine_basis="report", audit_probability=.2)
    market = DisclosureMarket(config, domain, PolicySchedule(regime))
    market.marginal_financing_saving = lambda: np.full(config.n_firms, 2e-3)
    market.attributes.cost_scale[:] = .05
    result = market.run()
    assert result.wash_cost_ratio.shape == (config.n_firms,)
    assert np.any(result.actions == INFLATE) and result.detections.any()
    prices = result.market.prices[::config.firm_decision_interval][:result.penalty_fraction.shape[0]]
    counts = result.penalty_fraction * prices / regime.fixed_fine
    caught = result.detections.astype(bool)
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-6)
    assert np.all(np.round(counts[caught]) >= 1) and np.all(counts[~caught] == 0)
    # Reversion keeps the cross section from collapsing.
    assert result.true_goodness[-1].std() > .5 * result.true_goodness[0].std()
