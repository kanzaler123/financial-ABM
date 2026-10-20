"""Lump-sum fines per detected case, and the closed-form washing conditions they imply."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import DomainSpec, load_domain
from abm.disclosure.financing import financing_signal_factors
from abm.disclosure.game import (
    IMPROVE,
    INFLATE,
    DisclosureMarket,
    new_case_probability,
    npv_choices,
)
from abm.disclosure.learning import discrete_candidates
from abm.disclosure.metrics import compute_metrics
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
SMALL = MarketConfig(n_investors=60, n_firms=6, trading_days=130, burn_in_days=10, initial_impact_share=.5)
REGIME = PolicyRegime(audit_probability=.01, penalty=0., transparency=.1, subsidy=0.)
# Small enough that neither action reaches the per-period cap.
BENEFIT, COST = np.array([5e-4]), np.array([.05])
COMMON = dict(cost_of_capital=np.full(1, .05), domain=ESG, period_days=60)


class NoAudit:
    def random(self, size):
        return np.ones(size)


def factors(regime=REGIME):
    return financing_signal_factors(
        np.full(1, .05), decision_day=0, horizon_days=1260, repricing_interval=60, period_days=60,
        implementation_delay=ESG.implementation_delay_days, disclosure_delay=ESG.disclosure_delay_days,
        audit_probability=regime.audit_probability, transparency=regime.transparency,
    )


def interior_values(benefit=BENEFIT, cost=COST, transparency=.1):
    real, _, wash, detection = factors()
    w = ESG.wash_cost_ratio
    wash_value = (benefit * (1 - transparency) * wash)**2 / (4 * w * cost)
    improve_value = (benefit * real)**2 / (4 * cost)
    return wash_value, improve_value, detection


def test_fixed_fine_is_validated_and_serialized_only_when_used():
    for bad in (-1e-6, np.nan, np.inf):
        with pytest.raises(ValueError, match="fixed_fine"):
            replace(REGIME, fixed_fine=bad)
    with pytest.raises(ValueError, match="baseline_fixed_fine"):
        replace(ESG, baseline_fixed_fine=-1.)
    schedule = PolicySchedule.constant(ESG).with_intervention(100, transparency=.6)
    legacy = asdict(schedule)
    for regime in ("baseline", "intervention"):
        for later_option in ("fixed_fine", "fixed_fine_basis", "audit_allocation", "audit_rule_public",
                             "audit_jump_threshold"):
            legacy[regime].pop(later_option)
    assert schedule.to_dict() == legacy
    assert "baseline_fixed_fine" not in ESG.to_dict()
    fined = replace(ESG, baseline_fixed_fine=4.5e-5)
    assert DomainSpec.from_dict(fined.to_dict()) == fined
    assert PolicyRegime.baseline(fined).fixed_fine == 4.5e-5
    serialized = PolicySchedule.constant(fined).to_dict()["baseline"]
    assert serialized["fixed_fine"] == 4.5e-5
    assert PolicyRegime(**serialized) == PolicyRegime.baseline(fined)


def test_new_report_opens_a_case_only_without_undetected_inflation():
    pending = [(80, INFLATE, np.array([1]), np.array([.1])), (130, INFLATE, np.array([2]), np.array([.1])),
               (140, INFLATE, np.array([3]), np.array([.1])), (150, INFLATE, np.array([4]), np.array([.1])),
               (90, IMPROVE, np.array([5]), np.array([.1]))]
    inflation = np.array([.2, 0, 0, 0, 0, 0, 0])
    # Disclosure on day 140: the day-120 audit can close an earlier case first.
    probability = new_case_probability(inflation, pending, decision_day=60, wash_due=140, period_days=60,
                                       audit_probability=.3)
    np.testing.assert_allclose(probability, [.3, .3, 0, 0, 1, 1, 1])
    # Disclosure before the next audit: any undetected inflation is the same case.
    probability = new_case_probability(inflation, pending, decision_day=60, wash_due=80, period_days=60,
                                       audit_probability=.3)
    np.testing.assert_allclose(probability, [0, 0, 1, 1, 1, 1, 1])


def test_fixed_fine_decides_whether_to_wash_but_not_how_much():
    real, _, wash, detection = factors()
    free = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, **COMMON)
    assert free.actions[0] == INFLATE
    np.testing.assert_allclose(free.amounts, BENEFIT * .9 * wash / (2 * ESG.wash_cost_ratio * COST))
    wash_value, improve_value, detection = interior_values()
    # Washing survives exactly while the expected fine is below its advantage over improving.
    critical = (wash_value - improve_value) / detection
    for fine, action in ((.99 * critical, INFLATE), (1.01 * critical, IMPROVE)):
        choice = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, new_case_fine=fine, **COMMON)
        assert choice.actions[0] == action
        if action == INFLATE:
            assert choice.amounts[0] == free.amounts[0]
            assert choice.npv[0] == pytest.approx(wash_value[0] - fine[0] * detection[0], rel=1e-12)


def test_brute_force_value_with_a_lump_sum_fine_and_a_binding_cap():
    benefit, fine = np.array([2e-3]), np.array([1e-5])
    regime = replace(REGIME, penalty=1e-4)
    _, _, wash, detection = factors(regime)
    grid = np.linspace(0, ESG.max_change_per_period, 200001)[1:]
    values = (benefit * .9 * wash - regime.penalty * detection) * grid - ESG.wash_cost_ratio * COST * grid**2
    assert values.argmax() == grid.size - 1
    choice = npv_choices(benefit=benefit, marginal_cost=COST, regime=regime, new_case_fine=fine, **COMMON)
    assert choice.actions[0] == INFLATE
    assert choice.npv[0] == pytest.approx(values.max() - fine[0] * detection[0], abs=1e-12)


def test_fixed_fine_falls_with_size_so_large_firms_wash_first():
    wash_value, improve_value, detection = interior_values()
    critical = float((wash_value - improve_value)[0] / detection[0])
    prices = np.array([.3, 1., 3., 10.])
    common = dict(benefit=np.full(4, BENEFIT[0]), marginal_cost=np.full(4, COST[0]), regime=REGIME,
                  cost_of_capital=np.full(4, .05), domain=ESG, period_days=60)
    # A money fine whose market-cap weight crosses the critical value at a cap of 1.6.
    choices = npv_choices(new_case_fine=1.6 * critical / prices, **common)
    np.testing.assert_array_equal(choices.actions, [IMPROVE, IMPROVE, INFLATE, INFLATE])
    # The same fine as a share of market cap leaves no size gradient.
    same_share = npv_choices(new_case_fine=np.full(4, 1.6 * critical), **common)
    np.testing.assert_array_equal(same_share.actions, IMPROVE)


@pytest.mark.parametrize("fine", [0., 4e-5, 4e-4])
def test_firm_transparency_threshold_matches_the_closed_form(fine):
    benefit = np.array([2e-3])
    real, _, wash, detection = factors()
    w = ESG.wash_cost_ratio
    threshold = 1 - np.sqrt(w) * np.sqrt(real**2 + 4 * COST * fine * detection / benefit**2) / wash
    common_threshold = 1 - np.sqrt(w) * real / wash
    assert threshold[0] <= common_threshold[0]
    if fine == 0:
        assert threshold[0] == pytest.approx(common_threshold[0])
    for transparency, action in ((threshold[0] - 1e-3, INFLATE), (threshold[0] + 1e-3, IMPROVE)):
        choice = npv_choices(benefit=benefit, marginal_cost=COST, regime=replace(REGIME, transparency=transparency),
                             new_case_fine=np.array([fine]), **COMMON)
        assert choice.actions[0] == action
        assert choice.amounts[0] < ESG.max_change_per_period


def test_discrete_candidates_charge_the_lump_sum_on_positive_wash_amplitudes():
    common = dict(improve_capacity=np.array([.25, .25]), inflate_capacity=np.array([.25, 0.]),
                  marginal_cost=np.array([.05, .05]), max_change_per_period=.25, wash_cost_ratio=.2, subsidy=0.,
                  real_value_per_sd=np.full(2, .01), displaced_real_value_per_sd=np.full(2, .001),
                  wash_value_per_sd=np.full(2, .02), fine_value_per_sd=np.zeros(2), amplitude_fractions=(.1, 1.))
    free = discrete_candidates(**common)
    fined = discrete_candidates(**common, fixed_fine_value=np.array([3e-4, 3e-4]))
    wash = free.kinds == INFLATE
    np.testing.assert_array_equal(fined.values[:, ~wash], free.values[:, ~wash])
    np.testing.assert_allclose(fined.values[0, wash], free.values[0, wash] - 3e-4)
    np.testing.assert_array_equal(fined.values[1, wash], free.values[1, wash])
    with pytest.raises(ValueError, match="nonnegative"):
        discrete_candidates(**common, fixed_fine_value=-1.)


def market_with_benefit(policy="npv", benefit=2e-3, **domain_changes):
    config = replace(SMALL, firm_policy=policy, firm_action_fractions=(1.,) if policy == "npv_discrete" else (.01,))
    market = DisclosureMarket(config, replace(ESG, natural_decay=0., **domain_changes))
    market.marginal_financing_saving = lambda: np.full(SMALL.n_firms, benefit)
    market.attributes.cost_scale[:] = .05
    market.attributes.goodness[:] = 0.
    return market


@pytest.mark.parametrize("policy", ["npv", "npv_discrete"])
def test_detected_case_pays_one_fixed_fine_whatever_its_size(policy):
    fine = 3e-3
    market = market_with_benefit(policy, baseline_audit_probability=1., baseline_penalty=0., baseline_fixed_fine=fine)
    market.attributes.inflation[:] = [0., .01, .5, 0., .2, 0.]
    market._audit_and_decide(60, 1)
    caught = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    np.testing.assert_array_equal(market.detection_history[1], caught)
    np.testing.assert_allclose(market.penalty_history[1] * market.prices, np.where(caught, fine, 0.), rtol=1e-12)
    np.testing.assert_array_equal(market.attributes.inflation, 0.)
    resource = market.cost_history[1] * market.prices
    np.testing.assert_allclose(market.pending_value_change, -resource - np.where(caught, fine, 0.), rtol=1e-12)


@pytest.mark.parametrize("policy", ["npv", "npv_discrete"])
def test_inflation_joining_an_open_case_carries_no_fixed_fine(policy):
    open_case = np.array([.3, .3, .3, 0., 0., 0.])
    # A fixed fine large enough to deter opening a case; no audit this period.
    market = market_with_benefit(policy, baseline_penalty=0., baseline_fixed_fine=1.)
    without = market_with_benefit(policy, baseline_penalty=0.)
    for item in (market, without):
        item.audit_rng = NoAudit()
        item.attributes.inflation[:] = open_case
        item._audit_and_decide(60, 1)
    np.testing.assert_array_equal(without.action_history[1], INFLATE)
    np.testing.assert_array_equal(market.action_history[1, :3], INFLATE)
    assert not np.any(market.action_history[1, 3:] == INFLATE)
    np.testing.assert_array_equal(market.amount_history[1, :3], without.amount_history[1, :3])


class AuditSequence:
    """No audit on day 60, every firm checked on day 120."""

    def __init__(self):
        self.draws = [np.ones, np.zeros]

    def random(self, size):
        return self.draws.pop(0)(size)


def test_run_records_one_fine_per_detection():
    fine = 1e-3
    market = market_with_benefit(baseline_penalty=0., baseline_fixed_fine=fine)
    market.audit_rng = AuditSequence()
    result = market.run()
    washed = result.actions[1] == INFLATE
    assert washed.any() and not washed.all()
    np.testing.assert_array_equal(result.detections[2], washed)
    decision_prices = result.market.prices[::SMALL.firm_decision_interval]
    np.testing.assert_allclose(result.penalty_fraction * decision_prices,
                               np.where(result.detections.astype(bool), fine, 0.), rtol=1e-12, atol=0)
    metrics = compute_metrics(result, market.config, market.domain)
    assert metrics["actions"]["detections"] == int(washed.sum())
    assert metrics["actions"]["wash"] == int((result.actions[1:] == INFLATE).sum())
    assert metrics["costs"]["fine_transfer"] == pytest.approx(fine * washed.sum(), rel=1e-12)


def test_zero_fixed_fine_keeps_the_proportional_regime_unchanged():
    assert replace(ESG, baseline_fixed_fine=0.) == ESG
    assert DisclosureMarket(SMALL, replace(ESG, baseline_fixed_fine=0.)).run_id == DisclosureMarket(SMALL, ESG).run_id
    assert DisclosureMarket(SMALL, replace(ESG, baseline_fixed_fine=1e-4)).run_id != DisclosureMarket(SMALL, ESG).run_id
