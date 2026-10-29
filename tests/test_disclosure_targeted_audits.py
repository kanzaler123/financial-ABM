"""Targeted audits for contract section 21 (stage E.1): allocation rules, firm-level audit
probabilities in the firm's decision, and defaults that keep earlier runs unchanged."""

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.config import TRADING_DAYS_PER_YEAR
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_discount_factors, financing_signal_factors
from abm.disclosure.game import INFLATE, MAINTAIN, DisclosureMarket, npv_choices
from abm.disclosure.policy import PolicyRegime, PolicySchedule, audit_probabilities

ESG = load_domain("configs/domains/esg.json")
# Without the proportional fine firms inflate, so flags and detections occur in small runs.
WASHING = replace(ESG, baseline_penalty=0.0, baseline_audit_probability=0.1)
SMALL = MarketConfig(n_investors=60, n_firms=12, trading_days=400, burn_in_days=120, initial_impact_share=.5,
                     taste_median=.2)
REGIME = PolicyRegime(audit_probability=.01, penalty=0., transparency=.1, subsidy=0.)
TIMING = dict(decision_day=60, horizon_days=1260, repricing_interval=60, period_days=60,
              implementation_delay=120, disclosure_delay=20)


# -- policy fields and allocation ---------------------------------------------

def test_defaults_are_omitted_from_identities_and_validated():
    payload = REGIME.to_dict()
    assert not {"audit_allocation", "audit_rule_public", "audit_jump_threshold"} & set(payload)
    value = replace(REGIME, audit_allocation="value")
    assert value.to_dict()["audit_allocation"] == "value" and "audit_jump_threshold" not in value.to_dict()
    hidden_jump = replace(REGIME, audit_allocation="jump", audit_rule_public=False)
    assert hidden_jump.to_dict()["audit_rule_public"] is False
    assert hidden_jump.to_dict()["audit_jump_threshold"] == .125
    for regime in (value, hidden_jump):
        assert PolicyRegime(**regime.to_dict()) == regime
    with pytest.raises(ValueError, match="audit_allocation"):
        replace(REGIME, audit_allocation="size")
    with pytest.raises(ValueError, match="audit_rule_public"):
        replace(REGIME, audit_rule_public=1)
    for bad in (0., -.1, np.nan):
        with pytest.raises(ValueError, match="audit_jump_threshold"):
            replace(REGIME, audit_jump_threshold=bad)
    plain = PolicySchedule.constant(ESG)
    explicit = PolicySchedule(replace(plain.baseline, audit_allocation="uniform", audit_rule_public=True))
    assert DisclosureMarket(SMALL, ESG, explicit).run_id == DisclosureMarket(SMALL, ESG, plain).run_id


def test_value_allocation_is_proportional_and_keeps_the_budget():
    values = np.random.default_rng(3).lognormal(0., .9, 200)
    probability = audit_probabilities(replace(REGIME, audit_allocation="value"), values)
    assert probability.sum() == pytest.approx(2., abs=1e-12)
    np.testing.assert_allclose(probability / values, probability[0] / values[0], rtol=1e-12)
    # A dominant firm is capped at one and its excess is shared by value among the others.
    values = np.array([100., 1., 2., 3., 4.])
    probability = audit_probabilities(replace(REGIME, audit_allocation="value", audit_probability=.4), values)
    assert probability[0] == 1. and probability.sum() == pytest.approx(2., abs=1e-12)
    np.testing.assert_allclose(probability[1:] / values[1:], 1. / values[1:].sum(), rtol=1e-12)


def test_jump_allocation_halves_the_random_share_and_keeps_the_budget():
    regime = replace(REGIME, audit_allocation="jump")
    values = np.ones(200)
    flagged = np.zeros(200, dtype=bool)
    flagged[:20] = True
    probability = audit_probabilities(regime, values, flagged)
    np.testing.assert_allclose(probability[:20], .005 + 1. / 20, rtol=1e-12)
    np.testing.assert_allclose(probability[20:], .005, rtol=1e-12)
    assert probability.sum() == pytest.approx(2., abs=1e-12)
    np.testing.assert_array_equal(audit_probabilities(regime, values, np.zeros(200, dtype=bool)), .01)
    np.testing.assert_array_equal(audit_probabilities(regime, values, None), .01)
    # A single flagged firm is capped at one; the excess goes equally to the others.
    one = np.zeros(200, dtype=bool)
    one[7] = True
    probability = audit_probabilities(regime, values, one)
    assert probability[7] == 1. and probability.sum() == pytest.approx(2., abs=1e-12)
    np.testing.assert_allclose(np.delete(probability, 7), .005 + .005 / 199, rtol=1e-12)
    np.testing.assert_array_equal(audit_probabilities(REGIME, values, flagged), .01)


# -- financing factors with firm-level probabilities --------------------------

def test_firm_rates_reproduce_the_common_rate():
    rates = np.full(3, .05)
    for q in (0., .004, .01, .3):
        scalar = financing_discount_factors(rates, audit_probability=q, **TIMING)
        for firm in (financing_discount_factors(rates, audit_probability=np.full(3, q), **TIMING),
                     financing_discount_factors(rates, audit_probability=q, first_audit_probability=q, **TIMING)):
            for left, right in zip(scalar, firm):
                np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-15)
    common = financing_signal_factors(rates, audit_probability=.02, transparency=.1, **TIMING)
    firm = financing_signal_factors(rates, audit_probability=np.full(3, .02), transparency=.1, **TIMING)
    for left, right in zip(common, firm):
        np.testing.assert_allclose(left, right, rtol=1e-12)


def test_two_rate_factors_match_a_direct_sum():
    rate, first, later = .05, .2, .005
    _, wash, fine = financing_discount_factors(np.array([rate]), audit_probability=later,
                                               first_audit_probability=first, **TIMING)
    day, period = TIMING["decision_day"], TIMING["period_days"]
    wash_due = day + TIMING["disclosure_delay"]
    first_audit = ((wash_due + period - 1) // period) * period
    expected_wash = 0.
    for t in range(1, TIMING["horizon_days"] + 1):
        quoted = (day + t) // TIMING["repricing_interval"] * TIMING["repricing_interval"] - 1
        if quoted < wash_due:
            continue
        # Audits strictly before the quote day remove the report from the quote.
        audits = len(range(first_audit, quoted, period))
        survival = 1. if audits == 0 else (1 - first) * (1 - later) ** (audits - 1)
        expected_wash += survival * np.exp(-rate * t / TRADING_DAYS_PER_YEAR) / TRADING_DAYS_PER_YEAR
    expected_fine = sum((first if k == 0 else (1 - first) * (1 - later) ** (k - 1) * later)
                        * np.exp(-rate * (first_audit + k * period - day) / TRADING_DAYS_PER_YEAR)
                        for k in range(5000))
    assert wash[0] == pytest.approx(expected_wash, rel=1e-12)
    assert fine[0] == pytest.approx(expected_fine, rel=1e-9)


def test_firm_rates_are_rejected_outside_the_main_timing():
    rates = np.full(2, .05)
    # Reverting true values are allowed since contract section 25.9 (d).
    for changes in (dict(publish_audit_results=True), dict(reporting_interval_days=5)):
        with pytest.raises(ValueError, match="firm-level audit probabilities"):
            financing_signal_factors(rates, audit_probability=np.full(2, .01), transparency=.1, **TIMING, **changes)
        with pytest.raises(ValueError, match="firm-level audit probabilities"):
            financing_signal_factors(rates, audit_probability=.01, first_audit_probability=.2, transparency=.1,
                                     **TIMING, **changes)


# -- the firm's decision ------------------------------------------------------

DECISION = dict(cost_of_capital=np.full(4, .05), domain=ESG, period_days=60, decision_day=60)
BENEFIT = np.array([2e-4, 5e-4, 1e-3, 3e-3])
COST = np.array([.05, .08, .03, .12])


def test_planning_with_the_regime_rate_changes_nothing():
    default = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, **DECISION)
    planned = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, audit_probability=np.full(4, .01),
                          **DECISION)
    np.testing.assert_array_equal(default.actions, planned.actions)
    for left, right in zip((default.amounts, default.npv), (planned.amounts, planned.npv)):
        np.testing.assert_allclose(left, right, rtol=1e-10)


def test_each_firm_plans_with_its_own_rate():
    rates = np.array([.002, .01, .05, .2])
    together = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, audit_probability=rates, **DECISION)
    for firm, q in enumerate(rates):
        alone = npv_choices(benefit=BENEFIT[[firm]], marginal_cost=COST[[firm]], regime=replace(REGIME, audit_probability=q),
                            **{**DECISION, "cost_of_capital": np.full(1, .05)})
        assert together.actions[firm] == alone.actions[0]
        assert together.amounts[firm] == pytest.approx(alone.amounts[0], rel=1e-10)
        assert together.npv[firm] == pytest.approx(alone.npv[0], rel=1e-10)


@pytest.mark.parametrize("fine", [None, 2e-4])
@pytest.mark.parametrize("room", [-.05, 0., .03, .08, .15, .3])
@pytest.mark.parametrize("first", [.055, .2, 1.])
def test_jump_rule_inflation_is_the_best_amount_on_either_side_of_the_room(room, first, fine):
    later = .005
    benefit, cost = np.array([3e-3, 1e-3, 6e-4]), np.array([.05, .04, .2])
    size = benefit.size
    common = dict(cost_of_capital=np.full(size, .05), domain=ESG, period_days=60, decision_day=60,
                  benefit=benefit, marginal_cost=cost, regime=REGIME)
    new_case_fine = None if fine is None else np.full(size, fine)
    choice = npv_choices(**common, audit_probability=later, first_audit_probability=first,
                         flag_room=np.full(size, room), new_case_fine=new_case_fine)
    real, _, wash_u, detect_u = financing_signal_factors(np.full(size, .05), audit_probability=later,
                                                         transparency=.1, **TIMING)
    _, _, wash_f, detect_f = financing_signal_factors(np.full(size, .05), audit_probability=later,
                                                      first_audit_probability=first, transparency=.1, **TIMING)
    cap, wash_cost = ESG.max_change_per_period, ESG.wash_cost_ratio * cost
    grid = np.union1d(np.linspace(0., cap, 200001), [min(max(room, 0.), cap)])[:, None]
    unflagged = grid <= room
    gain = np.where(unflagged, benefit * .9 * wash_u, benefit * .9 * wash_f)
    penalty = 0. if fine is None else np.where(grid > 0, fine * np.where(unflagged, detect_u, detect_f), 0.)
    value = gain * grid - wash_cost * grid**2 - penalty
    # The improve option is interior here, so its value is (B A_r)^2 / 4c.
    assert (benefit * real / (2 * cost) <= cap).all()
    expected = np.maximum(value.max(axis=0), (benefit * real) ** 2 / (4 * cost))
    chosen = np.where(choice.actions == MAINTAIN, 0., choice.npv)
    np.testing.assert_allclose(chosen, expected, rtol=1e-6, atol=1e-12)
    inflating = choice.actions == INFLATE
    np.testing.assert_allclose(choice.amounts[inflating], grid[value[:, inflating].argmax(axis=0), 0], atol=2e-6)


# -- the market -----------------------------------------------------------------

def run(changes: dict, domain=WASHING, config=SMALL):
    schedule = PolicySchedule.constant(domain).with_intervention(121, **changes)
    return DisclosureMarket(config, domain, schedule).run()


def test_defaults_keep_results_and_archive_no_new_arrays():
    plain = DisclosureMarket(SMALL, WASHING, PolicySchedule.constant(WASHING).with_intervention(121, transparency=.3)).run()
    assert plain.audit_probability is None and plain.audit_flags is None
    targeted = run({"transparency": .3, "audit_allocation": "value"})
    # Before the switch the targeted run is the plain run, decision for decision.
    np.testing.assert_array_equal(targeted.actions[:3], plain.actions[:3])
    np.testing.assert_array_equal(targeted.audit_selected[:3], plain.audit_selected[:3])


def test_value_rule_audits_in_proportion_to_market_value():
    result = run({"audit_allocation": "value"})
    prices = result.market.prices
    for period in range(1, result.actions.shape[0]):
        q = result.audit_probability[period]
        assert q.sum() == pytest.approx(SMALL.n_firms * .1, abs=1e-12)
        if period * 60 > 120:
            np.testing.assert_allclose(q / prices[period * 60], q[0] / prices[period * 60, 0], rtol=1e-10)
        else:
            np.testing.assert_array_equal(q, .1)
    assert not result.audit_flags.any()


@pytest.mark.parametrize("public", [True, False])
def test_jump_flags_follow_the_report_and_keep_the_budget(public):
    result = run({"audit_allocation": "jump", "audit_rule_public": public})
    top = WASHING.report_bound
    report = np.minimum(result.true_goodness + result.inflation, top)
    for period in range(1, result.actions.shape[0]):
        day = period * 60
        q, flags = result.audit_probability[period], result.audit_flags[period]
        assert q.sum() == pytest.approx(SMALL.n_firms * .1, abs=1e-12)
        if day <= 120:
            assert not flags.any()
            continue
        # The report just before this audit against the report just after the previous one.
        before = np.minimum(result.true_goodness[day] + result.inflation[day - 1], top)
        np.testing.assert_array_equal(flags, before - report[day - 60] > .125)
        if flags.any() and not flags.all():
            assert q[flags].min() > q[~flags].max()
    assert result.audit_flags.any()


@pytest.mark.parametrize("allocation", ["value", "jump"])
def test_firms_plan_with_their_own_rates_only_under_a_public_rule(monkeypatch, allocation):
    import abm.disclosure.game as game

    calls = []
    original = game.npv_choices

    def recording(**kwargs):
        calls.append((kwargs["decision_day"], kwargs["audit_probability"], kwargs["first_audit_probability"],
                      kwargs["flag_room"]))
        return original(**kwargs)

    monkeypatch.setattr(game, "npv_choices", recording)
    run({"audit_allocation": allocation, "audit_rule_public": False})
    assert all(planned is None and first is None and room is None for _, planned, first, room in calls)
    calls.clear()
    result = run({"audit_allocation": allocation})
    for day, planned, first, room in calls:
        if day <= 120:
            assert planned is None and first is None and room is None
        elif allocation == "value":
            np.testing.assert_array_equal(planned, result.audit_probability[day // 60])
            assert first is None and room is None
        else:
            # Unflagged and flagged rates from this audit's flag count (at least one); half the
            # mean rate of .1 is random, and the other half (.6 audits) is shared by flagged firms.
            flags = max(1, int(result.audit_flags[day // 60].sum()))
            if flags < SMALL.n_firms:
                assert (first, planned) == pytest.approx((.05 + .6 / flags, .05))
            else:
                assert (first, planned) == pytest.approx((.1, .1))
            assert room.shape == (SMALL.n_firms,) and (room <= .125 + WASHING.natural_decay * 60 / 252 + 1e-12).all()


def test_targeting_rejects_unsupported_settings():
    schedule = PolicySchedule.constant(WASHING).with_intervention(121, audit_allocation="value")
    with pytest.raises(ValueError, match="targeted audits"):
        DisclosureMarket(replace(SMALL, firm_policy="npv_discrete"), WASHING, schedule)
    with pytest.raises(ValueError, match="targeted audits"):
        DisclosureMarket(SMALL, WASHING, PolicySchedule.constant(WASHING).with_intervention(
            121, audit_allocation="jump", reporting_interval_days=20))
