"""Assurance providers for contract section 22 (stage E.2): policy fields, financing factors with
pre-publication checks, providers' yearly quality and clients, and defaults that keep earlier runs
unchanged."""

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.config import TRADING_DAYS_PER_YEAR
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_signal_factors
from abm.disclosure.game import ASSURANCE_ARRAYS, DisclosureMarket, npv_choices
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
# Without the proportional fine firms inflate, so checks find inflation in small runs.
WASHING = replace(ESG, baseline_penalty=0.0, baseline_audit_probability=0.1)
SMALL = MarketConfig(n_investors=60, n_firms=12, trading_days=700, burn_in_days=120, initial_impact_share=.5,
                     taste_median=.2)
REGIME = PolicyRegime(audit_probability=.01, penalty=0., transparency=.1, subsidy=0.)
TIMING = dict(decision_day=60, horizon_days=1260, repricing_interval=60, period_days=60,
              implementation_delay=120, disclosure_delay=20)
# lambda = 2e-3 * 0.1 / (2 * 1e-4) = 1 at the washing domain's audit probability.
ASSURED = dict(assurance_scope="all", assurer_effort_cost=1e-4, assurer_liability=2e-3, assurance_fee=1e-6)
SWITCH = 121
FIRST = 180  # first decision day under the switched regime: the first assurance year starts here


# -- policy fields --------------------------------------------------------------

def test_defaults_are_omitted_from_identities_and_validated():
    assert not {"assurance_scope", "assurer_selection", "assurer_liability", "assurance_fee",
                "assurer_effort_cost", "assurer_count"} & set(REGIME.to_dict())
    assured = replace(REGIME, assurance_scope="all", assurer_effort_cost=1e-4, assurer_liability=2e-3)
    payload = assured.to_dict()
    assert payload["assurance_scope"] == "all" and payload["assurer_liability"] == 2e-3
    assert not {"assurer_selection", "assurance_fee", "assurer_count"} & set(payload)
    assert PolicyRegime(**payload) == assured
    assert assured.assurer_slope == pytest.approx(2e-3 * .01 / 2e-4)
    assert REGIME.assurer_slope == 0.
    picked = replace(assured, assurer_selection="firm", assurer_count=3, assurance_fee=1e-6)
    assert PolicyRegime(**picked.to_dict()) == picked
    with pytest.raises(ValueError, match="need an assurance_scope"):
        replace(REGIME, assurer_liability=1e-3)
    with pytest.raises(ValueError, match="positive assurer_effort_cost"):
        replace(REGIME, assurance_scope="all")
    with pytest.raises(ValueError, match="assurance_scope"):
        replace(assured, assurance_scope="small")
    with pytest.raises(ValueError, match="assurer_selection"):
        replace(assured, assurer_selection="regulator")
    for bad in (0, -1, 2.):
        with pytest.raises(ValueError, match="assurer_count"):
            replace(assured, assurer_count=bad)
    for bad in (-1e-3, np.inf, True):
        with pytest.raises(ValueError, match="assurer_liability"):
            replace(assured, assurer_liability=bad)
    plain = PolicySchedule.constant(ESG)
    explicit = PolicySchedule(replace(plain.baseline, assurance_scope="none", assurer_count=4))
    assert DisclosureMarket(SMALL, ESG, explicit).run_id == DisclosureMarket(SMALL, ESG, plain).run_id


# -- financing factors with checks ------------------------------------------------

def test_assured_factors_reduce_to_the_unassured_ones():
    rates = np.full(3, .05)
    for q in (0., .01, .3):
        plain = financing_signal_factors(rates, audit_probability=q, transparency=.1, **TIMING)
        checked = financing_signal_factors(rates, audit_probability=q, transparency=.1,
                                           assurance_probability=np.zeros(3), **TIMING)
        for left, right in zip(plain, checked):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("a", [.02, .07, .5, 1.])
def test_assured_factors_match_a_direct_sum(a):
    rate, q, tau = .05, .02, .1
    real, displaced, wash, fine = financing_signal_factors(np.array([rate]), audit_probability=q, transparency=tau,
                                                           assurance_probability=np.array([a]), **TIMING)
    day, period, repricing = TIMING["decision_day"], TIMING["period_days"], TIMING["repricing_interval"]
    wash_due, real_due = day + TIMING["disclosure_delay"], day + TIMING["implementation_delay"]
    first_audit = ((wash_due + period - 1) // period) * period
    expected_real = expected_wash = expected_old = 0.
    for t in range(1, TIMING["horizon_days"] + 1):
        quoted = (day + t) // repricing * repricing - 1
        discount = np.exp(-rate * t / TRADING_DAYS_PER_YEAR) / TRADING_DAYS_PER_YEAR
        # Checks on disclosure days up to and including the quote day; audits strictly before it.
        checks = len(range(wash_due, quoted + 1, period))
        if quoted >= real_due:
            expected_real += discount
            expected_old += (1 - q) ** len(range(day + period, quoted, period)) * (1 - a) ** checks * discount
        if quoted >= wash_due:
            expected_wash += (1 - q) ** len(range(first_audit, quoted, period)) * (1 - a) ** checks * discount
    expected_fine = sum(q * (1 - q) ** k * (1 - a) ** len(range(wash_due, first_audit + k * period, period))
                        * np.exp(-rate * (first_audit + k * period - day) / TRADING_DAYS_PER_YEAR)
                        for k in range(5000))
    assert real[0] == pytest.approx(expected_real, rel=1e-12)
    assert wash[0] == pytest.approx(expected_wash, rel=1e-12, abs=1e-15)
    assert displaced[0] == pytest.approx(expected_real - (1 - tau) * expected_old, rel=1e-12)
    assert fine[0] == pytest.approx(expected_fine, rel=1e-9, abs=1e-15)


def test_assurance_is_rejected_outside_the_main_timing():
    rates, checks = np.full(2, .05), np.full(2, .1)
    for changes in (dict(publish_audit_results=True), dict(reporting_interval_days=5),
                    dict(real_persistence=np.ones(1261)), dict(first_audit_probability=.2),
                    dict(audit_probability=np.full(2, .01))):
        arguments = {**TIMING, "audit_probability": .01, **changes}
        with pytest.raises(ValueError, match="assurance checks are modelled only"):
            financing_signal_factors(rates, transparency=.1, assurance_probability=checks, **arguments)
    for changes in (dict(decision_day=70), dict(disclosure_delay=60), dict(disclosure_delay=0)):
        with pytest.raises(ValueError, match="decisions on audit days"):
            financing_signal_factors(rates, audit_probability=.01, transparency=.1, assurance_probability=checks,
                                     **{**TIMING, **changes})


# -- the firm's decision ------------------------------------------------------

DECISION = dict(cost_of_capital=np.full(4, .05), domain=ESG, period_days=60, decision_day=60)
BENEFIT = np.array([2e-4, 5e-4, 1e-3, 3e-3])
COST = np.array([.05, .08, .03, .12])


def test_unassured_firms_plan_exactly_as_before():
    default = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, **DECISION)
    zero = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, assurance_probability=np.zeros(4),
                       **DECISION)
    mixed = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME,
                        assurance_probability=np.array([0., .3, 0., .3]), **DECISION)
    for name in ("actions", "amounts", "npv", "cost_fraction", "subsidy_fraction"):
        np.testing.assert_array_equal(getattr(default, name), getattr(zero, name))
    for firm in (0, 2):
        assert (mixed.actions[firm], mixed.amounts[firm], mixed.npv[firm]) == (
            default.actions[firm], default.amounts[firm], default.npv[firm])


def test_each_firm_plans_with_its_own_assurance():
    checks = np.array([.005, .02, .08, .4])
    together = npv_choices(benefit=BENEFIT, marginal_cost=COST, regime=REGIME, assurance_probability=checks,
                           **DECISION)
    for firm, a in enumerate(checks):
        alone = npv_choices(benefit=BENEFIT[[firm]], marginal_cost=COST[[firm]], regime=REGIME,
                            assurance_probability=np.array([a]), **{**DECISION, "cost_of_capital": np.full(1, .05)})
        assert together.actions[firm] == alone.actions[0]
        assert together.amounts[firm] == pytest.approx(alone.amounts[0], rel=1e-12)
        assert together.npv[firm] == pytest.approx(alone.npv[0], rel=1e-12)


def test_strict_checks_stop_inflation():
    # Firms that inflate without checks stop once checks find inflation for sure.
    benefit = np.full(4, 3e-3)
    unchecked = npv_choices(benefit=benefit, marginal_cost=COST, regime=REGIME, **DECISION)
    certain = npv_choices(benefit=benefit, marginal_cost=COST, regime=REGIME, assurance_probability=np.ones(4),
                          **DECISION)
    assert (unchecked.actions == 1).any() and not (certain.actions == 1).any()


# -- the market -----------------------------------------------------------------

def schedule(changes: dict, domain=WASHING) -> PolicySchedule:
    return PolicySchedule.constant(domain).with_intervention(SWITCH, **changes)


def run(changes: dict, domain=WASHING, config=SMALL):
    return DisclosureMarket(config, domain, schedule(changes, domain)).run()


@pytest.fixture(scope="module")
def assured():
    return run(ASSURED)


def test_defaults_keep_results_and_archive_no_new_arrays(assured):
    plain = run({})
    assert all(getattr(plain, name) is None for name in ASSURANCE_ARRAYS)
    assert all(getattr(assured, name) is not None for name in ASSURANCE_ARRAYS)
    # Up to the first decision under assurance the assured run is the plain run.
    periods = FIRST // 60
    np.testing.assert_array_equal(assured.actions[:periods], plain.actions[:periods])
    np.testing.assert_array_equal(assured.inflation[:FIRST + 1], plain.inflation[:FIRST + 1])
    np.testing.assert_array_equal(assured.market.prices[:FIRST + 1], plain.market.prices[:FIRST + 1])


def test_checks_and_quality_follow_the_rule(assured):
    h = {name: getattr(assured, name) for name in ASSURANCE_ARRAYS}
    first = FIRST // 60
    assert (h["assurer_assignment"][:first] == -1).all() and not h["assurance_corrections"][:first].any()
    assignment = h["assurer_assignment"]
    assert (assignment[first:] >= 0).all()  # every firm is mandated; assigned once, evenly
    np.testing.assert_array_equal(np.bincount(assignment[first], minlength=4), [3, 3, 3, 3])
    assert (assignment[first:] == assignment[first]).all()
    corrections, inflated = h["assurance_corrections"], h["assurance_checked_inflated"]
    assert not (corrections & ~inflated).any()
    for period in np.flatnonzero(corrections.any(axis=1)):
        check_day = period * 60 + 20
        np.testing.assert_array_equal(assured.inflation[check_day][corrections[period]], 0.)
    lam = PolicyRegime(**{**REGIME.to_dict(), "audit_probability": .1, **ASSURED}).assurer_slope
    # First year: the share of each provider's clients whose report carries inflation after the audit.
    inflation = assured.inflation[FIRST]
    share = np.array([np.mean(inflation[assignment[first] == k] > 0) for k in range(4)])
    np.testing.assert_allclose(h["assurer_inflation_share"][first], share, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h["assurer_quality"][first], np.minimum(1., lam * share), rtol=0, atol=1e-15)
    # Second year: the share of the past year's checks that found inflation.
    second = first + 4
    year = slice(first, second)
    share = np.array([inflated[year][assignment[year] == k].mean() for k in range(4)])
    np.testing.assert_allclose(h["assurer_inflation_share"][second], share, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h["assurer_quality"][second], np.minimum(1., lam * share), rtol=0, atol=1e-15)
    for start in (first, second):
        assert (h["assurer_quality"][start:start + 4] == h["assurer_quality"][start]).all()
    assert inflated.any() and corrections.any()
    # An audit charges a provider only for an assured firm's detected inflation.
    events = h["assurer_liability_events"].sum(axis=1)
    assert (events <= (assured.detections * (assignment >= 0)).sum(axis=1)).all()
    assert events[:first + 1].sum() == 0


def test_no_liability_means_no_quality():
    result = run({**ASSURED, "assurer_liability": 0.})
    assert not result.assurer_quality.any() and not result.assurance_corrections.any()
    assert result.assurance_checked_inflated.any()


def test_voluntary_assurance_is_never_bought():
    plain, voluntary = run({}), run({**ASSURED, "assurance_scope": "voluntary"})
    assert (voluntary.assurer_assignment == -1).all() and not voluntary.assurer_quality.any()
    for name in ("actions", "amounts", "inflation", "true_goodness", "detections"):
        np.testing.assert_array_equal(getattr(voluntary, name), getattr(plain, name))
    np.testing.assert_array_equal(voluntary.market.prices, plain.market.prices)


def test_large_scope_checks_only_large_firms():
    result = run({**ASSURED, "assurance_scope": "large"})
    large = result.market.size_class == 0
    assert large.any() and not large.all()
    assignment = result.assurer_assignment[FIRST // 60:]
    assert (assignment[:, large] >= 0).all() and (assignment[:, ~large] == -1).all()
    assert not result.assurance_checked_inflated[:, ~large].any()


def test_firms_that_pick_take_the_provider_worth_most_to_them():
    regime_changes = {**ASSURED, "assurer_selection": "firm"}
    market = DisclosureMarket(SMALL, WASHING, schedule(regime_changes))
    regime = market.policy.intervention
    market.attributes.inflation[:] = np.where(np.arange(12) % 3 == 0, .1, 0.)
    initial = np.empty(12, dtype=np.int64)
    initial[np.random.default_rng(np.random.SeedSequence([SMALL.seed, 2])).permutation(12)] = np.arange(12) % 4
    weight = np.where(np.arange(12) < 6, 1., 0.)  # firms 0-5 lose value with quality, the rest are indifferent
    market._assurance_year(FIRST, regime, lambda a: -weight * a)
    quality = market.assurance.quality
    share = np.array([np.mean(market.attributes.inflation[initial == k] > 0) for k in range(4)])
    np.testing.assert_allclose(quality, np.minimum(1., regime.assurer_slope * share), rtol=0, atol=1e-15)
    chosen = market.assurance.client_of
    for firm in range(12):
        if weight[firm] and quality[initial[firm]] > quality.min():
            assert quality[chosen[firm]] == quality.min()
        elif not weight[firm] or quality[initial[firm]] == quality.min():
            assert chosen[firm] == initial[firm]
    # Not the start of a year: nothing changes.
    before = chosen.copy()
    market._assurance_year(FIRST + 60, regime, lambda a: -a)
    np.testing.assert_array_equal(market.assurance.client_of, before)


def test_assurance_rejects_unsupported_settings():
    with pytest.raises(ValueError, match="policy switch"):
        DisclosureMarket(SMALL, WASHING, PolicySchedule(replace(PolicyRegime.baseline(WASHING), **ASSURED)))
    with pytest.raises(ValueError, match="assurance is modelled only"):
        DisclosureMarket(replace(SMALL, firm_policy="npv_discrete"), WASHING, schedule(ASSURED))
    with pytest.raises(ValueError, match="assurance is modelled only"):
        DisclosureMarket(SMALL, replace(WASHING, natural_decay=0., natural_reversion=.3), schedule(ASSURED))
    for changes in ({"audit_allocation": "value"}, {"reporting_interval_days": 20},
                    {"publish_audit_results": True}, {"disclosure_delay_days": 60}):
        with pytest.raises(ValueError, match="assurance is modelled only"):
            DisclosureMarket(SMALL, WASHING, schedule({**ASSURED, **changes}))
