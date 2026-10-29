"""Targeted audits and assurance under reverting true values (research contract section 25.9 (d)):
financing factors that discount reverting real changes, NPV decisions against brute force, the
jump rule's room with the expected reversion drift, and exact equality when nothing reverts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.config import TRADING_DAYS_PER_YEAR
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_signal_factors
from abm.disclosure.game import (
    IMPROVE, INFLATE, MAINTAIN, MINIMUM_NPV, DisclosureMarket, npv_choices, real_persistence,
)
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ESG = load_domain("configs/domains/esg.json")
MAIN = "configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json"
NEEDS_MAIN = pytest.mark.skipif(not Path(MAIN).exists(), reason="the CSI 300 domain config holds licensed Huazheng scores and is not in the repository (.gitignore)")
# The stage H point estimate (contract section 25.16).
Q = dict(natural_decay=0.0, natural_reversion=0.505, reversion_permanent_share=0.423)
PERSISTENCE = real_persistence(replace(ESG, **{**Q, "reversion_permanent_share": 0.0}), 200, 1260)
FLAT = np.ones(1261)
RATE, TAU = .05, .1
TIMING = dict(decision_day=60, horizon_days=1260, repricing_interval=60, period_days=60,
              implementation_delay=120, disclosure_delay=20)
REGIME = PolicyRegime(audit_probability=.01, penalty=.003, transparency=TAU, subsidy=0.)
BENEFIT = np.array([2e-4, 6e-4, 1.5e-3, 3e-3])
COST = np.array([.05, .08, .03, .12])
DECISION = dict(cost_of_capital=np.full(4, RATE), domain=ESG, period_days=60, decision_day=60, regime=REGIME,
                benefit=BENEFIT, marginal_cost=COST)


# -- direct sums ------------------------------------------------------------------

def direct_factors(q_later, q_first=None, a=0., persistence=None):
    """Real, displaced and wash annuities and the fine factor of one firm, summed day by day."""
    day, period, repricing = TIMING["decision_day"], TIMING["period_days"], TIMING["repricing_interval"]
    real_due, wash_due = day + TIMING["implementation_delay"], day + TIMING["disclosure_delay"]
    first_audit = ((wash_due + period - 1) // period) * period
    q_first = q_later if q_first is None else q_first
    real = old = wash = 0.
    for t in range(1, TIMING["horizon_days"] + 1):
        quoted = (day + t) // repricing * repricing - 1
        discount = np.exp(-RATE * t / TRADING_DAYS_PER_YEAR) / TRADING_DAYS_PER_YEAR
        # Checks on disclosure days up to and including the quote day; audits strictly before it.
        checks = len(range(wash_due, quoted + 1, period))
        if quoted >= real_due:
            kept = 1. if persistence is None else persistence[quoted - real_due]
            real += kept * discount
            old += (1 - q_later) ** len(range(day + period, quoted, period)) * (1 - a) ** checks * discount
        if quoted >= wash_due:
            audits = len(range(first_audit, quoted, period))
            survival = 1. if audits == 0 else (1 - q_first) * (1 - q_later) ** (audits - 1)
            wash += survival * (1 - a) ** checks * discount
    fine = 0.
    for k in range(5000):
        found = q_first if k == 0 else (1 - q_first) * (1 - q_later) ** (k - 1) * q_later
        fine += (found * (1 - a) ** len(range(wash_due, first_audit + k * period, period))
                 * np.exp(-RATE * (first_audit + k * period - day) / TRADING_DAYS_PER_YEAR))
    return real, real - (1 - TAU) * old, wash, fine


def brute_npv(benefit, cost, factors, inflate_cap, room=None, flagged=None):
    """Best NPV over fine grids of improvement and inflation amounts (subsidy zero, no new-case fine)."""
    cap = ESG.max_change_per_period
    real, displaced, wash, fine = factors
    amount = np.union1d(np.linspace(0., cap, 200001), [inflate_cap] + ([] if room is None else [max(room, 0.)]))
    improve = (benefit * real * np.minimum(amount, inflate_cap) + benefit * displaced * np.maximum(amount - inflate_cap, 0.)
               - cost * amount**2)
    washing = amount[amount <= inflate_cap]
    gain = max(benefit * (1 - TAU) * wash - REGIME.penalty * fine, 0.)
    value = gain * washing - ESG.wash_cost_ratio * cost * washing**2
    if room is not None:
        flagged_gain = max(benefit * (1 - TAU) * flagged[2] - REGIME.penalty * flagged[3], 0.)
        value = np.where(washing <= room, value, flagged_gain * washing - ESG.wash_cost_ratio * cost * washing**2)
    best = max(improve.max(), value.max())
    return best if best > MINIMUM_NPV else 0.


def chosen(choice):
    return np.where(choice.actions == MAINTAIN, 0., choice.npv)


# -- financing factors --------------------------------------------------------------

@pytest.mark.parametrize("case", ["firm_rates", "first_audit", "assured"])
def test_factors_with_reversion_match_a_direct_sum(case):
    rates = np.array([.002, .03, .2])
    for firm, q in enumerate(rates):
        if case == "firm_rates":
            arguments, expected = dict(audit_probability=rates), direct_factors(q, persistence=PERSISTENCE)
        elif case == "first_audit":
            arguments = dict(audit_probability=.005, first_audit_probability=q)
            expected = direct_factors(.005, q_first=q, persistence=PERSISTENCE)
        else:
            arguments = dict(audit_probability=.01, assurance_probability=rates)
            expected = direct_factors(.01, a=q, persistence=PERSISTENCE)
        factors = financing_signal_factors(np.full(3, RATE), transparency=TAU, real_persistence=PERSISTENCE,
                                           **arguments, **TIMING)
        for value, target in zip(factors, expected):
            assert value[firm] == pytest.approx(target, rel=1e-9, abs=1e-15)
    # Reversion lowers the real annuities and leaves wash survival and detection alone.
    plain = financing_signal_factors(np.full(3, RATE), transparency=TAU, **arguments, **TIMING)
    assert (factors[0] < plain[0]).all() and (factors[1] < plain[1]).all()
    np.testing.assert_array_equal(factors[2], plain[2])
    np.testing.assert_array_equal(factors[3], plain[3])


@pytest.mark.parametrize("arguments", [
    dict(audit_probability=np.array([.002, .03, .2])),
    dict(audit_probability=.005, first_audit_probability=.2),
    dict(audit_probability=.01, assurance_probability=np.array([0., .05, .5])),
])
def test_factors_without_reversion_are_unchanged_bit_for_bit(arguments):
    plain = financing_signal_factors(np.full(3, RATE), transparency=TAU, **arguments, **TIMING)
    flat = financing_signal_factors(np.full(3, RATE), transparency=TAU, real_persistence=FLAT, **arguments, **TIMING)
    for left, right in zip(plain, flat):
        np.testing.assert_array_equal(left, right)


# -- the firm's decision against brute force ------------------------------------------

INFLATE_CAP = np.array([.2, .02, .1, .2])


def test_firm_rate_decisions_under_reversion_match_brute_force():
    rates = np.array([.002, .01, .05, .2])
    choice = npv_choices(**DECISION, audit_probability=rates, real_persistence=PERSISTENCE, inflate_capacity=INFLATE_CAP)
    expected = [brute_npv(BENEFIT[i], COST[i], direct_factors(q, persistence=PERSISTENCE), INFLATE_CAP[i])
                for i, q in enumerate(rates)]
    np.testing.assert_allclose(chosen(choice), expected, rtol=1e-6, atol=1e-12)
    assert {IMPROVE, INFLATE} <= set(choice.actions)


@pytest.mark.parametrize("room", [-.05, .03, .08, .3])
def test_jump_rule_decisions_under_reversion_match_brute_force(room):
    later, first = .005, .2
    choice = npv_choices(**DECISION, audit_probability=later, first_audit_probability=first,
                         flag_room=np.full(4, room), real_persistence=PERSISTENCE, inflate_capacity=INFLATE_CAP)
    unflagged = direct_factors(later, persistence=PERSISTENCE)
    flagged = direct_factors(later, q_first=first, persistence=PERSISTENCE)
    expected = [brute_npv(BENEFIT[i], COST[i], unflagged, INFLATE_CAP[i], room=room, flagged=flagged) for i in range(4)]
    np.testing.assert_allclose(chosen(choice), expected, rtol=1e-6, atol=1e-12)
    assert INFLATE in choice.actions


def test_assured_decisions_under_reversion_match_brute_force():
    checks = np.array([0., .02, .08, .4])
    choice = npv_choices(**DECISION, assurance_probability=checks, real_persistence=PERSISTENCE,
                         inflate_capacity=INFLATE_CAP)
    expected = [brute_npv(BENEFIT[i], COST[i], direct_factors(.01, a=a, persistence=PERSISTENCE), INFLATE_CAP[i])
                for i, a in enumerate(checks)]
    np.testing.assert_allclose(chosen(choice), expected, rtol=1e-6, atol=1e-12)
    assert {IMPROVE, INFLATE} <= set(choice.actions)


@pytest.mark.parametrize("arguments", [
    dict(audit_probability=np.array([.002, .01, .05, .2])),
    dict(audit_probability=.005, first_audit_probability=.2, flag_room=np.array([-.05, .03, .08, .3])),
    dict(assurance_probability=np.array([0., .02, .08, .4])),
])
def test_decisions_without_reversion_are_unchanged_bit_for_bit(arguments):
    plain = npv_choices(**DECISION, **arguments, inflate_capacity=INFLATE_CAP)
    flat = npv_choices(**DECISION, **arguments, inflate_capacity=INFLATE_CAP, real_persistence=FLAT)
    for name in ("actions", "amounts", "npv", "cost_fraction", "subsidy_fraction"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(flat, name))


# -- the jump rule's room in the market ------------------------------------------------

SMALL = MarketConfig(n_investors=60, n_firms=12, trading_days=400, burn_in_days=120, initial_impact_share=.5,
                     taste_median=.2)
JUMP = dict(audit_allocation="jump")


def _jump_market(domain):
    schedule = PolicySchedule.constant(domain).with_intervention(121, **JUMP)
    market = DisclosureMarket(SMALL, domain, schedule)
    state = market.attributes
    state.goodness = np.linspace(-1.5, 1.2, SMALL.n_firms)
    if state.anchor is not None:
        state.anchor = np.linspace(.4, -.6, SMALL.n_firms)
    state.pending = [(150, IMPROVE, np.array([1, 4]), np.array([.05, .02])),
                     (200, INFLATE, np.array([4]), np.array([.03])),
                     (230, IMPROVE, np.array([6]), np.array([.07]))]
    return market, schedule.intervention


@NEEDS_MAIN
@pytest.mark.parametrize("dynamics", ["decay", "reversion", "q", "q_permanent"])
def test_jump_room_adds_the_expected_reversion_drift(dynamics):
    base = load_domain(MAIN)
    domain = {
        "decay": base,
        "reversion": replace(base, natural_decay=0., natural_reversion=.3),
        "q": replace(base, **Q),
        "q_permanent": replace(base, **Q, reversion_improvement="permanent"),
    }[dynamics]
    market, regime = _jump_market(domain)
    _, _, room = market._jump_plan(120, regime, np.zeros(SMALL.n_firms, dtype=bool))
    landing = np.zeros(SMALL.n_firms)
    landing[[1, 4]] += [.05, .02]
    old = regime.audit_jump_threshold - landing + domain.natural_decay * 60 / TRADING_DAYS_PER_YEAR
    if dynamics == "decay":
        np.testing.assert_array_equal(room, old)
        return
    state = market.attributes
    x = state.goodness if state.anchor is None else state.goodness - state.anchor
    drift = (1 - np.exp(-domain.natural_reversion * 60 / TRADING_DAYS_PER_YEAR)) * (x - x.mean())
    np.testing.assert_allclose(room, old + drift, rtol=0, atol=1e-14)
    # Firms above the mean expect their report to fall back, so they have more room.
    assert (room[x > x.mean()] > old[x > x.mean()]).all() and (room[x < x.mean()] < old[x < x.mean()]).all()


# -- the market with reverting true values -----------------------------------------------

WASHING = (replace(load_domain(MAIN), baseline_penalty=0.0, baseline_audit_probability=0.1, **Q)
           if Path(MAIN).exists() else None)
ASSURED = dict(assurance_scope="all", assurer_effort_cost=1e-4, assurer_liability=2e-3, assurance_fee=1e-6,
               assurer_selection="firm")


@NEEDS_MAIN
@pytest.mark.parametrize("changes", [dict(audit_allocation="value"), JUMP, ASSURED])
def test_markets_with_reversion_plan_with_its_persistence(monkeypatch, changes):
    import abm.disclosure.game as game

    calls = []
    original = game.npv_choices

    def recording(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(game, "npv_choices", recording)
    config = replace(SMALL, trading_days=700)
    schedule = PolicySchedule.constant(WASHING).with_intervention(121, **changes)
    result = DisclosureMarket(config, WASHING, schedule).run()
    expected = real_persistence(WASHING, config.n_firms, config.financing_horizon_days)
    assert calls and all(call["real_persistence"] is not None for call in calls)
    np.testing.assert_array_equal(calls[-1]["real_persistence"], expected)
    targeted = [call for call in calls if call["decision_day"] > 121]
    if "assurance_scope" in changes:
        assert any(call["assurance_probability"] is not None and np.any(call["assurance_probability"] > 0)
                   for call in targeted)
    elif changes["audit_allocation"] == "value":
        assert all(np.ndim(call["audit_probability"]) == 1 for call in targeted)
    else:
        assert all(call["flag_room"] is not None for call in targeted)
    assert np.isfinite(result.true_goodness).all() and (result.actions[2:] != MAINTAIN).any()


def test_remaining_restrictions_still_hold():
    reverting = replace(ESG, natural_decay=0., natural_reversion=.3)
    for changes in (dict(audit_allocation="value", reporting_interval_days=20),
                    dict(audit_allocation="jump", publish_audit_results=True)):
        with pytest.raises(ValueError):
            DisclosureMarket(SMALL, reverting, PolicySchedule.constant(reverting).with_intervention(121, **changes))
    with pytest.raises(ValueError, match="targeted audits"):
        DisclosureMarket(replace(SMALL, firm_policy="npv_discrete"), reverting,
                         PolicySchedule.constant(reverting).with_intervention(121, audit_allocation="value"))
    with pytest.raises(ValueError, match="assurance is modelled only"):
        DisclosureMarket(SMALL, reverting, PolicySchedule.constant(reverting).with_intervention(
            121, **{k: v for k, v in ASSURED.items() if k != "assurer_selection"}, disclosure_delay_days=60))
