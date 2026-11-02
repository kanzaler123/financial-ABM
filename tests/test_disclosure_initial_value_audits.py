"""The initial_value audit allocation for contract 28.37: audit probabilities in proportion to each firm's
initial value raised to a power, at the budget of random audits, fixed for the run; exponent zero is random
audits bit for bit, and firms plan with their own rates only under a public rule."""

from dataclasses import fields, replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.policy import PolicyRegime, PolicySchedule, audit_probabilities

ESG = load_domain("configs/domains/esg.json")
# Without the proportional fine firms inflate, so detections occur in small runs.
WASHING = replace(ESG, baseline_penalty=0.0, baseline_audit_probability=0.1)
SMALL = MarketConfig(n_investors=60, n_firms=12, trading_days=400, burn_in_days=120, initial_impact_share=.5,
                     taste_median=.2)
REGIME = PolicyRegime(audit_probability=.01, penalty=0., transparency=.1, subsidy=0.)
# Result arrays that only targeted schedules record.
TARGETED_ONLY = ("audit_probability", "audit_flags")


def initial_value(beta: float, public: bool = True) -> PolicySchedule:
    """The whole run under the initial_value allocation, as in the contract 28.37 scan."""
    plain = PolicySchedule.constant(WASHING)
    return PolicySchedule(replace(plain.baseline, audit_allocation="initial_value", audit_value_exponent=beta,
                                  audit_rule_public=public))


# -- policy field and allocation ----------------------------------------------

def test_the_exponent_stays_out_of_other_identities_and_is_validated():
    for regime in (REGIME, replace(REGIME, audit_allocation="value"), replace(REGIME, audit_allocation="jump")):
        assert "audit_value_exponent" not in regime.to_dict()
    for beta in (0., -.5):
        regime = replace(REGIME, audit_allocation="initial_value", audit_value_exponent=beta)
        assert regime.to_dict()["audit_allocation"] == "initial_value"
        assert regime.to_dict()["audit_value_exponent"] == beta
        assert PolicyRegime(**regime.to_dict()) == regime
    hidden = replace(REGIME, audit_allocation="initial_value", audit_value_exponent=-1., audit_rule_public=False)
    assert hidden.to_dict()["audit_rule_public"] is False and PolicyRegime(**hidden.to_dict()) == hidden
    for bad in (np.nan, np.inf, "-1", True):
        with pytest.raises(ValueError, match="audit_value_exponent"):
            replace(REGIME, audit_allocation="initial_value", audit_value_exponent=bad)
    for allocation in ("uniform", "value", "jump"):
        with pytest.raises(ValueError, match="audit_value_exponent"):
            replace(REGIME, audit_allocation=allocation, audit_value_exponent=-.5)
    plain = PolicySchedule.constant(ESG)
    explicit = PolicySchedule(replace(plain.baseline, audit_value_exponent=0.))
    assert DisclosureMarket(SMALL, ESG, explicit).run_id == DisclosureMarket(SMALL, ESG, plain).run_id


def test_initial_values_are_required_and_checked():
    regime = replace(REGIME, audit_allocation="initial_value", audit_value_exponent=-1.)
    prices = np.ones(4)
    with pytest.raises(ValueError, match="initial value"):
        audit_probabilities(regime, prices)
    for bad in (np.ones(3), np.array([1., 0., 1., 1.]), np.array([1., np.nan, 1., 1.])):
        with pytest.raises(ValueError, match="initial values"):
            audit_probabilities(regime, prices, initial_values=bad)


def test_exponent_zero_gives_the_mean_rate_bit_for_bit():
    values = np.random.default_rng(5).lognormal(0., .9, 200)
    rates = audit_probabilities(replace(REGIME, audit_allocation="initial_value"), 3. * values, initial_values=values)
    np.testing.assert_array_equal(rates, audit_probabilities(REGIME, values))
    np.testing.assert_array_equal(rates, np.full(200, .01))


@pytest.mark.parametrize("beta", [-1.5, -1., -.5, .5])
def test_shares_follow_the_power_of_initial_value_and_keep_the_budget(beta):
    rng = np.random.default_rng(3)
    initial, prices = rng.lognormal(0., .9, 200), rng.lognormal(0., .9, 200)
    regime = replace(REGIME, audit_allocation="initial_value", audit_value_exponent=beta)
    rates = audit_probabilities(regime, prices, initial_values=initial)
    assert rates.sum() == pytest.approx(2., abs=1e-12) and rates.max() < 1.
    np.testing.assert_allclose(rates / initial**beta, rates[0] / initial[0]**beta, rtol=1e-12)
    # Market values play no part.
    np.testing.assert_array_equal(rates, audit_probabilities(regime, prices[::-1], initial_values=initial))
    ascending = rates[np.argsort(initial)]
    assert np.all(np.diff(ascending) < 0) if beta < 0 else np.all(np.diff(ascending) > 0)


def test_a_rate_above_one_is_capped_and_its_excess_shared_by_weight():
    regime = replace(REGIME, audit_allocation="initial_value", audit_value_exponent=-1., audit_probability=.4)
    initial = np.array([.001, 1., 2., 3., 4.])
    rates = audit_probabilities(regime, np.ones(5), initial_values=initial)
    assert rates[0] == 1. and rates.sum() == pytest.approx(2., abs=1e-12)
    weights = 1. / initial[1:]
    np.testing.assert_allclose(rates[1:], weights / weights.sum(), rtol=1e-12)
    # Two rounds: the excess of the first cap pushes the second firm above one as well.
    rates = audit_probabilities(replace(regime, audit_probability=.6), np.ones(5),
                                initial_values=np.array([.001, .002, 1., 1., 1.]))
    np.testing.assert_allclose(rates, [1., 1., 1 / 3, 1 / 3, 1 / 3], rtol=1e-12)


# -- the market -----------------------------------------------------------------

def assert_same_history(left, right) -> None:
    assert left.market.fingerprint() == right.market.fingerprint()
    for field in fields(left):
        if field.name in ("market", *TARGETED_ONLY):
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        assert (a is None) == (b is None), field.name
        if a is not None:
            assert a.dtype == b.dtype, field.name
            np.testing.assert_array_equal(a, b, err_msg=field.name)


@pytest.mark.parametrize("public", [True, False])
def test_exponent_zero_runs_are_random_audits_bit_for_bit(public):
    plain = DisclosureMarket(SMALL, WASHING, PolicySchedule.constant(WASHING)).run()
    targeted = DisclosureMarket(SMALL, WASHING, initial_value(0., public)).run()
    assert plain.audit_probability is None and plain.detections.sum() > 0
    np.testing.assert_array_equal(targeted.audit_probability[1:], .1)
    assert not targeted.audit_flags.any()
    assert_same_history(targeted, plain)


def test_rates_stay_fixed_at_the_initial_values_while_prices_move():
    market = DisclosureMarket(SMALL, WASHING, initial_value(-1.))
    initial = market.initial_intrinsic_value.copy()
    result = market.run()
    np.testing.assert_array_equal(initial, result.market.intrinsic_values[0])
    expected = audit_probabilities(market.policy.baseline, result.market.prices[0], initial_values=initial)
    assert expected.sum() == pytest.approx(SMALL.n_firms * .1, abs=1e-12)
    for period in range(1, result.actions.shape[0]):
        np.testing.assert_array_equal(result.audit_probability[period], expected)
    # The smaller the initial value, the higher the rate; the size classes rank the same values.
    assert np.all(np.diff(expected[np.argsort(initial)]) < 0)
    assert result.market.size_class[np.argmax(initial)] == 0 and result.market.size_class[np.argmin(initial)] == 2
    # Values and prices move during the run; the rates do not follow them.
    assert not np.array_equal(result.market.intrinsic_values[-1], initial)
    assert not np.allclose(result.market.prices[60], result.market.prices[360])


def test_firms_plan_with_their_own_rates_only_under_a_public_rule_and_a_nonzero_exponent(monkeypatch):
    import abm.disclosure.game as game

    calls = []
    original = game.npv_choices

    def recording(**kwargs):
        calls.append((kwargs["decision_day"], kwargs["audit_probability"]))
        return original(**kwargs)

    monkeypatch.setattr(game, "npv_choices", recording)
    for schedule in (initial_value(-1., public=False), initial_value(0.)):
        DisclosureMarket(SMALL, WASHING, schedule).run()
        assert calls and all(planned is None for _, planned in calls)
        calls.clear()
    result = DisclosureMarket(SMALL, WASHING, initial_value(-1.)).run()
    assert calls
    for day, planned in calls:
        np.testing.assert_array_equal(planned, result.audit_probability[day // 60])


def test_the_allocation_is_rejected_outside_the_targeting_settings():
    with pytest.raises(ValueError, match="targeted audits"):
        DisclosureMarket(replace(SMALL, firm_policy="npv_discrete"), WASHING, initial_value(-1.))
