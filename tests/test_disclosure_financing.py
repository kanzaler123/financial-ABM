from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_discount_factors, financing_signal_factors
from abm.disclosure.game import MAINTAIN, DisclosureMarket
from abm.disclosure.metrics import compute_metrics
from abm.disclosure.runner import load_result, save_result


DOMAIN = load_domain("configs/domains/esg.json")
CONFIG = MarketConfig(n_investors=30, n_firms=6, trading_days=80, burn_in_days=20,
                      initial_impact_share=1.0)


def test_contract_locks_until_repricing_and_uses_previous_quote():
    market = DisclosureMarket(CONFIG, DOMAIN)
    initial = market.contract_rate.copy()
    exposure = market.financing_exposure.copy()
    market.prices *= 2
    market._pre_trade(1)
    np.testing.assert_array_equal(market.contract_rate, initial)
    np.testing.assert_array_equal(market.financing_adjustment_history[1], 0)
    before = market.firms.intrinsic_value.copy()
    news = market._pre_trade(60)
    expected = CONFIG.risk_free_rate + CONFIG.financing_spread * exposure / (exposure + market.prices)
    np.testing.assert_array_equal(market.contract_rate, expected)
    np.testing.assert_array_equal(market.financing_exposure, exposure)
    adjustment = exposure * (initial - expected) / 252
    np.testing.assert_allclose(market.firms.intrinsic_value - before, adjustment)
    np.testing.assert_allclose(news, np.log1p(adjustment / before), atol=1e-15)
    np.testing.assert_allclose(market.financing_cost_history[60], exposure * expected / 252)


def test_financing_derivative_matches_fee_difference_not_quote_gain(monkeypatch):
    market = DisclosureMarket(CONFIG, DOMAIN)
    response = np.linspace(.01, .06, CONFIG.n_firms)
    monkeypatch.setattr(market, "marginal_signal_value", lambda: response)
    predicted = market.marginal_financing_saving()
    prices = market.prices.copy()
    before = market.quoted_financing_rate()
    epsilon = 1e-5
    market.prices *= np.exp(epsilon * response)
    after = market.quoted_financing_rate()
    finite_difference = market.financing_exposure * (before - after) / (prices * epsilon)
    np.testing.assert_allclose(predicted, finite_difference, rtol=1e-6)


@pytest.mark.parametrize("changes", [{"financing_exposure_fraction": 0.0}, {"financing_spread": 0.0}])
def test_closed_financing_channel_has_no_residual_quote_reward(changes, monkeypatch):
    market = DisclosureMarket(replace(CONFIG, **changes), DOMAIN)
    monkeypatch.setattr(market, "marginal_signal_value", lambda: np.ones(CONFIG.n_firms))
    np.testing.assert_array_equal(market.marginal_financing_saving(), 0)
    market._audit_and_decide(60, 1)
    np.testing.assert_array_equal(market.action_history[1], MAINTAIN)
    np.testing.assert_array_equal(market.cost_history[1], 0)


def test_daily_cashflow_timing_with_certain_audit_and_zero_discount():
    real, wash, fine = financing_discount_factors(
        np.array([0.0]), decision_day=0, horizon_days=8, repricing_interval=2,
        period_days=2, implementation_delay=2, disclosure_delay=0, audit_probability=1.0,
    )
    # Real: due day 2, quote day 2, first reprice day 4: days 4..8.
    # Wash: due day 1, reprice day 2 before audit; fee stays locked days 2,3.
    np.testing.assert_allclose(real, [5 / 252])
    np.testing.assert_allclose(wash, [2 / 252])
    np.testing.assert_array_equal(fine, 1)


@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
def test_discount_factors_match_enumerated_audit_paths(probability):
    rate, decision, horizon = 0.08, 3, 8
    actual = financing_discount_factors(
        np.array([rate]), decision_day=decision, horizon_days=horizon, repricing_interval=3,
        period_days=4, implementation_delay=2, disclosure_delay=1, audit_probability=probability,
    )
    real = wash = 0.0
    for time in range(1, horizon + 1):
        day = decision + time
        reprice = max([0] + [d for d in range(1, day + 1) if d % 3 == 0])
        quote = reprice - 1
        discount = np.exp(-rate * time / 252) / 252
        real += discount if quote >= 5 else 0
        if quote >= 4:
            chance = 1.0
            for _ in range(4, quote, 4):
                chance *= 1 - probability
            wash += discount * chance
    fine = sum(probability * (1 - probability)**index * np.exp(-rate * (1 + 4 * index) / 252)
               for index in range(20000))
    np.testing.assert_allclose(np.array(actual)[:, 0], [real, wash, fine], atol=1e-12)


def test_horizon_before_repricing_has_no_financing_saving():
    real, wash, fine = financing_discount_factors(
        np.array([0.05]), decision_day=60, horizon_days=20, repricing_interval=60,
        period_days=60, implementation_delay=0, disclosure_delay=0, audit_probability=.1,
    )
    np.testing.assert_array_equal(real, 0)
    np.testing.assert_array_equal(wash, 0)
    assert fine[0] > 0  # A future liability is not erased at the forecast horizon.


def _enumerate_report_paths(*, decision, horizon, reporting, period, repricing,
                            real_delay, wash_delay, probability, publish, known, rate):
    real_due, wash_due = decision + max(1, real_delay), decision + max(1, wash_delay)
    audits = [day for day in range(decision + 1, decision + horizon + 1) if day % period == 0]
    answer = np.zeros(3)
    for path in product((False, True), repeat=len(audits)):
        selected = dict(zip(audits, path))
        chance = np.prod([probability if checked else 1 - probability for checked in path])
        baseline_inflation = np.array([0., 1., 0.])
        action_inflation = baseline_inflation.copy()
        action_goodness = np.zeros(3)
        cached_difference = np.zeros(3)
        certificate = (decision, np.zeros(3)) if publish and known else None
        quotes = {}
        for day in range(decision + 1, decision + horizon + 1):
            if day == real_due:
                action_goodness[:2] = 1
                action_inflation[1] = 0
            if day == wash_due:
                action_inflation[2] = 1
            if day % reporting == 0:
                cached_difference = action_goodness + .75 * (action_inflation - baseline_inflation)
            if certificate is not None and certificate[0] < day <= certificate[0] + period:
                quotes[day] = certificate[1].copy()
            else:
                quotes[day] = cached_difference.copy()
            if selected.get(day, False):
                baseline_inflation[:] = action_inflation[:] = 0
                cached_difference = action_goodness.copy()
                if publish:
                    certificate = day, action_goodness.copy()
        for day in range(decision + 1, decision + horizon + 1):
            last_reprice = max([0] + [date for date in range(1, day + 1) if date % repricing == 0])
            if last_reprice - 1 in quotes:
                answer += chance * quotes[last_reprice - 1] * np.exp(-rate * (day - decision) / 252) / 252
    answer[2] /= .75  # The caller applies the common transparency weight to wash.
    return answer


@pytest.mark.parametrize("reporting", [1, 3, 7])
@pytest.mark.parametrize("probability", [0., .35, 1.])
@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("calendar", [(0, 9, 3, 2, 2, 2), (4, 16, 4, 3, 5, 6),
                                       (4, 9, 4, 12, 0, 1)])
def test_periodic_report_prediction_matches_counterfactual_record_paths(reporting, probability, publish, calendar):
    decision, horizon, period, repricing, real_delay, wash_delay = calendar
    rates, known = np.array([0., .08]), np.array([False, True])
    args = dict(decision_day=decision, horizon_days=horizon, repricing_interval=repricing,
                period_days=period, implementation_delay=real_delay, disclosure_delay=wash_delay,
                audit_probability=probability, transparency=.25, publish_audit_results=publish,
                published_now=known)
    actual = financing_signal_factors(rates, reporting_interval_days=reporting, **args)
    for firm in range(2):
        expected = _enumerate_report_paths(
            decision=decision, horizon=horizon, reporting=reporting, period=period, repricing=repricing,
            real_delay=real_delay, wash_delay=wash_delay, probability=probability, publish=publish,
            known=bool(known[firm]), rate=rates[firm],
        )
        np.testing.assert_allclose(np.asarray(actual[:3])[:, firm], expected, atol=1e-14)
    np.testing.assert_array_equal(actual[3], financing_signal_factors(rates, **args)[3])


@pytest.mark.parametrize("calendar, expected", [
    ((4, 3, 2, 8, 2, 0.), (3., .75, 3.)),
    ((8, 3, 2, 8, 2, 1.), (3., 3., 0.)),
    ((3, 3, 2, 6, 3, 1.), (3., 1.5, 2.)),
    ((3, 3, 1, 8, 4, 1.), (2., 2., 1.)),
])
def test_periodic_report_audit_and_repricing_boundaries(calendar, expected):
    reporting, period, repricing, horizon, delay, probability = calendar
    actual = financing_signal_factors(
        np.array([0.]), decision_day=0, horizon_days=horizon, repricing_interval=repricing,
        period_days=period, implementation_delay=delay, disclosure_delay=delay,
        audit_probability=probability, transparency=.25, reporting_interval_days=reporting,
    )
    np.testing.assert_allclose(np.asarray(actual[:3])[:, 0] * 252, expected, atol=1e-14)
    if reporting == 8:
        assert actual[3][0] == 1.0


def test_accrual_ledger_metrics_and_archive_roundtrip(tmp_path):
    result = DisclosureMarket(CONFIG, DOMAIN).run()
    metrics = compute_metrics(result, CONFIG, DOMAIN)["financing"]
    assert metrics["status"] == "modeled_accrual"
    assert metrics["maximum_ledger_error"] < 1e-14
    assert metrics["baseline_accrued_cost"] - metrics["accrued_cost"] == pytest.approx(metrics["value_adjustment"])
    save_result(tmp_path / "result.npz", result)
    assert load_result(tmp_path / "result.npz").fingerprint() == result.fingerprint()
    legacy = replace(result, financing_rate=None, financing_cost=None, financing_value_adjustment=None)
    save_result(tmp_path / "legacy.npz", legacy)
    loaded = load_result(tmp_path / "legacy.npz")
    assert loaded.fingerprint() == legacy.fingerprint()
    assert compute_metrics(loaded, CONFIG, DOMAIN)["financing"]["status"] == "not_recorded"


def test_exhausted_value_anchor_fails_instead_of_creating_value():
    domain = replace(DOMAIN, baseline_audit_probability=1.0, baseline_penalty=100.0)
    market = DisclosureMarket(CONFIG, domain)
    market.attributes.inflation[:] = 1
    with pytest.raises(ValueError, match="exhausts the value anchor"):
        market._audit_and_decide(60, 1)


def test_committed_money_is_not_rescaled_by_next_day_fundamental_news():
    domain = replace(DOMAIN, baseline_audit_probability=1.0, baseline_penalty=.01)
    market = DisclosureMarket(replace(CONFIG, financing_exposure_fraction=0.0), domain)
    market.prices[:] = 100
    market.firms.intrinsic_value[:] = 100
    market.attributes.inflation[:] = 1
    market._audit_and_decide(60, 1)
    np.testing.assert_array_equal(market.penalty_history[1] * market.prices, 1)
    market.firms.intrinsic_value[:] = 200
    news = market._pre_trade(61)
    np.testing.assert_array_equal(market.firms.intrinsic_value, 199)
    np.testing.assert_allclose(news, np.log(199 / 200))
    market._pre_trade(62)
    np.testing.assert_array_equal(market.firms.intrinsic_value, 199)


def test_expenditure_is_checked_again_at_its_application_day():
    domain = replace(DOMAIN, baseline_audit_probability=1.0, baseline_penalty=.01)
    market = DisclosureMarket(replace(CONFIG, financing_exposure_fraction=0.0), domain)
    market.prices[:] = 100
    market.firms.intrinsic_value[:] = 100
    market.attributes.inflation[:] = 1
    market._audit_and_decide(60, 1)
    market.firms.intrinsic_value[:] = .5
    with pytest.raises(ValueError, match="exhausts the value anchor"):
        market._pre_trade(61)
