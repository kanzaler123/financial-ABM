"""The debt rate's direct response to the public rating (research contract section 26.20): the quoted
rate, the firms' marginal saving, NPV decisions against brute force, the realized saving against the
financing factors, and exact equality with the price-only contract when the slope is zero."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.config import TRADING_DAYS_PER_YEAR
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_signal_factors
from abm.disclosure.game import IMPROVE, INFLATE, DisclosureMarket, npv_choices
from abm.disclosure.policy import PolicySchedule

from test_disclosure_reversion_targeting import (
    ASSURED, DECISION, INFLATE_CAP, MAIN, PERSISTENCE, Q, RATE, TAU, TIMING, brute_npv, chosen, direct_factors,
)

if not Path(MAIN).exists():
    pytest.skip("the CSI 300 domain config holds licensed Huazheng scores and is not in the repository (.gitignore)", allow_module_level=True)

SLOPE = 25.75e-4                      # the first guess of lambda*, 25.75 bp per SD per year
SMALL = MarketConfig(n_investors=60, n_firms=12, trading_days=400, burn_in_days=120, initial_impact_share=.5,
                     taste_median=.2)
RATED = replace(SMALL, financing_rating_slope=SLOPE)
DOMAIN = replace(load_domain(MAIN), **Q)


def price_term(market):
    exposure = market.financing_exposure
    return market.config.risk_free_rate + market.config.financing_spread * exposure / (exposure + market.prices)


# -- configuration ------------------------------------------------------------------

def test_the_slope_is_omitted_from_identities_at_its_default():
    assert "financing_rating_slope" not in SMALL.to_dict()
    assert RATED.to_dict()["financing_rating_slope"] == SLOPE
    assert MarketConfig.from_dict(RATED.to_dict()) == RATED
    assert MarketConfig.from_dict(SMALL.to_dict()) == replace(SMALL, financing_rating_slope=0.0)
    for bad in (-1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(SMALL, financing_rating_slope=bad)


# -- the quoted rate -----------------------------------------------------------------

class Recording(DisclosureMarket):
    """Records every quote with the signal day and prices it used."""

    def quoted_financing_rate(self):
        rate = super().quoted_financing_rate()
        self.quotes.append((self.signal_day, self.prices.copy(), rate.copy()))
        return rate

    def _prepare_signals(self, day):
        if not hasattr(self, "quotes"):
            self.quotes = []
        super()._prepare_signals(day)


def test_the_quoted_rate_subtracts_the_rating_term_on_the_latest_signal():
    market = Recording(RATED, DOMAIN, PolicySchedule.constant(DOMAIN))
    signal = market.public_signal_history[0]
    rating = SLOPE * (signal - signal.mean())
    np.testing.assert_allclose(market.initial_contract_rate, price_term(market) - rating, rtol=0, atol=1e-16)
    assert market.quotes[0][0] == 0                              # the initial rate carries the day-0 rating term
    result = market.run()
    exposure = market.financing_exposure
    repricings = market.quotes[1:]
    assert [day for day, _, _ in repricings] == list(range(59, RATED.trading_days, 60))
    for day, prices, rate in repricings:
        signal = result.public_signal[day]                       # day d reprices on day d - 1's signal
        expected = RATED.risk_free_rate + RATED.financing_spread * exposure / (exposure + prices) \
            - SLOPE * (signal - signal.mean())
        np.testing.assert_allclose(rate, expected, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(result.financing_rate[day + 1], rate)
        # The rating term averages zero across firms: the rate level is the price term's.
        assert abs((SLOPE * (signal - signal.mean())).mean()) < 1e-17


def test_a_zero_slope_quotes_the_price_term_bit_for_bit():
    market = DisclosureMarket(SMALL, DOMAIN, PolicySchedule.constant(DOMAIN))
    np.testing.assert_array_equal(market.contract_rate, price_term(market))
    np.testing.assert_array_equal(market.financing_rate_history[0], price_term(market))
    zero = DisclosureMarket(replace(SMALL, financing_rating_slope=0.0), DOMAIN, PolicySchedule.constant(DOMAIN))
    assert zero.run_id == market.run_id


# -- the firms' marginal saving --------------------------------------------------------

def test_marginal_saving_adds_the_full_slope(monkeypatch):
    for config in (SMALL, RATED):
        market = DisclosureMarket(config, DOMAIN, PolicySchedule.constant(DOMAIN))
        market.prices = market.prices * np.linspace(.6, 1.4, config.n_firms)
        m = np.linspace(.01, .03, config.n_firms)
        monkeypatch.setattr(market, "marginal_signal_value", lambda: m)
        exposure, prices = market.financing_exposure, market.prices
        rho_price = config.financing_spread * exposure * prices / (exposure + prices)**2 * m
        expected = exposure / prices * (rho_price + config.financing_rating_slope)
        np.testing.assert_allclose(market.marginal_financing_saving(), expected, rtol=1e-14, atol=0)
        if not config.financing_rating_slope:
            np.testing.assert_array_equal(market.marginal_financing_saving(),
                                          exposure / prices * (config.financing_spread * exposure * prices
                                                               / (exposure + prices)**2) * m)


def test_finite_differences_of_the_quote_match_each_channel_and_their_sum(monkeypatch):
    """Contract 26.22 (d): with a positive slope, moving prices by their local response alone changes the fee
    by (E / P) * rho_price, and moving prices and the firm's own signal together (the cross-sectional mean held
    fixed) changes it by (E / P) * (rho_price + slope), the local approximation of marginal_financing_saving."""
    market = DisclosureMarket(RATED, DOMAIN, PolicySchedule.constant(DOMAIN))
    market.prices = market.prices * np.linspace(.6, 1.4, RATED.n_firms)
    m = np.linspace(.01, .03, RATED.n_firms)
    monkeypatch.setattr(market, "marginal_signal_value", lambda: m)
    exposure, prices = market.financing_exposure, market.prices.copy()
    rho_price = RATED.financing_spread * exposure * prices / (exposure + prices)**2 * m
    predicted = market.marginal_financing_saving()
    np.testing.assert_allclose(predicted, exposure / prices * (rho_price + SLOPE), rtol=1e-14, atol=0)
    before = market.quoted_financing_rate()
    epsilon = 1e-5
    market.prices = prices * np.exp(epsilon * m)
    moved = market.quoted_financing_rate()
    np.testing.assert_allclose(exposure * (before - moved) / (prices * epsilon), exposure / prices * rho_price,
                               rtol=1e-6)
    signal = market.public_signal_history[market.signal_day].copy()
    others = RATED.n_firms - 1
    for firm in range(RATED.n_firms):
        shifted = signal - epsilon / others
        shifted[firm] = signal[firm] + epsilon                  # the cross-sectional mean stays put
        market.public_signal_history[market.signal_day] = shifted
        both = market.quoted_financing_rate()[firm]
        assert exposure[firm] * (before[firm] - both) / (prices[firm] * epsilon) \
            == pytest.approx(predicted[firm], rel=1e-6)
    market.public_signal_history[market.signal_day] = signal


# -- the realized saving against the financing factors --------------------------------------

@pytest.mark.parametrize("persistence", [None, PERSISTENCE])
def test_realized_rating_saving_matches_the_real_annuity(persistence):
    """A real change of one SD, landing on its due day, lowers the firm's quoted rate by the slope times
    whatever is left of it at each repricing; with prices and the cross-sectional mean held fixed the
    discounted fee saving per unit of market cap equals (E / P) * slope * the real annuity."""
    market = DisclosureMarket(RATED, DOMAIN, PolicySchedule.constant(DOMAIN))
    firm, change = 3, 1.0
    base = market.public_signal_history[0].copy()
    others = np.arange(RATED.n_firms) != firm
    day, repricing = TIMING["decision_day"], TIMING["repricing_interval"]
    real_due = day + TIMING["implementation_delay"]
    base_rate = market.quoted_financing_rate()[firm]
    saving = 0.
    for t in range(1, TIMING["horizon_days"] + 1):
        quoted = (day + t) // repricing * repricing - 1
        kept = 0. if quoted < real_due else (1. if persistence is None else persistence[quoted - real_due])
        signal = base.copy()
        signal[firm] += change * kept
        signal[others] -= change * kept / others.sum()            # the cross-sectional mean stays put
        market.public_signal_history[market.signal_day] = signal
        rate = market.quoted_financing_rate()[firm]
        saving += market.financing_exposure[firm] * (base_rate - rate) / TRADING_DAYS_PER_YEAR \
            * np.exp(-RATE * t / TRADING_DAYS_PER_YEAR)
    real = financing_signal_factors(np.full(RATED.n_firms, RATE), transparency=TAU, audit_probability=.01,
                                    real_persistence=persistence, **TIMING)[0][firm]
    exposure_share = market.financing_exposure[firm] / market.prices[firm]
    assert saving / market.prices[firm] == pytest.approx(exposure_share * SLOPE * real * change, rel=1e-9)


# -- decisions against brute force ------------------------------------------------------

EXPOSURE_SHARE = np.array([.3, .5, .7, 1.])
RHO_PRICE = np.array([1.5e-4, 2.2e-4, 3.5e-4, 2e-4])


@pytest.mark.parametrize("slope", [SLOPE, 9.675e-4, 30.30e-4])
@pytest.mark.parametrize("audits", ["uniform", "firm_rates"])
def test_decisions_with_the_rating_channel_match_brute_force(slope, audits):
    benefit = EXPOSURE_SHARE * (RHO_PRICE + slope)
    cost = DECISION["marginal_cost"]
    if audits == "uniform":
        rates, arguments = np.full(4, DECISION["regime"].audit_probability), {}
    else:
        rates = np.array([.002, .01, .05, .2])
        arguments = dict(audit_probability=rates)
    choice = npv_choices(**{**DECISION, "benefit": benefit}, real_persistence=PERSISTENCE,
                         inflate_capacity=INFLATE_CAP, **arguments)
    expected = [brute_npv(benefit[i], cost[i], direct_factors(q, persistence=PERSISTENCE), INFLATE_CAP[i])
                for i, q in enumerate(rates)]
    np.testing.assert_allclose(chosen(choice), expected, rtol=1e-6, atol=1e-12)
    assert IMPROVE in set(choice.actions) or INFLATE in set(choice.actions)


# -- markets with the channel ---------------------------------------------------------------

@pytest.mark.parametrize("changes", [None, dict(audit_allocation="value"), ASSURED])
def test_markets_with_the_rating_channel_run(monkeypatch, changes):
    import abm.disclosure.game as game

    calls = []
    original = game.npv_choices

    def recording(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(game, "npv_choices", recording)
    config = replace(RATED, trading_days=700)
    schedule = PolicySchedule.constant(DOMAIN)
    if changes is not None:
        schedule = schedule.with_intervention(121, **changes)
    market = DisclosureMarket(config, DOMAIN, schedule)
    result = market.run()
    assert np.isfinite(result.financing_rate).all() and np.isfinite(result.market.intrinsic_values).all()
    # Every firm's saving per SD is at least the rating term's (E / P) * slope.
    for call in calls:
        assert (call["benefit"] > 0).all()
    plain = DisclosureMarket(replace(config, financing_rating_slope=0.0), DOMAIN, schedule).run()
    assert not np.array_equal(result.financing_rate, plain.financing_rate)
