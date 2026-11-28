"""Regression tests for marginal_signal_value equivalence, normal CDF, and G-band solver edges."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure.bayesian_audit import BayesianAuditFilter, _solve_menu_constrained_bands
from abm.disclosure.config import TRADING_DAYS_PER_YEAR, MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.harness import _cap_buys_to_cash
from abm.disclosure.policy import PolicyRegime, PolicySchedule, _normal_cdf

ESG = load_domain("configs/domains/esg.json")


def _reference_marginal_signal_value(market: DisclosureMarket) -> np.ndarray:
    """Exact full-matrix finite-difference reference matching the pre-optimization loop."""
    cfg, investors = market.config, market.investors
    if cfg.rebalance_speed == 0:
        return np.zeros(cfg.n_firms)
    who = np.arange(investors.size)
    signal = market.attribute_signal()
    expected = market.expected_returns()
    arguments = dict(
        prices=market.prices,
        expected_returns=expected,
        annual_market_covariance=market.market_covariance * TRADING_DAYS_PER_YEAR,
        market_expected_return=float(market.prices @ expected / market.prices.sum()),
        market_annual_variance=market.allocation_market_variance * TRADING_DAYS_PER_YEAR,
        config=cfg,
    )
    baseline = investors.desired_orders(who, attribute=signal, **arguments)
    target_shares = np.maximum(baseline / cfg.rebalance_speed + investors.holdings, 0.0)
    target_values = target_shares * market.prices
    target_equity = target_values.sum(axis=1, keepdims=True)
    weights = np.divide(target_values, target_equity, out=np.zeros_like(target_values), where=target_equity > 0)
    baseline = cfg.rebalance_speed * (target_shares - investors.holdings)
    baseline = np.maximum(baseline, -investors.holdings)
    baseline = _cap_buys_to_cash(baseline, investors.cash, market.prices, cfg.transaction_cost_rate)
    response = np.empty(cfg.n_firms)
    increment = 1e-4
    relative_tilt = np.expm1(increment * investors.choice_intensity * investors.taste)
    for firm in range(cfg.n_firms):
        targets = target_shares / (1.0 + weights[:, firm] * relative_tilt)[:, None]
        targets[:, firm] *= 1.0 + relative_tilt
        orders = cfg.rebalance_speed * (targets - investors.holdings)
        orders = np.maximum(orders, -investors.holdings)
        orders = _cap_buys_to_cash(orders, investors.cash, market.prices, cfg.transaction_cost_rate)
        response[firm] = (orders[:, firm] - baseline[:, firm]).sum() / increment
    return np.maximum(response, 0.0) * cfg.price_impact * cfg.permanent_impact_fraction / cfg.liquidity_fraction


@pytest.mark.parametrize(
    ("taste_median", "cash_scale"),
    [
        (0.0, 1.0),
        (0.0724, 1.0),
        (0.22, 1.0),
        (0.15, 0.01),
        (0.15, 0.0),
    ],
)
def test_marginal_signal_value_matches_full_matrix_reference(taste_median: float, cash_scale: float) -> None:
    """Fast vectorized path + tight-cash fallback match the full-matrix loop to machine precision."""
    cfg = MarketConfig(
        n_investors=120,
        n_firms=25,
        trading_days=120,
        burn_in_days=60,
        initial_impact_share=0.3,
        taste_median=taste_median,
        seed=9101,
    )
    market = DisclosureMarket(cfg, ESG, PolicySchedule.constant(ESG))
    for day_idx in range(60):
        market.step(day_idx)
    if cash_scale != 1.0:
        market.investors.cash *= cash_scale
    expected = _reference_marginal_signal_value(market)
    actual = market.marginal_signal_value()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


def test_normal_cdf_matches_scipy_ndtr() -> None:
    """Standard-library normal CDF helper agrees with scipy.special.ndtr across extreme tails."""
    scipy_special = pytest.importorskip("scipy.special")
    grid = np.linspace(-8.0, 8.0, 161)
    np.testing.assert_allclose(_normal_cdf(grid), scipy_special.ndtr(grid), rtol=1e-13, atol=1e-15)


def test_menu_constrained_bands_rejects_infeasible_two_band_erosion_and_conserves_budget() -> None:
    """Two-band and multi-band menu solvers strictly conserve budget without eroding floor bands."""
    q_grid = np.array([0.0, 0.005, 0.009, 0.02, 0.05, 0.10, 0.25, 1.0], dtype=np.float64)
    n = 40
    budget = 0.40  # q_bar = 0.01; menu [0.009, 1.0] has rem = 0.04 < dq = 0.991 and must not erode band 0
    # Step payoffs mimicking discrete deterrence thresholds
    thresholds = np.linspace(0.004, 0.12, n)[:, None]
    expected_u = (q_grid[None, :] >= thresholds).astype(np.float64) + 0.05 * q_grid[None, :]
    q_raw = np.full(n, budget / n, dtype=np.float64)

    for k_bands in (2, 3, 5):
        prob, labels = _solve_menu_constrained_bands(q_grid, expected_u, budget, k_bands, q_raw)
        assert prob.shape == (n,)
        assert labels.shape == (n,)
        assert np.all((prob >= 0.0) & (prob <= 1.0))
        assert prob.sum() == pytest.approx(budget, abs=1e-11)
        assert len(np.unique(np.round(prob, 12))) <= k_bands


def test_bayesian_audit_filter_switches_mode_across_bayesian_regimes() -> None:
    """Switching from stackelberg_obs to stackelberg_obs_val or stackelberg_obs_sig takes effect dynamically."""
    dom = replace(ESG, baseline_fixed_fine=4.56e-5, baseline_penalty=0.0, natural_decay=0.2)
    cfg = MarketConfig(
        n_investors=40,
        n_firms=16,
        trading_days=240,
        burn_in_days=60,
        initial_impact_share=0.2,
        taste_median=0.0724,
        seed=9102,
    )
    base_reg = replace(PolicyRegime.baseline(dom), audit_allocation="stackelberg_obs")
    val_reg = replace(base_reg, audit_allocation="stackelberg_obs_val")
    sched = PolicySchedule(baseline=base_reg, intervention=val_reg, start_day=121)
    market = DisclosureMarket(cfg, dom, sched)
    assert isinstance(market.bayesian_audit_filter, BayesianAuditFilter)
    res = market.run()
    assert res.audit_probability is not None
    budget = cfg.n_firms * dom.baseline_audit_probability
    for period in range(1, res.audit_probability.shape[0]):
        assert res.audit_probability[period].sum() == pytest.approx(budget, abs=1e-11)
