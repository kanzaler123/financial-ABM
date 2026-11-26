"""Numerical verification of Propositions 3-5 (dynamic assurance & Stackelberg audit theory)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from abm.disclosure.financing import assured_signal_factors, financing_signal_factors

# Default timing and baseline parameters from Study 1-3 (contract §§20-22)
DECISION_DAY = 1260
HORIZON_DAYS = 1260
REPRICING_INTERVAL = 60
PERIOD_DAYS = 60
IMPLEMENTATION_DELAY = 120
DISCLOSURE_DELAY = 20
DISCOUNT_RATE = 0.05
TRANSPARENCY = 0.1
WASH_COST_RATIO = 0.2
BASE_AUDIT = 0.01
ASSURANCE_YEAR_PERIODS = 4

# Study 3 calibrated assurance parameters (configs/disclosure_assurance_confirmatory_v1.json)
ASSURER_EFFORT_COST = 2.172138477935219e-06
ASSURER_LIABILITY_1X = 2.2807454018319797e-05


def _assurance_factors(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.full(a.shape, DISCOUNT_RATE, dtype=np.float64)
    real, _, wash, fine = assured_signal_factors(
        rates,
        decision_day=DECISION_DAY,
        horizon_days=HORIZON_DAYS,
        repricing_interval=REPRICING_INTERVAL,
        period_days=PERIOD_DAYS,
        implementation_delay=IMPLEMENTATION_DELAY,
        disclosure_delay=DISCLOSURE_DELAY,
        audit_probability=BASE_AUDIT,
        transparency=TRANSPARENCY,
        assurance_probability=a,
    )
    return real, wash, fine


def _audit_factors(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.full(q.shape, DISCOUNT_RATE, dtype=np.float64)
    real, _, wash, fine = financing_signal_factors(
        rates,
        decision_day=DECISION_DAY,
        horizon_days=HORIZON_DAYS,
        repricing_interval=REPRICING_INTERVAL,
        period_days=PERIOD_DAYS,
        implementation_delay=IMPLEMENTATION_DELAY,
        disclosure_delay=DISCLOSURE_DELAY,
        audit_probability=q,
        transparency=TRANSPARENCY,
    )
    return real, wash, fine


def _bisect_threshold(fn, lo: float = 0.0, hi: float = 1.0) -> float:
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if fn(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_proposition3_fine_insurance_interval_and_elasticity() -> None:
    """Prop. 3(1): small assurance quality strictly lowers clean firms' size threshold P*(a)."""
    a_star = _bisect_threshold(
        lambda a: (1.0 - TRANSPARENCY) * _assurance_factors(np.array([a]))[1][0]
        - np.sqrt(WASH_COST_RATIO) * _assurance_factors(np.array([a]))[0][0]
    )
    assert a_star == pytest.approx(0.082707, abs=1e-5)

    # Check exact elasticity condition at a = 0
    eps = 1e-6
    real, wash, fine = _assurance_factors(np.array([0.0, eps]))
    d_ln_d = (fine[1] - fine[0]) / (eps * fine[0])
    d_ln_aw2 = 2.0 * (wash[1] - wash[0]) / (eps * wash[0])
    gross_wash = ((1.0 - TRANSPARENCY) * wash[0]) ** 2 / WASH_COST_RATIO
    margin_ratio = gross_wash / (gross_wash - real[0] ** 2)
    assert abs(d_ln_d) > abs(d_ln_aw2) * margin_ratio

    # Verify exact fine-insurance interval (0, a_ins) and global minimum of P*(a) / P*(0)
    grid = np.linspace(0.0, a_star - 1e-4, 2000)
    r_g, w_g, d_g = _assurance_factors(grid)
    denom = ((1.0 - TRANSPARENCY) * w_g) ** 2 / WASH_COST_RATIO - r_g**2
    ratio = (d_g / denom) / (d_g[0] / denom[0])
    assert ratio[1] < 1.0
    min_idx = int(np.argmin(ratio))
    assert grid[min_idx] == pytest.approx(0.0156, abs=5e-4)
    assert ratio[min_idx] == pytest.approx(0.8728, abs=5e-4)
    ins_upper = grid[np.where(ratio < 1.0)[0][-1]]
    assert ins_upper == pytest.approx(0.0346, abs=5e-4)

    # 1x statutory liability slope stays strictly below a*
    lambda_1x = ASSURER_LIABILITY_1X * BASE_AUDIT / (2.0 * ASSURER_EFFORT_COST)
    assert lambda_1x < a_star


def test_proposition3_no_zero_inflation_fixed_point_and_exact_high_liability_3cycle() -> None:
    """Prop. 3(2)-(3): no deterrence fixed point; exact 3-cycle at high lambda_L; non-monotonicity."""
    a_star = _bisect_threshold(
        lambda a: (1.0 - TRANSPARENCY) * _assurance_factors(np.array([a]))[1][0]
        - np.sqrt(WASH_COST_RATIO) * _assurance_factors(np.array([a]))[0][0]
    )
    # Part (2): verify geometric stock contraction and finite-time relapse for any finite lambda_L
    for lam in (0.5, 0.84, 5.25, 50.0):
        u = 1.0
        a = min(1.0, lam * u)
        relapsed = False
        for _ in range(200):
            if a < a_star:
                relapsed = True
                break
            rho = (1.0 - a) * (1.0 - BASE_AUDIT)
            phi_m = (1.0 - rho**ASSURANCE_YEAR_PERIODS) / (ASSURANCE_YEAR_PERIODS * (1.0 - rho))
            s_y = phi_m * u
            u = (rho**ASSURANCE_YEAR_PERIODS) * u
            a = min(1.0, lam * s_y)
        assert relapsed, f"Expected finite-time relapse for lambda_L={lam}"

    # Part (3): exact 3-cycle when lambda_L >= max(1, m * a* / (1 - qbar)) / theta(0)
    theta0 = 0.19
    assert ASSURANCE_YEAR_PERIODS * a_star / (1.0 - BASE_AUDIT) < 1.0
    lam_3c = 1.0 / theta0

    def step_homogeneous(u: float, a: float, lam: float) -> tuple[float, float, float]:
        if a >= a_star:
            rho = (1.0 - a) * (1.0 - BASE_AUDIT)
            phi_m = (1.0 - rho**ASSURANCE_YEAR_PERIODS) / (ASSURANCE_YEAR_PERIODS * (1.0 - rho))
            return (rho**ASSURANCE_YEAR_PERIODS) * u, min(1.0, lam * phi_m * u), 0.0
        # Relapse year at a = 0 (homogeneous cohort of size theta0):
        # Every period has theta0 inflating; end-of-year stock is (1 - qbar) * theta0
        return (1.0 - BASE_AUDIT) * theta0, min(1.0, lam * theta0), theta0

    for lam in (lam_3c + 0.05, 10.0, 100.0):
        u, a = 0.0, 0.0
        seq_a = []
        seq_inf = []
        for _ in range(9):
            seq_a.append(a)
            u, a, inf_y = step_homogeneous(u, a, lam)
            seq_inf.append(inf_y)
        expected_a2 = min(1.0, lam * (1.0 - BASE_AUDIT) * theta0 / ASSURANCE_YEAR_PERIODS)
        assert seq_a[0] == pytest.approx(0.0)
        assert seq_a[1] == pytest.approx(1.0)
        assert seq_a[2] == pytest.approx(expected_a2)
        assert seq_a[3] == pytest.approx(0.0)
        assert seq_a[4] == pytest.approx(1.0)
        assert seq_a[5] == pytest.approx(expected_a2)
        assert np.mean(seq_inf) == pytest.approx(theta0 / 3.0)

    # Verify strict non-monotonicity in heterogeneous mean-field population
    r0, w0, d0 = _assurance_factors(np.array([0.0]))
    base_denom = ((1.0 - TRANSPARENCY) * w0[0]) ** 2 / WASH_COST_RATIO - r0[0] ** 2
    sigma = 1.04
    mu = sigma * norm.ppf(0.19)
    z = mu + sigma * norm.ppf((np.arange(500) + 0.5) / 500.0)

    def long_run_inflation(mult: float) -> float:
        lam = mult * ASSURER_LIABILITY_1X * BASE_AUDIT / (2.0 * ASSURER_EFFORT_COST)
        u_vec = np.full(500, 0.25)
        a_cur = min(1.0, lam * float(u_vec.mean()))
        rates = []
        for year in range(60):
            rr, ww, dd = _assurance_factors(np.array([a_cur]))
            denom = ((1.0 - TRANSPARENCY) * ww[0]) ** 2 / WASH_COST_RATIO - rr[0] ** 2
            if denom <= 0.0:
                initiates = np.zeros(500, dtype=bool)
                continues = False
            else:
                ratio = (dd[0] / denom) / (d0[0] / base_denom)
                initiates = z > np.log(ratio)
                continues = True
            rho = (1.0 - a_cur) * (1.0 - BASE_AUDIT)
            s_p = []
            inf_p = []
            for _ in range(ASSURANCE_YEAR_PERIODS):
                inf_p.append(float(np.mean((u_vec if continues else 0.0) + (1.0 - u_vec) * initiates)))
                u_pre = np.where(initiates, 1.0, u_vec)
                s_p.append(float(np.mean(u_pre)))
                u_vec = rho * u_pre
            if year >= 20:
                rates.append(float(np.mean(inf_p)))
            a_cur = min(1.0, lam * float(np.mean(s_p)))
        return float(np.mean(rates))

    inf_25x = long_run_inflation(25.0)
    inf_100x = long_run_inflation(100.0)
    assert inf_25x < inf_100x
    assert inf_25x < theta0 / 3.0


def test_proposition4_sorting_jensen_gap_and_escape_cycle() -> None:
    """Prop. 4: laxest-provider concentration, client-weighted quality gap, and annual rotation."""
    a_star = 0.082707
    a_ins = 0.0346
    # Verify strict monotonicity of V_W(a) for open-case firms on [0, 1] and V*(a) non-increasing on [a_ins, 1]
    a_grid = np.linspace(0.0, 1.0, 401)
    r_g, w_g, d_g = _assurance_factors(a_grid)
    # Realistic post-burn-in large/mid firm scales (B ~ 4.5e-4, c ~ 0.013, f = F/P ~ 1.5e-5)
    B, c, f = 4.5e-4, 0.013, 1.5e-5
    vi = (B * r_g) ** 2 / (4.0 * c)
    vw_open = (B * (1.0 - TRANSPARENCY) * w_g) ** 2 / (4.0 * WASH_COST_RATIO * c)
    assert np.all(np.diff(vw_open) < 0.0)
    vw_clean = vw_open - f * d_g
    v_star_clean = np.maximum(vi, vw_clean)
    above_ins = a_grid >= a_ins
    # V*(a) is non-increasing on [a_ins, 1] and strictly decreasing wherever clean firm inflates
    assert np.all(np.diff(v_star_clean[above_ins]) <= 1e-18)
    active_clean = above_ins & (vw_clean > vi)
    assert active_clean.sum() >= 5
    assert np.all(np.diff(vw_clean[active_clean]) < 0.0)

    # Verify client-weighted quality < unweighted provider quality and escape-cycle rotation
    M = 4
    theta = 0.25
    lam = 16.0 * ASSURER_LIABILITY_1X * BASE_AUDIT / (2.0 * ASSURER_EFFORT_COST)  # 0.84
    assert lam * (theta / ((1.0 - theta) / M + theta)) > a_star

    # Simulate deterministic M=4 provider escape cycle starting from one lax provider (k=0)
    qualities = np.array([0.0, 0.20, 0.20, 0.20])
    lax_history = []
    for _ in range(8):
        lax = int(np.argmin(qualities))
        lax_history.append(lax)
        weights = np.full(M, (1.0 - theta) / M)
        weights[lax] += theta
        client_quality = float(np.dot(weights, qualities))
        unweighted_quality = float(np.mean(qualities))
        assert client_quality < unweighted_quality
        # Lax provider receives all inflating firms; strict providers (with 0 old stock) see 0
        shares = np.zeros(M)
        shares[lax] = theta / weights[lax]
        next_q = np.minimum(1.0, lam * shares)
        assert next_q[lax] >= a_star
        assert np.all(next_q[np.arange(M) != lax] == 0.0)
        qualities = next_q
    # Strict rotation: laxest provider never stays the same in consecutive years
    assert all(a != b for a, b in zip(lax_history, lax_history[1:]))


def test_proposition5_concavity_value_targeting_failure_and_knapsack_optimality() -> None:
    """Prop. 5: D(q) strict concavity, q_det monotonicity in P, V failure, and knapsack optimum."""
    # Part (1): exact derivative and strict concavity of D(q)
    beta = np.exp(-DISCOUNT_RATE * PERIOD_DAYS / 252.0)
    qs = np.linspace(0.0, 1.0, 501)
    r_g, w_g, d_g = _audit_factors(qs)
    d_closed = beta * qs / ((1.0 - beta) + beta * qs)
    assert np.allclose(d_g, d_closed, rtol=1e-12, atol=1e-12)
    # Second differences strictly negative everywhere on [0, 1]
    assert np.all(np.diff(d_g, 2) < 0.0)
    # Marginal deterrence ratio D'(0) / D'(0.08) > 50
    dprime_0 = beta * (1.0 - beta) / (1.0 - beta) ** 2
    dprime_08 = beta * (1.0 - beta) / ((1.0 - beta) + beta * 0.08) ** 2
    assert dprime_0 / dprime_08 > 50.0

    # q* for open-case firms
    q_star = _bisect_threshold(
        lambda q: (1.0 - TRANSPARENCY) * _audit_factors(np.array([q]))[1][0]
        - np.sqrt(WASH_COST_RATIO) * _audit_factors(np.array([q]))[0][0]
    )
    assert q_star == pytest.approx(0.104924, abs=1e-5)

    # Firm-level q_det(P) strictly increasing in P and bounded by q*
    # Realistic post-burn-in model scales (day 1500: B ~ 4.4e-4, c ~ 0.013, F = 4.56e-5)
    B, c, F = 4.4e-4, 0.013, 4.56e-5

    def q_det_for_price(P: float, pi: float = 1.0) -> float:
        def surplus(q: float) -> float:
            rr, ww, dd = _audit_factors(np.array([q]))
            return (B**2 / (4.0 * c)) * (((1.0 - TRANSPARENCY) * ww[0]) ** 2 / WASH_COST_RATIO - rr[0] ** 2) - pi * (F / P) * dd[0]

        if surplus(0.0) <= 0.0:
            return 0.0
        return _bisect_threshold(surplus, 0.0, q_star)

    prices = np.geomspace(0.15, 20.0, 25)
    q_dets = np.array([q_det_for_price(float(p)) for p in prices])
    assert np.all(q_dets < q_star)
    assert np.all(np.diff(q_dets) > 0.0)
    assert q_det_for_price(10.0, pi=0.0) == pytest.approx(q_star, abs=1e-6)

    # Part (2)-(3): heterogeneous fleet (with pre-policy large/mid open cases + clean small firms)
    # where Value targeting V loses to Uniform B, while Stackelberg ascending-q_det knapsack
    # beats both B and V at identical budget K = N * qbar.
    n_firms = 200
    rng = np.random.default_rng(12345)
    firm_prices = np.sort(np.exp(rng.normal(0.0, 0.915, size=n_firms)))
    firm_prices /= firm_prices.mean()
    # Top 30% of firms carry pre-policy open cases (pi = 0), bottom 70% start clean (pi = 1)
    firm_pi = np.where(np.arange(n_firms) >= int(0.7 * n_firms), 0.0, 1.0)
    firm_q_det = np.array([q_det_for_price(float(p), float(pi)) for p, pi in zip(firm_prices, firm_pi)])
    budget = n_firms * BASE_AUDIT  # K = 2.0

    q_uniform = np.full(n_firms, BASE_AUDIT)
    q_value = budget * firm_prices / firm_prices.sum()

    # Exact Stackelberg equal-weight knapsack: sort positive q_det ascending
    q_stack = np.zeros(n_firms)
    rem = budget
    for idx in np.argsort(firm_q_det):
        need = firm_q_det[idx]
        if need <= 0.0:
            continue
        if need <= rem:
            q_stack[idx] = need
            rem -= need
        else:
            break

    deterred_uniform = int(np.sum(q_uniform >= firm_q_det))
    deterred_value = int(np.sum(q_value >= firm_q_det))
    deterred_stack = int(np.sum(q_stack >= firm_q_det))
    assert deterred_value < deterred_uniform < deterred_stack
    assert np.sum(q_stack) <= budget + 1e-12


def test_proposition3_flip_bifurcation_and_trapping_band_nonmonotonicity() -> None:
    """Prop. 3(3a)-(3c): interior FP loses stability via flip bifurcation; 25x traps relapse quality > a_ins."""
    a_star = 0.082707
    a_ins = 0.0346
    r0, w0, d0 = _assurance_factors(np.array([0.0]))
    base_denom = ((1.0 - TRANSPARENCY) * w0[0]) ** 2 / WASH_COST_RATIO - r0[0] ** 2
    sigma = 1.04
    theta0 = 0.19
    mu = sigma * norm.ppf(theta0)

    def theta_of(a: float) -> float:
        if a >= a_star:
            return 0.0
        rr, ww, dd = _assurance_factors(np.array([a]))
        denom = ((1.0 - TRANSPARENCY) * ww[0]) ** 2 / WASH_COST_RATIO - rr[0] ** 2
        if denom <= 0.0:
            return 0.0
        ratio = (dd[0] / denom) / (d0[0] / base_denom)
        return float(1.0 - norm.cdf((np.log(ratio) - mu) / sigma))

    # Flip-bifurcation elasticity |a * theta'(a) / theta(a)| > 1 for all interior targets a_tilde >= 0.045
    eps = 1e-5
    for a_tilde in (0.045, 0.055, 0.065, 0.075):
        th = theta_of(a_tilde)
        dth = (theta_of(a_tilde + eps) - theta_of(a_tilde - eps)) / (2.0 * eps)
        elast = abs(a_tilde * dth / th)
        assert elast > 1.0

    # Global trapping-band check at 25x: relapse years stay strictly inside (a_ins, a*) — never crash to a=0!
    z = mu + sigma * norm.ppf((np.arange(500) + 0.5) / 500.0)
    lam_25 = 25.0 * ASSURER_LIABILITY_1X * BASE_AUDIT / (2.0 * ASSURER_EFFORT_COST)
    u_vec = np.zeros(500)
    a_cur = 0.0
    relapse_qualities = []
    for year in range(40):
        rr, ww, dd = _assurance_factors(np.array([a_cur]))
        denom = ((1.0 - TRANSPARENCY) * ww[0]) ** 2 / WASH_COST_RATIO - rr[0] ** 2
        if denom <= 0.0:
            init = np.zeros(500, dtype=bool)
            cont = False
        else:
            ratio = (dd[0] / denom) / (d0[0] / base_denom)
            init = z > np.log(ratio)
            cont = True
            if year >= 3:
                relapse_qualities.append(a_cur)
        rho = (1.0 - a_cur) * (1.0 - BASE_AUDIT)
        s_p = []
        for _ in range(ASSURANCE_YEAR_PERIODS):
            u_pre = np.where(init, 1.0, u_vec) if cont else u_vec
            s_p.append(float(np.mean(u_pre)))
            u_vec = rho * u_pre
        a_cur = min(1.0, lam_25 * float(np.mean(s_p)))
    assert len(relapse_qualities) >= 8
    assert min(relapse_qualities) > a_ins
    assert max(relapse_qualities) < a_star


def test_proposition6_monitoring_blindness_and_hysteresis_remedy() -> None:
    """Prop. 6: wrong-way Bayesian inertia ratio; runtime smoothing restores monotonicity and seals escape hatch."""
    a_star = 0.082707
    # Part (1)-(2): exact wrong-way inertia ratio gamma_eff(0) / gamma_eff(1)
    n_k, nu = 200.0, 2.0
    g_lax = nu / (nu + n_k * BASE_AUDIT)
    g_strict = nu / (nu + n_k * 1.0)
    assert g_lax == pytest.approx(0.5)
    assert g_lax / g_strict == pytest.approx((nu + n_k) / (nu + n_k * BASE_AUDIT))
    assert g_lax / g_strict > 50.0

    # Part (3): exact consecutive zero-inflation years lower bound T(gamma; a_peak)
    gamma = 0.7
    a_peak = 1.0
    t_bound = int(np.floor(np.log(a_peak / a_star) / np.log(1.0 / gamma))) + 1
    assert t_bound == 7
    assert (gamma ** (t_bound - 1)) * a_peak >= a_star
    assert (gamma ** t_bound) * a_peak < a_star

    # Verify smoothing gamma=0.7 restores strict monotonicity across 4x..100x in heterogeneous mean field
    r0, w0, d0 = _assurance_factors(np.array([0.0]))
    base_denom = ((1.0 - TRANSPARENCY) * w0[0]) ** 2 / WASH_COST_RATIO - r0[0] ** 2
    sigma, theta0 = 1.04, 0.19
    mu = sigma * norm.ppf(theta0)
    z = mu + sigma * norm.ppf((np.arange(500) + 0.5) / 500.0)

    def long_run_rate(mult: float, smooth: float) -> float:
        lam = mult * ASSURER_LIABILITY_1X * BASE_AUDIT / (2.0 * ASSURER_EFFORT_COST)
        u_vec = np.zeros(500)
        a_cur = 0.0
        rates = []
        for year in range(100):
            rr, ww, dd = _assurance_factors(np.array([a_cur]))
            denom = ((1.0 - TRANSPARENCY) * ww[0]) ** 2 / WASH_COST_RATIO - rr[0] ** 2
            if denom <= 0.0:
                init = np.zeros(500, dtype=bool)
                cont = False
            else:
                ratio = (dd[0] / denom) / (d0[0] / base_denom)
                init = z > np.log(ratio)
                cont = True
            rho = (1.0 - a_cur) * (1.0 - BASE_AUDIT)
            s_p, inf_p = [], []
            for _ in range(ASSURANCE_YEAR_PERIODS):
                inf_p.append(float(np.mean((u_vec if cont else 0.0) + (1.0 - u_vec) * init)))
                u_pre = np.where(init, 1.0, u_vec) if cont else u_vec
                s_p.append(float(np.mean(u_pre)))
                u_vec = rho * u_pre
            if year >= 30:
                rates.append(float(np.mean(inf_p)))
            target = min(1.0, lam * float(np.mean(s_p)))
            a_cur = smooth * a_cur + (1.0 - smooth) * target
        return float(np.mean(rates))

    smooth_rates = [long_run_rate(m, 0.7) for m in (4.0, 10.0, 16.0, 25.0, 50.0, 100.0)]
    assert all(a > b for a, b in zip(smooth_rates, smooth_rates[1:]))
    assert smooth_rates[-1] < 0.02  # drops from >6% myopic 3-cycle down to ~1.1%!
