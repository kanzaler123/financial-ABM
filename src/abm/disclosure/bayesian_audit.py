"""Joint Type-State Bayesian Dynamic Stackelberg Auditor (zero private-state leakage).

Formulates multi-period budget-constrained audit allocation under persistent unobserved firm cost
types and unobserved open-case states as a Weakly-Coupled POMDP / Restless Multi-Armed Bandit
(RMAB) Stackelberg Audit Game.

Strictly public information set used by ``BayesianAuditFilter``:
- Public market capitalizations ``P_i`` and public size tiers;
- Public financing-rate sensitivity ``B_i`` (computed from public market signals and investor demand);
- Publicly claimed attribute reports ``r_i = min(g_i + x_i, C)`` (the same observable used by jump targeting);
- Known structural domain/policy parameters (cost prior distribution, decay, delays, fine/transparency);
- The regulator's own past audit selections and outcomes (``selected``, ``caught``), plus the exact
  post-audit certified score of audited firms.

Zero access to private firm cost shocks (``cost_scale``), latent true attribute (``goodness``), latent
inflation stock (``inflation``), or private wash-cost realizations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import TRADING_DAYS_PER_YEAR
from .domain import DomainSpec
from .policy import PolicyRegime, _distribute_budget_proportionally, _upper_concave_hull_segments

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class _PlanSnapshot:
    decision_day: int
    prices: FloatArray
    benefit: FloatArray
    planned_q: FloatArray
    regime: PolicyRegime
    report_ref: FloatArray
    g_dirty_ref: FloatArray


def _timing_factors(
    *,
    discount_rate: float,
    decision_day: int,
    horizon_days: int,
    repricing_interval: int,
    period_days: int,
    implementation_delay: int,
    disclosure_delay: int,
    real_persistence: FloatArray | None,
) -> tuple[float, FloatArray, FloatArray, float, float]:
    """Return ``(a_real, wash_weights, audit_counts, beta_1, beta)`` for the decision calendar."""
    times = np.arange(1, horizon_days + 1)
    calendar = decision_day + times
    quoted_day = (calendar // repricing_interval) * repricing_interval - 1
    real_due = decision_day + max(1, implementation_delay)
    wash_due = decision_day + max(1, disclosure_delay)
    real_visible = quoted_day >= real_due
    wash_visible = quoted_day >= wash_due
    first_audit = ((wash_due + period_days - 1) // period_days) * period_days
    audits = np.maximum(0, (quoted_day - 1 - first_audit) // period_days + 1)
    discount = np.exp(-times * discount_rate / TRADING_DAYS_PER_YEAR)

    if real_persistence is None:
        a_real = float((real_visible @ discount) / TRADING_DAYS_PER_YEAR)
    else:
        days_since = np.clip(quoted_day - real_due, 0, None)
        a_real = float(((real_visible * real_persistence[days_since]) @ discount) / TRADING_DAYS_PER_YEAR)

    wash_weights = np.bincount(audits, weights=(wash_visible * discount) / TRADING_DAYS_PER_YEAR)
    audit_counts = np.arange(wash_weights.size, dtype=np.float64)
    beta_1 = float(np.exp(-discount_rate * (first_audit - decision_day) / TRADING_DAYS_PER_YEAR))
    beta = float(np.exp(-discount_rate * period_days / TRADING_DAYS_PER_YEAR))
    return a_real, wash_weights, audit_counts, beta_1, beta


def _solve_zero_fine_threshold(
    *,
    a_real: float,
    wash_weights: FloatArray,
    audit_counts: FloatArray,
    beta_1: float,
    beta: float,
    tau: float,
    penalty: float,
    subsidy: float,
    wash_cost_ratio: float,
    max_benefit: float,
) -> float:
    """Upper-bound deterrence threshold ``q*`` when ``fixed_fine == 0`` or ``pi == 0``."""
    if max_benefit <= 0.0 and subsidy <= 0.0:
        return 0.0

    def positive_wash_margin(q: float) -> bool:
        aw = float(np.power(1.0 - q, audit_counts) @ wash_weights)
        d_q = 0.0 if q <= 0.0 else (beta_1 * q) / (1.0 - (1.0 - q) * beta)
        net_w = max(0.0, max_benefit * (1.0 - tau) * aw - penalty * d_q)
        m_r = max_benefit * a_real + subsidy
        return net_w > np.sqrt(wash_cost_ratio) * m_r

    if not positive_wash_margin(0.0):
        return 0.0
    if positive_wash_margin(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if positive_wash_margin(mid):
            lo = mid
        else:
            hi = mid
    return min(1.0, hi + 1e-5)


def _evaluate_type_choices(
    *,
    c_mat: FloatArray,
    benefit: FloatArray,
    f_frac: FloatArray,
    pi: float,
    q_vec: FloatArray,
    a_real: float,
    wash_weights: FloatArray,
    audit_counts: FloatArray,
    beta_1: float,
    beta: float,
    regime: PolicyRegime,
    wash_cost_ratio: float,
    cap: float,
) -> tuple[BoolArray, FloatArray, FloatArray]:
    """Return ``(deterred, step_real, step_wash)`` of shape ``(N, K)`` for firm-specific ``q_vec``."""
    b_col = np.maximum(benefit, 0.0)[:, None]
    f_col = f_frac[:, None]
    q_col = np.clip(q_vec, 0.0, 1.0)
    aw = (np.power(1.0 - q_col[:, None], audit_counts[None, :]) @ wash_weights)[:, None]
    denom = 1.0 - (1.0 - q_col) * beta
    d_q = np.divide(beta_1 * q_col, denom, out=np.zeros_like(q_col), where=q_col > 0.0)[:, None]

    m_real = b_col * a_real + regime.subsidy
    step_real = np.clip(m_real / (2.0 * c_mat), 0.0, cap)
    vi = np.maximum(0.0, m_real * step_real - c_mat * step_real * step_real)

    net_wash = np.maximum(b_col * (1.0 - regime.transparency) * aw - regime.penalty * d_q, 0.0)
    step_wash = np.clip(net_wash / (2.0 * wash_cost_ratio * c_mat), 0.0, cap)
    vw = net_wash * step_wash - wash_cost_ratio * c_mat * step_wash * step_wash - pi * f_col * d_q
    deterred = vw <= vi + 1e-15
    return deterred, step_real, step_wash


class BayesianAuditFilter:
    """Maintains persistent-type x Markov-state posterior beliefs purely from public observables."""

    def __init__(
        self,
        *,
        domain: DomainSpec,
        size_class: IntArray,
        initial_report: FloatArray,
        report_bound: float,
        discount_rate: float,
        horizon_days: int,
        repricing_interval: int,
        period_days: int,
        real_persistence: FloatArray | None = None,
        use_report_signal: bool = False,
        value_weighted: bool = False,
        quadrature_points: int = 25,
        grid_points: int = 64,
        slip_rate: float = 0.005,
        dirty_clear_slip: float = 0.002,
    ) -> None:
        self.domain = domain
        self.size_class = np.asarray(size_class, dtype=np.int64).copy()
        self.n_firms = self.size_class.size
        self.report_bound = float(report_bound)
        self.discount_rate = float(discount_rate)
        self.horizon_days = int(horizon_days)
        self.repricing_interval = int(repricing_interval)
        self.period_days = int(period_days)
        self.real_persistence = real_persistence
        self.use_report_signal = bool(use_report_signal)
        self.value_weighted = bool(value_weighted)
        self.grid_points = int(grid_points)
        self.slip_rate = float(slip_rate)
        self.dirty_clear_slip = float(dirty_clear_slip)

        multipliers = np.asarray(domain.size_cost_multipliers, dtype=np.float64)
        self.tier_base_scale = domain.real_cost_scale * multipliers[self.size_class]

        sigma = float(domain.real_cost_log_sd)
        if sigma > 1e-6 and quadrature_points > 1:
            gh_x, gh_w = np.polynomial.hermite.hermgauss(quadrature_points)
            self.prior_w = (gh_w / np.sqrt(np.pi)).astype(np.float64)
            self.eps_nodes = np.exp(np.sqrt(2.0) * sigma * gh_x - 0.5 * sigma * sigma).astype(np.float64)
        else:
            self.prior_w = np.array([1.0], dtype=np.float64)
            self.eps_nodes = np.array([1.0], dtype=np.float64)
        self.k_nodes = self.prior_w.size

        init_rep = np.asarray(initial_report, dtype=np.float64).copy()
        self.w_clean = np.tile(self.prior_w[None, :], (self.n_firms, 1))
        self.w_dirty = np.zeros((self.n_firms, self.k_nodes), dtype=np.float64)
        self.g_dirty_est = init_rep.copy()
        self.last_post_audit_report = init_rep.copy()
        self.last_plan: _PlanSnapshot | None = None
        self._predicted_day: int = -1

    @property
    def p_clean(self) -> FloatArray:
        return self.w_clean.sum(axis=1)

    @property
    def p_dirty(self) -> FloatArray:
        return self.w_dirty.sum(axis=1)

    def _cost_matrix(self, g_level: FloatArray) -> FloatArray:
        base = self.tier_base_scale * np.exp(self.domain.cost_level_elasticity * g_level)
        return base[:, None] * self.eps_nodes[None, :]

    def _decay_g_estimate(self, g_est: FloatArray, upper_bound: FloatArray) -> FloatArray:
        low, _ = self.domain.goodness_bounds
        decay = self.domain.natural_decay * self.period_days / TRADING_DAYS_PER_YEAR
        decayed = np.maximum(g_est - decay, low)
        reversion = getattr(self.domain, "natural_reversion", 0.0) or 0.0
        if reversion > 0.0:
            retention = float(np.exp(-reversion * self.period_days / TRADING_DAYS_PER_YEAR))
            mean_g = float(decayed.mean())
            decayed = mean_g + retention * (decayed - mean_g)
        return np.minimum(decayed, upper_bound)

    def record_commitment(
        self,
        *,
        day: int,
        prices: FloatArray,
        benefit: FloatArray,
        planned_q: FloatArray,
        regime: PolicyRegime,
        post_audit_report: FloatArray,
    ) -> None:
        """Store public market context and committed audit vector for next-period transition."""
        self.last_post_audit_report = np.asarray(post_audit_report, dtype=np.float64).copy()
        self.last_plan = _PlanSnapshot(
            decision_day=int(day),
            prices=np.asarray(prices, dtype=np.float64).copy(),
            benefit=np.asarray(benefit, dtype=np.float64).copy(),
            planned_q=np.asarray(planned_q, dtype=np.float64).copy(),
            regime=regime,
            report_ref=self.last_post_audit_report.copy(),
            g_dirty_ref=self.g_dirty_est.copy(),
        )

    def predict(self, *, day: int, pre_audit_report: FloatArray) -> None:
        """Advance persistent-type x open-case prior across the preceding firm decision (before audit)."""
        if self._predicted_day == day or self.last_plan is None:
            return
        pre_rep = np.asarray(pre_audit_report, dtype=np.float64)
        snap = self.last_plan
        disclosure_delay = (
            self.domain.disclosure_delay_days
            if snap.regime.disclosure_delay_days is None
            else snap.regime.disclosure_delay_days
        )
        a_real, wash_weights, audit_counts, beta_1, beta = _timing_factors(
            discount_rate=self.discount_rate,
            decision_day=snap.decision_day,
            horizon_days=self.horizon_days,
            repricing_interval=self.repricing_interval,
            period_days=self.period_days,
            implementation_delay=self.domain.implementation_delay_days,
            disclosure_delay=disclosure_delay,
            real_persistence=self.real_persistence,
        )
        c_clean = self._cost_matrix(snap.report_ref)
        f_frac = snap.regime.fixed_fine / snap.prices
        d_clean, step_real_clean, _ = _evaluate_type_choices(
            c_mat=c_clean,
            benefit=snap.benefit,
            f_frac=f_frac,
            pi=1.0,
            q_vec=snap.planned_q,
            a_real=a_real,
            wash_weights=wash_weights,
            audit_counts=audit_counts,
            beta_1=beta_1,
            beta=beta,
            regime=snap.regime,
            wash_cost_ratio=self.domain.wash_cost_ratio,
            cap=self.domain.max_change_per_period,
        )

        # 1. Exact persistent-type Markov transition (clean deterred types do NOT decay!)
        p_stay_clean = np.where(d_clean, 1.0 - self.slip_rate, self.slip_rate)
        w_clean_pred = self.w_clean * p_stay_clean + self.w_dirty * self.dirty_clear_slip
        w_dirty_pred = self.w_dirty * (1.0 - self.dirty_clear_slip) + self.w_clean * (1.0 - p_stay_clean)

        # 2. Optional smooth Bayesian likelihood update from public report changes Delta r_i
        if self.use_report_signal:
            delta_r = pre_rep - snap.report_ref
            decay = self.domain.natural_decay * self.period_days / TRADING_DAYS_PER_YEAR
            reversion = getattr(self.domain, "natural_reversion", 0.0) or 0.0
            if reversion > 0.0:
                retention = float(np.exp(-reversion * self.period_days / TRADING_DAYS_PER_YEAR))
                rev_drift = -(1.0 - retention) * (snap.report_ref - float(snap.report_ref.mean()))
                sigma_r = max(0.05, float(np.sqrt(max(1e-6, 1.0 - retention**2))))
            else:
                rev_drift = np.zeros(self.n_firms, dtype=np.float64)
                sigma_r = 0.035

            mu_clean_upper = 1.25 * step_real_clean - decay + rev_drift[:, None]
            excess = np.maximum(0.0, delta_r[:, None] - mu_clean_upper - 0.015)
            lik_clean = np.maximum(0.02, np.exp(-0.5 * (excess / sigma_r) ** 2))

            c_dirty = self._cost_matrix(snap.g_dirty_ref)
            pi_dirty = 1.0 if snap.regime.fixed_fine_basis == "report" else 0.0
            d_dirty, _, step_wash_dirty = _evaluate_type_choices(
                c_mat=c_dirty,
                benefit=snap.benefit,
                f_frac=f_frac,
                pi=pi_dirty,
                q_vec=snap.planned_q,
                a_real=a_real,
                wash_weights=wash_weights,
                audit_counts=audit_counts,
                beta_1=beta_1,
                beta=beta,
                regime=snap.regime,
                wash_cost_ratio=self.domain.wash_cost_ratio,
                cap=self.domain.max_change_per_period,
            )
            not_at_ceiling = (pre_rep < self.report_bound - 0.05)[:, None]
            expected_jump = (~d_dirty) & (step_wash_dirty > step_real_clean + 0.03) & not_at_ceiling
            no_jump_observed = delta_r[:, None] <= mu_clean_upper + 0.005
            lik_dirty = np.where(expected_jump & no_jump_observed, 0.40, 1.0)

            w_clean_pred = w_clean_pred * lik_clean
            w_dirty_pred = w_dirty_pred * lik_dirty

        totals = w_clean_pred.sum(axis=1) + w_dirty_pred.sum(axis=1)
        valid = totals > 1e-12
        self.w_clean = np.where(valid[:, None], w_clean_pred / np.where(valid, totals, 1.0)[:, None], self.prior_w[None, :])
        self.w_dirty = np.where(valid[:, None], w_dirty_pred / np.where(valid, totals, 1.0)[:, None], 0.0)
        self._predicted_day = int(day)

    def condition_on_audit(
        self,
        *,
        day: int,
        pre_audit_report: FloatArray,
        post_audit_report: FloatArray,
        selected: BoolArray,
        caught: BoolArray,
    ) -> None:
        """Condition predicted posterior on the regulator's own audit outcome (after audit)."""
        self.predict(day=day, pre_audit_report=pre_audit_report)
        post_rep = np.asarray(post_audit_report, dtype=np.float64)
        sel = np.asarray(selected, dtype=bool)
        hit = np.asarray(caught, dtype=bool)

        decayed_dirty_g = self._decay_g_estimate(self.g_dirty_est, post_rep)
        self.g_dirty_est = np.where(sel, post_rep, decayed_dirty_g)

        for i in np.flatnonzero(sel):
            branch = self.w_dirty[i] if hit[i] else self.w_clean[i]
            total = float(branch.sum())
            self.w_clean[i] = branch / total if total > 1e-12 else self.prior_w
            self.w_dirty[i] = 0.0

        self.last_post_audit_report = post_rep.copy()

    def allocate(
        self,
        *,
        day: int,
        regime: PolicyRegime,
        prices: FloatArray,
        benefit: FloatArray,
        post_audit_report: FloatArray,
    ) -> FloatArray:
        """Compute budget-conserving RMAB-Stackelberg (public) or optimal-clearing (hidden) audits."""
        prices_arr = np.asarray(prices, dtype=np.float64)
        benefit_arr = np.asarray(benefit, dtype=np.float64)
        rep_arr = np.asarray(post_audit_report, dtype=np.float64)
        size = self.n_firms
        budget = size * regime.audit_probability
        if budget <= 0.0:
            return np.zeros(size, dtype=np.float64)
        if budget >= size:
            return np.ones(size, dtype=np.float64)

        disclosure_delay = (
            self.domain.disclosure_delay_days
            if regime.disclosure_delay_days is None
            else regime.disclosure_delay_days
        )
        a_real, wash_weights, audit_counts, beta_1, beta = _timing_factors(
            discount_rate=self.discount_rate,
            decision_day=day,
            horizon_days=self.horizon_days,
            repricing_interval=self.repricing_interval,
            period_days=self.period_days,
            implementation_delay=self.domain.implementation_delay_days,
            disclosure_delay=disclosure_delay,
            real_persistence=self.real_persistence,
        )
        q_star = _solve_zero_fine_threshold(
            a_real=a_real,
            wash_weights=wash_weights,
            audit_counts=audit_counts,
            beta_1=beta_1,
            beta=beta,
            tau=regime.transparency,
            penalty=regime.penalty,
            subsidy=regime.subsidy,
            wash_cost_ratio=self.domain.wash_cost_ratio,
            max_benefit=float(np.max(np.maximum(benefit_arr, 0.0))),
        )
        if q_star <= 1e-12:
            return np.full(size, regime.audit_probability, dtype=np.float64)

        weights = prices_arr if self.value_weighted else np.ones(size, dtype=np.float64)

        # Build q grid spanning [0, q_star] densely + endpoint 1.0 for post-q* fast clearing
        n_inner = max(12, self.grid_points - 1)
        u = np.linspace(0.0, 1.0, n_inner)
        q_inner = q_star * (u ** 1.6)
        q_inner[-1] = q_star
        if q_star < 1.0 - 1e-6:
            q_grid = np.append(q_inner, 1.0)
        else:
            q_grid = q_inner
        g_count = q_grid.size

        aw = np.power(1.0 - q_grid[:, None], audit_counts[None, :]) @ wash_weights
        denom = 1.0 - (1.0 - q_grid) * beta
        d_q = np.divide(beta_1 * q_grid, denom, out=np.zeros_like(q_grid), where=q_grid > 0.0)

        c_clean = self._cost_matrix(rep_arr)
        c_dirty = self._cost_matrix(self.g_dirty_est)
        f_frac = regime.fixed_fine / prices_arr
        b_col = np.maximum(benefit_arr, 0.0)[:, None]
        f_col = f_frac[:, None]
        cap = float(self.domain.max_change_per_period)
        w_ratio = float(self.domain.wash_cost_ratio)

        # Clean-branch value V_I
        m_real = b_col * a_real + regime.subsidy
        dr_c = np.clip(m_real / (2.0 * c_clean), 0.0, cap)
        vi_c = np.maximum(0.0, m_real * dr_c - c_clean * dr_c * dr_c)

        # Dirty-branch post-clear value V_I (at decayed g_dirty_est, immune to exp(alpha*x) bias)
        dr_d = np.clip(m_real / (2.0 * c_dirty), 0.0, cap)
        vi_d = np.maximum(0.0, m_real * dr_d - c_dirty * dr_d * dr_d)

        pi_dirty = 1.0 if regime.fixed_fine_basis == "report" else 0.0
        d_clean_grid = np.zeros((size, self.k_nodes, g_count), dtype=np.float64)
        d_dirty_grid = np.zeros((size, self.k_nodes, g_count), dtype=np.float64)
        d_after_clear_grid = np.zeros((size, self.k_nodes, g_count), dtype=np.float64)

        for g in range(g_count):
            net_w = np.maximum(b_col * (1.0 - regime.transparency) * aw[g] - regime.penalty * d_q[g], 0.0)
            dw_c = np.clip(net_w / (2.0 * w_ratio * c_clean), 0.0, cap)
            vw_c = net_w * dw_c - w_ratio * c_clean * dw_c * dw_c - f_col * d_q[g]
            d_clean_grid[:, :, g] = (vw_c <= vi_c + 1e-15).astype(np.float64)

            dw_d = np.clip(net_w / (2.0 * w_ratio * c_dirty), 0.0, cap)
            vw_d_open = net_w * dw_d - w_ratio * c_dirty * dw_d * dw_d - pi_dirty * f_col * d_q[g]
            d_dirty_grid[:, :, g] = (vw_d_open <= vi_d + 1e-15).astype(np.float64)
            vw_d_cleared = net_w * dw_d - w_ratio * c_dirty * dw_d * dw_d - f_col * d_q[g]
            d_after_clear_grid[:, :, g] = (vw_d_cleared <= vi_d + 1e-15).astype(np.float64)

        # Exact 2-state RMAB Bellman continuation differential Delta J_{i,k} = V_clean - V_dirty
        # Derived directly from discounted budget-shadow-price savings over the financing horizon:
        # clearing an open case reduces future per-period deterrence cost from q* to q_clean.
        horizon_periods = max(1, self.horizon_days // self.period_days)
        horizon_factor = float(beta * (1.0 - beta**horizon_periods) / max(1e-12, 1.0 - beta))

        if regime.fixed_fine > 0.0 and regime.fixed_fine_basis == "case":
            # First q_grid entry where each (i, k) is deterred when clean
            has_det_c = d_clean_grid.any(axis=2)
            idx_c = np.argmax(d_clean_grid > 0.5, axis=2)
            q_det_c = np.where(has_det_c, q_grid[idx_c], q_star)
            delta_j_clean = horizon_factor * np.maximum(0.0, 1.0 - q_det_c / q_star)

            has_det_d = d_after_clear_grid.any(axis=2)
            idx_d = np.argmax(d_after_clear_grid > 0.5, axis=2)
            q_det_d = np.where(has_det_d, q_grid[idx_d], q_star)
            delta_j_dirty = horizon_factor * np.maximum(0.0, 1.0 - q_det_d / q_star)
        else:
            delta_j_clean = np.zeros((size, self.k_nodes), dtype=np.float64)
            delta_j_dirty = np.zeros((size, self.k_nodes), dtype=np.float64)

        # Unconfounded hidden-rule optimal allocation: firms plan with uniform audit_probability,
        # so commitment yields zero deterrence; optimal policy maximizes expected clearing value.
        if not regime.audit_rule_public:
            # Expected clearing value per unit q_i
            clear_val = weights * np.sum(self.w_dirty * (1.0 + delta_j_dirty), axis=1)
            if float(clear_val.max()) <= 1e-12:
                return np.full(size, regime.audit_probability, dtype=np.float64)
            order = sorted(range(size), key=lambda i: (-float(clear_val[i]), i))
            prob = np.zeros(size, dtype=np.float64)
            rem = float(budget)
            for idx in order:
                if rem <= 1e-15:
                    break
                alloc = min(1.0, rem)
                prob[idx] = alloc
                rem = max(0.0, rem - alloc)
            return prob

        # Public Stackelberg commitment: exact Bellman expected value U_i(q) over q_grid
        q_row = q_grid[None, None, :]
        q_clean_val = (
            d_clean_grid * (1.0 + delta_j_clean[:, :, None])
            + (1.0 - d_clean_grid) * q_row * delta_j_clean[:, :, None]
        )
        q_dirty_val = d_dirty_grid + q_row * delta_j_dirty[:, :, None]

        # Expected value per firm across joint posterior (w_clean, w_dirty), shape (N, G)
        expected_u = (
            np.einsum("ik,ikg->ig", self.w_clean, q_clean_val)
            + np.einsum("ik,ikg->ig", self.w_dirty, q_dirty_val)
        ) * weights[:, None]

        # Global Upper-Concave-Envelope Water-Filling (exact Lagrangian optimum)
        all_segments: list[tuple[float, int, float, float]] = []
        for i in range(size):
            for slope, q_left, q_right in _upper_concave_hull_segments(q_grid, expected_u[i]):
                all_segments.append((slope, i, q_left, q_right))
        all_segments.sort(key=lambda item: (-item[0], item[1], item[2]))

        probability = np.zeros(size, dtype=np.float64)
        remaining = float(budget)
        for _, idx, q_left, q_right in all_segments:
            if remaining <= 1e-15:
                break
            dq = q_right - q_left
            if dq <= remaining + 1e-15:
                probability[idx] = q_right
                remaining = max(0.0, remaining - dq)
            else:
                probability[idx] = q_left + remaining
                remaining = 0.0
                break

        if remaining > 1e-15:
            _distribute_budget_proportionally(probability, remaining, weights, np.ones(size, dtype=bool))
        return probability
