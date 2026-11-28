"""Regulatory regime and its schedule.

Instruments act through distinct channels:

* ``audit_probability`` – chance per decision period that a firm's report is
  checked; a detected inflation is removed and fined;
* ``penalty`` – fine per standard deviation of detected inflation, as a
  fraction of market cap;
* ``fixed_fine`` – a lump-sum fine per detected case in model money units
  (mean initial firm value = 1), whatever the amount inflated. One audit
  that finds inflation closes one case, so inflation added to a report that
  is already inflated and undetected adds no fixed fine. With
  ``fixed_fine_basis="report"`` the audit instead charges one fixed fine for
  every inflated report disclosed into the undetected stock;
* ``transparency`` – weight investors put on the true attribute rather than
  the report (``signal = true + (1 - transparency) * inflation``);
* ``subsidy`` – payment per standard deviation of real improvement, as a
  fraction of market cap;
* ``audit_allocation`` – who is audited. ``"uniform"`` checks every firm with
  ``audit_probability``. ``"value"`` and ``"jump"`` keep the expected number of
  audits, n * ``audit_probability``, and share it in proportion to market
  value, or half at random and half among firms whose report rose by more
  than ``audit_jump_threshold`` since the previous audit. Firms plan with
  their own probabilities only if ``audit_rule_public``;
* ``assurance_scope`` – which firms have their report checked by one of
  ``assurer_count`` assurance providers before every disclosure: none, all,
  only the large size class, or those that choose to buy it
  (``"voluntary"``). A check finds the inflation in the report with the
  provider's quality as probability and has it corrected before publication,
  without a fine. Providers set their quality once a year as a best response
  to ``assurer_liability`` (paid when an audit finds inflation in a report
  they checked) and to the effort cost ``assurer_effort_cost * quality**2``
  per check; firms pay ``assurance_fee`` per check. Under
  ``assurer_selection="firm"`` mandated firms pick their provider each year;
  under ``"assigned"`` they are assigned at random once;
* disclosure error and delay, and dated audit publications, affect the
  public signal separately from the audit and fine instruments.

In this model, reports follow a global publication calendar. Each checked
firm also updates its ordinary report at close, separately from certification.

Audit probability also determines how long inflated reports survive, so
equal probability-times-penalty does not hold either their value or the
discounted fine constant. A transparency comparison keeps audit probability,
penalty and subsidy individually fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import erfc, isfinite, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import TRADING_DAYS_PER_YEAR
from .domain import DomainSpec

_INV_SQRT2 = 1.0 / sqrt(2.0)


def _normal_cdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard normal CDF without requiring the optional ``scipy`` dependency."""
    arr = np.asarray(values, dtype=np.float64)
    return np.fromiter(
        (0.5 * erfc(-float(x) * _INV_SQRT2) for x in arr.flat),
        dtype=np.float64,
        count=arr.size,
    ).reshape(arr.shape)


def _timing_factors(
    *,
    discount_rate: float,
    decision_day: int,
    horizon_days: int,
    repricing_interval: int,
    period_days: int,
    implementation_delay: int,
    disclosure_delay: int,
    real_persistence: NDArray[np.float64] | None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float, float]:
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

FIXED_FINE_BASES = ("case", "report")
ORACLE_STACKELBERG_ALLOCATIONS = ("stackelberg_eq", "stackelberg_val")
BAYESIAN_STACKELBERG_ALLOCATIONS = ("stackelberg_obs", "stackelberg_obs_val", "stackelberg_obs_sig")
STACKELBERG_ALLOCATIONS = (*ORACLE_STACKELBERG_ALLOCATIONS, *BAYESIAN_STACKELBERG_ALLOCATIONS)
AUDIT_ALLOCATIONS = ("uniform", "value", "jump", *STACKELBERG_ALLOCATIONS)
ASSURANCE_SCOPES = ("none", "voluntary", "large", "all")
ASSURER_SELECTIONS = ("assigned", "firm", "rotating")
ASSURER_OBSERVATIONS = ("exact", "detected_bayes")
# Assurance options and their defaults; at the defaults they are left out of a regime's identity.
ASSURANCE_DEFAULTS = {
    "assurance_scope": "none",
    "assurer_selection": "assigned",
    "assurer_liability": 0.0,
    "assurance_fee": 0.0,
    "assurer_effort_cost": 0.0,
    "assurer_count": 4,
    "assurer_observation": "exact",
    "assurer_prior_share": 0.2,
    "assurer_prior_strength": 2.0,
    "assurer_smoothing": 0.0,
    "assurer_min_quality": 0.0,
}


@dataclass(frozen=True, slots=True)
class PolicyRegime:
    audit_probability: float
    penalty: float
    transparency: float
    subsidy: float
    disclosure_noise_sd: float = 0.0
    disclosure_delay_days: int | None = None
    publish_audit_results: bool = False
    reporting_interval_days: int = 1
    fixed_fine: float = 0.0
    # "case": one fixed fine per detected case; "report": one per inflated report in it.
    fixed_fine_basis: str = "case"
    # "uniform", "value" or "jump"; see audit_probabilities.
    audit_allocation: str = "uniform"
    # Whether firms know the allocation; otherwise they plan with audit_probability.
    audit_rule_public: bool = True
    # Rise of a firm's report (true value plus inflation) since the previous audit that flags it.
    audit_jump_threshold: float = 0.125
    # Optional coarse public risk-band quantization for Stackelberg commitments (0 = exact per-firm q_i, >=2 = G bands).
    audit_commitment_buckets: int = 0
    # When True (requires audit_commitment_buckets >= 2 and audit_rule_public), the regulator also
    # publishes each public risk band's target audit rate q_g so bounded-rationality learners need
    # not infer band rates purely from sparse audit counts.
    audit_band_rates_public: bool = False
    # "none", "voluntary", "large" (the large size class) or "all"; see the module docstring.
    assurance_scope: str = "none"
    # "assigned": mandated firms are assigned at random once; "firm": they pick a provider every year.
    assurer_selection: str = "assigned"
    # Paid by a provider when an audit finds inflation in a report it checked; model money units.
    assurer_liability: float = 0.0
    # Paid by a firm per check; model money units.
    assurance_fee: float = 0.0
    # kappa in a provider's effort cost kappa * quality**2 per check; model money units.
    assurer_effort_cost: float = 0.0
    assurer_count: int = 4
    # "exact": full lagged observation of s_k; "detected_bayes": updates only from own corrections & liability events.
    assurer_observation: str = "exact"
    assurer_prior_share: float = 0.2
    assurer_prior_strength: float = 2.0
    # Runtime scrutiny hysteresis gamma in [0, 1): a_{y+1} = max(min_q, gamma * a_y + (1 - gamma) * target).
    assurer_smoothing: float = 0.0
    # Statutory minimum assurance quality floor in [0, 1].
    assurer_min_quality: float = 0.0

    def __post_init__(self) -> None:
        if self.assurance_scope not in ASSURANCE_SCOPES:
            raise ValueError(f"assurance_scope must be one of {ASSURANCE_SCOPES}")
        if self.assurer_selection not in ASSURER_SELECTIONS:
            raise ValueError(f"assurer_selection must be one of {ASSURER_SELECTIONS}")
        if self.assurer_observation not in ASSURER_OBSERVATIONS:
            raise ValueError(f"assurer_observation must be one of {ASSURER_OBSERVATIONS}")
        if (type(self.assurer_prior_share) not in (int, float)
                or not isfinite(self.assurer_prior_share)
                or not 0.0 <= self.assurer_prior_share <= 1.0):
            raise ValueError("assurer_prior_share must be finite and in [0, 1]")
        if (type(self.assurer_prior_strength) not in (int, float)
                or not isfinite(self.assurer_prior_strength)
                or self.assurer_prior_strength <= 0.0):
            raise ValueError("assurer_prior_strength must be finite and positive")
        if (type(self.assurer_smoothing) not in (int, float)
                or not isfinite(self.assurer_smoothing)
                or not 0.0 <= self.assurer_smoothing < 1.0):
            raise ValueError("assurer_smoothing must be finite and in [0, 1)")
        if (type(self.assurer_min_quality) not in (int, float)
                or not isfinite(self.assurer_min_quality)
                or not 0.0 <= self.assurer_min_quality <= 1.0):
            raise ValueError("assurer_min_quality must be finite and in [0, 1]")
        for name in ("assurer_liability", "assurance_fee", "assurer_effort_cost"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.assurer_count) is not int or self.assurer_count < 1:
            raise ValueError("assurer_count must be a positive integer")
        if self.assurance_scope == "none":
            if any(getattr(self, name) != value for name, value in ASSURANCE_DEFAULTS.items()):
                raise ValueError("assurance options need an assurance_scope")
        elif self.assurer_effort_cost <= 0:
            raise ValueError("assurance needs a positive assurer_effort_cost")
        if self.fixed_fine_basis not in FIXED_FINE_BASES:
            raise ValueError(f"fixed_fine_basis must be one of {FIXED_FINE_BASES}")
        if self.audit_allocation not in AUDIT_ALLOCATIONS:
            raise ValueError(f"audit_allocation must be one of {AUDIT_ALLOCATIONS}")
        if type(self.audit_rule_public) is not bool:
            raise ValueError("audit_rule_public must be boolean")
        if not isfinite(self.audit_jump_threshold) or self.audit_jump_threshold <= 0:
            raise ValueError("audit_jump_threshold must be finite and positive")
        if type(self.audit_commitment_buckets) is not int or self.audit_commitment_buckets < 0 or self.audit_commitment_buckets == 1:
            raise ValueError("audit_commitment_buckets must be 0 (exact) or an integer >= 2")
        if self.audit_commitment_buckets >= 2 and (
            self.audit_allocation not in STACKELBERG_ALLOCATIONS or not self.audit_rule_public
        ):
            raise ValueError("audit_commitment_buckets requires a public Stackelberg audit_allocation")
        if type(self.audit_band_rates_public) is not bool:
            raise ValueError("audit_band_rates_public must be boolean")
        if self.audit_band_rates_public and self.audit_commitment_buckets < 2:
            raise ValueError("audit_band_rates_public requires audit_commitment_buckets >= 2")
        if not 0 <= self.audit_probability <= 1 or not 0 <= self.transparency <= 1:
            raise ValueError("audit_probability and transparency must be in [0, 1]")
        if self.penalty < 0 or self.subsidy < 0:
            raise ValueError("penalty and subsidy must be nonnegative")
        if not isfinite(self.fixed_fine) or self.fixed_fine < 0:
            raise ValueError("fixed_fine must be finite and nonnegative")
        if not isfinite(self.disclosure_noise_sd) or self.disclosure_noise_sd < 0:
            raise ValueError("disclosure_noise_sd must be finite and nonnegative")
        if self.disclosure_delay_days is not None and (
            type(self.disclosure_delay_days) is not int or self.disclosure_delay_days < 0
        ):
            raise ValueError("disclosure_delay_days must be a nonnegative integer or None")
        if type(self.publish_audit_results) is not bool:
            raise ValueError("publish_audit_results must be boolean")
        if type(self.reporting_interval_days) is not int or self.reporting_interval_days < 1:
            raise ValueError("reporting_interval_days must be a positive integer")

    @classmethod
    def baseline(cls, domain: DomainSpec) -> "PolicyRegime":
        return cls(
            audit_probability=domain.baseline_audit_probability,
            penalty=domain.baseline_penalty,
            transparency=domain.baseline_transparency,
            subsidy=domain.baseline_subsidy,
            fixed_fine=domain.baseline_fixed_fine,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialized regime; defaults of later options are omitted so earlier identities are unchanged."""
        payload = asdict(self)
        if not self.fixed_fine:
            payload.pop("fixed_fine")
        if self.fixed_fine_basis == "case":
            payload.pop("fixed_fine_basis")
        if self.audit_allocation == "uniform":
            payload.pop("audit_allocation")
        if self.audit_rule_public:
            payload.pop("audit_rule_public")
        if self.audit_allocation != "jump":
            payload.pop("audit_jump_threshold")
        if self.audit_commitment_buckets == 0:
            payload.pop("audit_commitment_buckets")
        if not self.audit_band_rates_public:
            payload.pop("audit_band_rates_public")
        for name, value in ASSURANCE_DEFAULTS.items():
            if payload[name] == value:
                payload.pop(name)
        return payload

    @property
    def assurer_slope(self) -> float:
        """lambda in a provider's best response min(1, lambda * s), s its share of checks that found inflation.

        Minimizing kappa a^2 + L q s (1 - a): an unchecked inflated report is found at the next audit
        with probability q, and the provider pays L.
        """
        if self.assurance_scope == "none":
            return 0.0
        return self.assurer_liability * self.audit_probability / (2.0 * self.assurer_effort_cost)


def audit_probabilities(
    regime: PolicyRegime, values: NDArray[np.float64], flagged: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Each firm's audit probability; all allocations expect n * audit_probability audits.

    ``"value"`` shares them in proportion to market value ``values``. ``"jump"``
    gives every firm half the mean probability and shares the other half
    equally among the ``flagged`` firms, or among all firms when none is
    flagged. A probability above one is capped and the excess goes to the
    other firms by the same rule: in proportion to value, or equally to the
    unflagged firms.
    """
    values = np.asarray(values, dtype=np.float64)
    size, mean = values.size, regime.audit_probability
    if regime.audit_allocation == "uniform":
        return np.full(size, mean)
    budget = size * mean
    if regime.audit_allocation == "value":
        probability = np.ones(size)
        free = np.ones(size, dtype=bool)
        while free.any():
            probability[free] = (budget - (size - free.sum())) * values[free] / values[free].sum()
            over = free & (probability > 1.0)
            if not over.any():
                break
            probability[over] = 1.0
            free &= ~over
        return probability
    if regime.audit_allocation in STACKELBERG_ALLOCATIONS:
        raise ValueError(f"{regime.audit_allocation} requires stackelberg_audit_probabilities with firm state")
    flagged = np.zeros(size, dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool)
    count = int(flagged.sum())
    if count == 0:
        return np.full(size, mean)
    probability = np.full(size, 0.5 * mean)
    probability[flagged] += 0.5 * budget / count
    # Flagged firms share equally, so either all of them exceed one or none does.
    if count < size and probability[flagged][0] > 1.0:
        probability[~flagged] += (probability[flagged] - 1.0).sum() / (size - count)
        probability[flagged] = 1.0
    return probability


def _distribute_budget_proportionally(
    probability: NDArray[np.float64],
    remaining: float,
    weights: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> float:
    """Add ``remaining`` audit budget proportionally to ``weights`` on ``mask``, capping entries at 1."""
    if remaining <= 1e-15:
        return 0.0
    active = mask & (probability < 1.0 - 1e-15)
    while remaining > 1e-15 and active.any():
        headroom = 1.0 - probability[active]
        total_headroom = float(headroom.sum())
        if total_headroom <= remaining + 1e-15:
            probability[active] = 1.0
            remaining = max(0.0, remaining - total_headroom)
            break
        w = np.maximum(weights[active], 0.0)
        w_sum = float(w.sum())
        shares = w / w_sum if w_sum > 0.0 else np.full(active.sum(), 1.0 / active.sum())
        add = remaining * shares
        over = add >= headroom
        if not over.any():
            probability[active] += add
            remaining = 0.0
            break
        idx = np.flatnonzero(active)
        capped_idx = idx[over]
        used = float((1.0 - probability[capped_idx]).sum())
        probability[capped_idx] = 1.0
        remaining = max(0.0, remaining - used)
        active[capped_idx] = False
    return remaining


def firm_deterrence_thresholds(
    *,
    prices: NDArray[np.float64],
    benefit: NDArray[np.float64],
    marginal_cost: NDArray[np.float64],
    new_case_prob: NDArray[np.float64],
    regime: PolicyRegime,
    wash_cost_ratio: float | NDArray[np.float64],
    discount_rate: float,
    decision_day: int,
    horizon_days: int,
    repricing_interval: int,
    period_days: int,
    implementation_delay: int,
    disclosure_delay: int,
    real_persistence: NDArray[np.float64] | None = None,
    max_change_per_period: float | None = None,
    bisection_steps: int = 40,
) -> NDArray[np.float64]:
    """Exact minimum deterring audit probability q_i^det in [0, 1] (or inf if undeterable) per firm.

    Uses the exact audit-count grouping of ``financing_discount_factors`` (at most ~21 distinct
    audit counts over a 5-year horizon), accounting for ``max_change_per_period`` caps when given.
    """
    prices = np.asarray(prices, dtype=np.float64)
    size = prices.size
    benefit_arr = np.broadcast_to(np.asarray(benefit, dtype=np.float64), (size,))
    cost_arr = np.broadcast_to(np.asarray(marginal_cost, dtype=np.float64), (size,))
    pi_arr = np.broadcast_to(np.asarray(new_case_prob, dtype=np.float64), (size,))
    w_arr = np.broadcast_to(np.asarray(wash_cost_ratio, dtype=np.float64), (size,))

    a_real, wash_weights, audit_counts, beta_1, beta = _timing_factors(
        discount_rate=discount_rate,
        decision_day=decision_day,
        horizon_days=horizon_days,
        repricing_interval=repricing_interval,
        period_days=period_days,
        implementation_delay=implementation_delay,
        disclosure_delay=disclosure_delay,
        real_persistence=real_persistence,
    )

    tau = regime.transparency
    penalty = regime.penalty
    subsidy = regime.subsidy
    fixed_fine_frac = (regime.fixed_fine / prices) * pi_arr
    m_real = benefit_arr * a_real + subsidy
    step_real = np.maximum(m_real / (2.0 * cost_arr), 0.0)
    if max_change_per_period is not None:
        step_real = np.minimum(step_real, max_change_per_period)
    vi = m_real * step_real - cost_arr * step_real**2

    def surplus(q: NDArray[np.float64]) -> NDArray[np.float64]:
        aw = np.power(1.0 - q[:, None], audit_counts[None, :]) @ wash_weights
        denom = 1.0 - (1.0 - q) * beta
        fine_disc = np.divide(beta_1 * q, denom, out=np.zeros_like(q), where=q > 0.0)
        net_wash = np.maximum(benefit_arr * (1.0 - tau) * aw - penalty * fine_disc, 0.0)
        step_wash = net_wash / (2.0 * w_arr * cost_arr)
        if max_change_per_period is not None:
            step_wash = np.minimum(step_wash, max_change_per_period)
        vw = net_wash * step_wash - w_arr * cost_arr * step_wash**2 - fixed_fine_frac * fine_disc
        return vw - vi

    q_det = np.zeros(size, dtype=np.float64)
    s0 = surplus(np.zeros(size, dtype=np.float64))
    s1 = surplus(np.ones(size, dtype=np.float64))
    undeterable = s1 > 0.0
    q_det[undeterable] = np.inf
    active = (s0 > 0.0) & (~undeterable)
    if active.any():
        lo = np.zeros(size, dtype=np.float64)
        hi = np.ones(size, dtype=np.float64)
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            pos = surplus(mid) > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        q_det[active] = hi[active]
    return q_det


def stackelberg_audit_probabilities(
    *,
    regime: PolicyRegime,
    prices: NDArray[np.float64],
    benefit: NDArray[np.float64],
    marginal_cost: NDArray[np.float64],
    new_case_prob: NDArray[np.float64],
    wash_cost_ratio: float | NDArray[np.float64],
    discount_rate: float,
    decision_day: int,
    horizon_days: int,
    repricing_interval: int,
    period_days: int,
    implementation_delay: int,
    disclosure_delay: int,
    real_persistence: NDArray[np.float64] | None = None,
    max_change_per_period: float | None = None,
    deterrence_margin: float = 1e-5,
    has_open_case: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Budget-conserving full-information oracle Stackelberg audit allocation (Proposition 5).

    Under ``audit_rule_public=True``, allocates expected budget ``K = N * regime.audit_probability``
    to maximize weighted single-period deterrence (equal weights for ``stackelberg_eq``; market-value
    weights for ``stackelberg_val``) by funding firms' exact deterrence thresholds ``q_i^det`` in
    bang-per-buck order, then channels any residual budget to undeterred firms to accelerate
    open-case clearance while strictly conserving ``sum(q) == K``.

    Under ``audit_rule_public=False`` (unconfounded hidden-rule optimum), firms plan with uniform
    ``regime.audit_probability`` so commitment has no deterrence effect; the oracle instead maximizes
    expected weighted open-case detections/clearings subject to ``q_i in [0, 1]`` and ``sum(q) == K``.
    """
    if regime.audit_allocation not in STACKELBERG_ALLOCATIONS:
        raise ValueError(f"unsupported Stackelberg allocation: {regime.audit_allocation}")
    prices = np.asarray(prices, dtype=np.float64)
    size = prices.size
    budget = size * regime.audit_probability
    if budget <= 0.0:
        return np.zeros(size, dtype=np.float64)
    if budget >= size:
        return np.ones(size, dtype=np.float64)

    q_det = firm_deterrence_thresholds(
        prices=prices,
        benefit=benefit,
        marginal_cost=marginal_cost,
        new_case_prob=new_case_prob,
        regime=regime,
        wash_cost_ratio=wash_cost_ratio,
        discount_rate=discount_rate,
        decision_day=decision_day,
        horizon_days=horizon_days,
        repricing_interval=repricing_interval,
        period_days=period_days,
        implementation_delay=implementation_delay,
        disclosure_delay=disclosure_delay,
        real_persistence=real_persistence,
        max_change_per_period=max_change_per_period,
    )

    weights = prices if regime.audit_allocation == "stackelberg_val" else np.ones(size, dtype=np.float64)

    if not regime.audit_rule_public:
        # Unconfounded optimal hidden-rule detection/clearing: target firms with open cases first,
        # then firms expected to inflate under uniform audit_probability.
        open_mask = (
            np.asarray(new_case_prob, dtype=np.float64) < 1.0 - 1e-12
            if has_open_case is None
            else np.asarray(has_open_case, dtype=bool)
        )
        will_inflate = q_det > regime.audit_probability + 1e-12
        priority = np.where(open_mask, 2.0, np.where(will_inflate, 1.0, 0.0))
        probability = np.zeros(size, dtype=np.float64)
        remaining = float(budget)
        if float(priority.max()) > 0.0:
            order = sorted(range(size), key=lambda i: (-priority[i], -float(weights[i]), i))
            for idx in order:
                if remaining <= 1e-15 or priority[idx] <= 0.0:
                    break
                alloc = min(1.0, remaining)
                probability[idx] = alloc
                remaining = max(0.0, remaining - alloc)
        if remaining > 1e-15:
            _distribute_budget_proportionally(probability, remaining, weights, np.ones(size, dtype=bool))
        return probability

    probability = np.zeros(size, dtype=np.float64)
    deterred = q_det <= 0.0
    candidates = np.flatnonzero((q_det > 0.0) & np.isfinite(q_det) & (q_det <= 1.0))

    # Bang-per-buck order: maximize omega_i / q_i^det (exact optimum for equal weights omega_i == 1).
    needs = np.minimum(1.0, q_det + deterrence_margin)
    order = sorted(
        candidates.tolist(),
        key=lambda i: (-weights[i] / needs[i], needs[i], i),
    )

    remaining = float(budget)
    for idx in order:
        need = float(needs[idx])
        if need <= remaining + 1e-15:
            alloc = min(need, remaining)
            probability[idx] = alloc
            remaining = max(0.0, remaining - alloc)
            deterred[idx] = True

    # For value-weighted knapsack, check single-best-item 2-approximation safeguard.
    if regime.audit_allocation == "stackelberg_val" and candidates.size:
        feasible = [idx for idx in candidates.tolist() if float(needs[idx]) <= budget + 1e-15]
        if feasible:
            best_single = max(feasible, key=lambda i: weights[i])
            packed_val = float(weights[deterred & (q_det > 0.0)].sum())
            if weights[best_single] > packed_val:
                probability[:] = 0.0
                deterred = q_det <= 0.0
                alloc = min(float(needs[best_single]), budget)
                probability[best_single] = alloc
                deterred[best_single] = True
                remaining = max(0.0, budget - alloc)

    # Channel any residual budget to undeterred firms first (accelerates clearing open cases),
    # then across all firms if every firm is already deterred, conserving sum(q) == K exactly.
    undeterred = ~deterred
    if remaining > 1e-15 and undeterred.any():
        remaining = _distribute_budget_proportionally(probability, remaining, prices, undeterred)
    if remaining > 1e-15:
        _distribute_budget_proportionally(probability, remaining, prices, np.ones(size, dtype=bool))
    return probability


def _upper_concave_hull_segments(
    q_grid: NDArray[np.float64], p_values: NDArray[np.float64],
) -> list[tuple[float, float, float]]:
    """Return strictly decreasing-slope segments ``(slope, q_start, q_end)`` of upper concave hull."""
    hull_idx: list[int] = [0]
    for idx in range(1, q_grid.size):
        if p_values[idx] <= p_values[hull_idx[-1]] + 1e-15:
            continue
        while len(hull_idx) >= 2:
            i0, i1 = hull_idx[-2], hull_idx[-1]
            slope_prev = (p_values[i1] - p_values[i0]) / (q_grid[i1] - q_grid[i0])
            slope_curr = (p_values[idx] - p_values[i1]) / (q_grid[idx] - q_grid[i1])
            if slope_curr >= slope_prev - 1e-15:
                hull_idx.pop()
            else:
                break
        hull_idx.append(idx)
    segments: list[tuple[float, float, float]] = []
    for left, right in zip(hull_idx, hull_idx[1:]):
        dq = float(q_grid[right] - q_grid[left])
        dp = float(p_values[right] - p_values[left])
        if dq > 1e-15 and dp > 1e-15:
            segments.append((dp / dq, float(q_grid[left]), float(q_grid[right])))
    return segments


def bayesian_stackelberg_audit_probabilities(
    *,
    regime: PolicyRegime,
    prices: NDArray[np.float64],
    benefit_median: float | NDArray[np.float64],
    base_cost: NDArray[np.float64],
    cost_log_sd_eff: float,
    wash_cost_ratio: float,
    discount_rate: float,
    decision_day: int,
    horizon_days: int,
    repricing_interval: int,
    period_days: int,
    implementation_delay: int,
    disclosure_delay: int,
    real_persistence: NDArray[np.float64] | None = None,
    max_change_per_period: float | None = None,
    p_clean: NDArray[np.float64] | None = None,
    grid_points: int = 60,
    quadrature_points: int = 25,
    return_clean_deterrence: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observable-only Bayesian Stackelberg allocation maximizing expected deterred firms.

    Uses only publicly observable prices ``P_i``, public benefit ``benefit_median`` (scalar or
    per-firm public vector), public tier/report base cost ``base_cost``, log-cost uncertainty
    ``cost_log_sd_eff``, optional period action cap ``max_change_per_period``, and optional
    regulator belief ``p_clean`` maintained strictly from past audit selections (zero private firm
    cost shocks or hidden inflation state). Integrates exact capped deterrence via Gauss-Hermite
    quadrature and solves the upper-concave-envelope water-filling problem in ``O(N G log(N G))``
    while strictly conserving ``sum(q) == N * regime.audit_probability``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    base_cost = np.asarray(base_cost, dtype=np.float64)
    size = prices.size
    benefit_arr = np.broadcast_to(np.asarray(benefit_median, dtype=np.float64), (size,)).astype(np.float64)
    budget = size * regime.audit_probability
    if budget <= 0.0:
        z_out = np.zeros(size, dtype=np.float64)
        return (z_out, np.zeros(size, dtype=np.float64)) if return_clean_deterrence else z_out
    if budget >= size or float(np.max(benefit_arr)) <= 0.0 or regime.fixed_fine <= 0.0:
        u_out = np.full(size, min(1.0, regime.audit_probability), dtype=np.float64)
        return (u_out, np.ones(size, dtype=np.float64)) if return_clean_deterrence else u_out

    a_real, wash_weights, audit_counts, beta_1, beta = _timing_factors(
        discount_rate=discount_rate,
        decision_day=decision_day,
        horizon_days=horizon_days,
        repricing_interval=repricing_interval,
        period_days=period_days,
        implementation_delay=implementation_delay,
        disclosure_delay=disclosure_delay,
        real_persistence=real_persistence,
    )

    # Find universal zero-fine threshold q* so the grid spans [0, min(1, q* + eps)].
    tau = regime.transparency
    lo, hi = 0.0, 1.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        aw_m = float(np.power(1.0 - mid, audit_counts) @ wash_weights)
        if (1.0 - tau) * aw_m > np.sqrt(wash_cost_ratio) * a_real:
            lo = mid
        else:
            hi = mid
    q_star = min(1.0, hi + 1e-5)
    # Cluster grid points near 0 where D'(q) is steepest, plus exact endpoint q_star.
    u = np.linspace(0.0, 1.0, grid_points)
    q_grid = q_star * (u ** 1.6)
    q_grid[-1] = q_star

    aw = np.power(1.0 - q_grid[:, None], audit_counts[None, :]) @ wash_weights
    denom = 1.0 - (1.0 - q_grid) * beta
    d_q = np.divide(beta_1 * q_grid, denom, out=np.zeros_like(q_grid), where=q_grid > 0.0)

    sigma = max(float(cost_log_sd_eff), 1e-3)
    f_frac = regime.fixed_fine / prices  # (size,)
    p_det_clean = np.zeros((size, q_grid.size), dtype=np.float64)

    if max_change_per_period is None:
        surplus_num = (benefit_arr[:, None] ** 2) * np.maximum(
            ((1.0 - tau) * aw[None, :]) ** 2 / wash_cost_ratio - a_real ** 2, 0.0
        )
        for g in range(q_grid.size):
            if d_q[g] <= 0.0:
                p_det_clean[:, g] = np.where(surplus_num[:, g] <= 0.0, 1.0, 0.0)
            else:
                c_req = surplus_num[:, g] / (4.0 * f_frac * d_q[g])
                z = (np.log(np.maximum(c_req / base_cost, 1e-30)) + 0.5 * sigma**2) / sigma
                p_det_clean[:, g] = np.where(surplus_num[:, g] <= 0.0, 1.0, 1.0 - _normal_cdf(z))
    else:
        # Exact Gauss-Hermite integration over unobserved log-normal cost shock (E[eps] = 1)
        # accounting for period action-amplitude cap M (where V_W - V_I is hump-shaped in c).
        gh_x, gh_w = np.polynomial.hermite.hermgauss(quadrature_points)
        gh_w = gh_w / np.sqrt(np.pi)
        eps = np.exp(np.sqrt(2.0) * sigma * gh_x - 0.5 * sigma * sigma)
        c_mat = base_cost[:, None] * eps[None, :]
        b_col = np.maximum(benefit_arr, 0.0)[:, None]
        f_col = f_frac[:, None]
        cap = float(max_change_per_period)
        ar = b_col * a_real
        dr = np.minimum(ar / (2.0 * c_mat), cap)
        vi = np.maximum(0.0, ar * dr - c_mat * dr * dr)
        for g in range(q_grid.size):
            aw_col = b_col * (1.0 - tau) * aw[g]
            dw = np.minimum(aw_col / (2.0 * wash_cost_ratio * c_mat), cap)
            vw = aw_col * dw - wash_cost_ratio * c_mat * dw * dw - f_col * d_q[g]
            p_det_clean[:, g] = (vw <= vi + 1e-15).astype(np.float64) @ gh_w

    if p_clean is None:
        p_det = p_det_clean
    else:
        clean_col = np.clip(np.asarray(p_clean, dtype=np.float64), 0.0, 1.0)[:, None]
        at_qstar = (q_grid >= q_star - 1e-12).astype(np.float64)[None, :]
        p_det = clean_col * p_det_clean + (1.0 - clean_col) * at_qstar

    all_segments: list[tuple[float, int, float, float]] = []
    for i in range(size):
        for slope, q_left, q_right in _upper_concave_hull_segments(q_grid, p_det[i]):
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
        _distribute_budget_proportionally(probability, remaining, np.ones(size), np.ones(size, dtype=bool))

    if return_clean_deterrence:
        clean_det = np.array([float(np.interp(probability[i], q_grid, p_det_clean[i])) for i in range(size)])
        return probability, clean_det
    return probability


@dataclass(frozen=True, slots=True)
class PolicySchedule:
    """A baseline regime and one optional switch at ``start_day``."""

    baseline: PolicyRegime
    intervention: PolicyRegime | None = None
    start_day: int = 0

    def __post_init__(self) -> None:
        if self.start_day < 0:
            raise ValueError("start_day must be nonnegative")

    def at(self, day: int) -> PolicyRegime:
        if self.intervention is not None and day >= self.start_day:
            return self.intervention
        return self.baseline

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline.to_dict(),
                "intervention": None if self.intervention is None else self.intervention.to_dict(),
                "start_day": self.start_day}

    @classmethod
    def constant(cls, domain: DomainSpec) -> "PolicySchedule":
        return cls(baseline=PolicyRegime.baseline(domain))

    def with_intervention(self, start_day: int, **changes: float | bool | None) -> "PolicySchedule":
        return replace(self, intervention=replace(self.baseline, **changes), start_day=start_day)

    def with_information_intervention(
        self, start_day: int, *, transparency: float | None = None,
        disclosure_noise_sd: float | None = None, disclosure_delay_days: int | None = None,
        publish_audit_results: bool | None = None,
        reporting_interval_days: int | None = None,
    ) -> "PolicySchedule":
        """Change information instruments, retaining audit, fine and subsidy."""
        changes = {name: value for name, value in {
            "transparency": transparency, "disclosure_noise_sd": disclosure_noise_sd,
            "disclosure_delay_days": disclosure_delay_days, "publish_audit_results": publish_audit_results,
            "reporting_interval_days": reporting_interval_days,
        }.items() if value is not None}
        return self.with_intervention(start_day, **changes)
