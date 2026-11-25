"""Zero-leakage empirical-belief bounded rationality for disclosure firms (Phase 3 / L2).

Relaxes rational expectations over enforcement parameters without using any oracle parameter
reading or synthetic surrogate rewards:

- Firms retain local corporate-finance appraisal (own marginal cost, local financing-rate
  sensitivity, and public disclosure transparency) over the identical continuous action space
  as the ``npv`` benchmark, eliminating discrete-grid confounds.
- Firms NEVER read ``regime.audit_probability``, targeted audit rules ``q_i``, ``regime.fixed_fine``,
  ``regime.penalty``, or true assurance provider quality ``a_k``.
- Instead, firms learn online strictly from realized public enforcement outcomes:
  1. Discounted Beta-Binomial tier audit hazard ``q_hat[tier]`` from realized peer audits;
  2. Exponentially smoothed statutory fine ``F_hat`` (and proportional penalty ``p_hat``) from
     realized enforcement penalties on caught firms;
  3. Discounted Beta-Binomial assurance detection rate ``a_hat[k]`` per provider from realized
     pre-publication checks and corrections, paired with Discounted Thompson Sampling for
     provider selection under firm choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class EmpiricalBeliefLearner:
    """Online empirical enforcement belief state learned strictly from realized market events."""

    n_tiers: int = 3
    n_providers: int = 4
    decay: float = 0.15
    # Weak pre-simulation baseline priors (initialized from domain baseline, never from interventions)
    audit_prior_strength: float = 20.0
    baseline_audit_probability: float = 0.01
    baseline_fixed_fine: float = 0.0
    baseline_penalty: float = 0.0
    provider_prior_alpha: float = 0.1
    provider_prior_beta: float = 1.9
    audit_prior_alpha: float = field(init=False)
    audit_prior_beta: float = field(init=False)
    audit_alpha: FloatArray = field(init=False)
    audit_beta: FloatArray = field(init=False)
    emp_fixed_fine_money: float = field(init=False)
    emp_penalty_rate: float = field(init=False)
    fine_seen: bool = field(init=False)
    penalty_seen: bool = field(init=False)
    provider_alpha: FloatArray = field(init=False)
    provider_beta: FloatArray = field(init=False)
    band_alpha: FloatArray | None = field(init=False, default=None)
    band_beta: FloatArray | None = field(init=False, default=None)
    audit_updates: int = 0
    assurance_updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.audit_prior_strength <= 0.0:
            raise ValueError("audit_prior_strength must be positive")
        q0 = float(np.clip(self.baseline_audit_probability, 1e-4, 1.0 - 1e-4))
        self.audit_prior_alpha = self.audit_prior_strength * q0
        self.audit_prior_beta = self.audit_prior_strength * (1.0 - q0)
        self.audit_alpha = np.full(self.n_tiers, self.audit_prior_alpha, dtype=np.float64)
        self.audit_beta = np.full(self.n_tiers, self.audit_prior_beta, dtype=np.float64)
        self.emp_fixed_fine_money = float(max(0.0, self.baseline_fixed_fine))
        self.fine_seen = self.emp_fixed_fine_money > 0.0
        self.emp_penalty_rate = float(max(0.0, self.baseline_penalty))
        self.penalty_seen = self.emp_penalty_rate > 0.0
        self.provider_alpha = np.full(self.n_providers, self.provider_prior_alpha, dtype=np.float64)
        self.provider_beta = np.full(self.n_providers, self.provider_prior_beta, dtype=np.float64)

    @property
    def tier_audit_probability(self) -> FloatArray:
        """Posterior mean empirical audit probability per size tier."""
        return self.audit_alpha / (self.audit_alpha + self.audit_beta)

    @property
    def pooled_audit_probability(self) -> float:
        """Market-wide pooled posterior mean empirical audit probability."""
        total_a = float(self.audit_alpha.sum())
        total_b = float(self.audit_beta.sum())
        return total_a / max(1e-12, total_a + total_b)

    @property
    def provider_quality_mean(self) -> FloatArray:
        """Posterior mean empirical pre-publication detection probability per provider."""
        return self.provider_alpha / (self.provider_alpha + self.provider_beta)

    def firm_audit_probabilities(self, size_class: IntArray) -> FloatArray:
        """Map each firm to its size tier's empirically learned audit probability."""
        return self.tier_audit_probability[size_class]

    def band_audit_probabilities(self, public_band: IntArray, n_bands: int) -> FloatArray:
        """Map each firm to its public risk band's empirically learned audit probability (zero oracle peeking)."""
        if self.band_alpha is None or self.band_beta is None or self.band_alpha.size != n_bands:
            q0 = self.audit_prior_alpha / max(1e-12, self.audit_prior_alpha + self.audit_prior_beta)
            return np.full(public_band.size, q0, dtype=np.float64)
        means = self.band_alpha / (self.band_alpha + self.band_beta)
        return means[public_band]

    def observe_audits(
        self,
        *,
        size_class: IntArray,
        selected: BoolArray,
        caught: BoolArray,
        realized_fixed_fine_money: FloatArray,
        realized_proportional_rate: FloatArray,
        public_band: IntArray | None = None,
        n_bands: int = 0,
    ) -> None:
        """Update tier/band audit hazards and statutory penalty beliefs strictly from realized audits."""
        retain = 1.0 - self.decay
        for tier in range(self.n_tiers):
            mask = size_class == tier
            n_tier = int(mask.sum())
            if n_tier > 0:
                k_sel = float(selected[mask].sum())
                self.audit_alpha[tier] = (
                    retain * self.audit_alpha[tier] + self.decay * self.audit_prior_alpha + k_sel
                )
                self.audit_beta[tier] = (
                    retain * self.audit_beta[tier] + self.decay * self.audit_prior_beta + (n_tier - k_sel)
                )

        if public_band is not None and n_bands >= 2:
            if self.band_alpha is None or self.band_beta is None or self.band_alpha.size != n_bands:
                self.band_alpha = np.full(n_bands, self.audit_prior_alpha, dtype=np.float64)
                self.band_beta = np.full(n_bands, self.audit_prior_beta, dtype=np.float64)
            for band in range(n_bands):
                mask = public_band == band
                n_b = int(mask.sum())
                if n_b > 0:
                    k_sel = float(selected[mask].sum())
                    self.band_alpha[band] = (
                        retain * self.band_alpha[band] + self.decay * self.audit_prior_alpha + k_sel
                    )
                    self.band_beta[band] = (
                        retain * self.band_beta[band] + self.decay * self.audit_prior_beta + (n_b - k_sel)
                    )

        if caught.any():
            mean_fixed = float(np.mean(realized_fixed_fine_money[caught]))
            if not self.fine_seen:
                self.emp_fixed_fine_money = mean_fixed
                self.fine_seen = True
            else:
                self.emp_fixed_fine_money = retain * self.emp_fixed_fine_money + self.decay * mean_fixed

            mean_prop = float(np.mean(realized_proportional_rate[caught]))
            if not self.penalty_seen:
                self.emp_penalty_rate = mean_prop
                self.penalty_seen = True
            else:
                self.emp_penalty_rate = retain * self.emp_penalty_rate + self.decay * mean_prop

        self.audit_updates += 1

    def observe_assurance_period(
        self,
        *,
        checked_inflated: BoolArray,
        corrected: BoolArray,
        client_of: IntArray,
    ) -> None:
        """Update discounted Beta posteriors per provider strictly from realized check outcomes."""
        retain = 1.0 - self.decay
        for k in range(self.n_providers):
            mask = checked_inflated & (client_of == k)
            n_obs = int(mask.sum())
            n_corr = float(corrected[mask].sum()) if n_obs > 0 else 0.0
            # Gentle forgetting toward the weak prior widens uncertainty on untested providers,
            # allowing Discounted Thompson Sampling to probe previously strict providers naturally.
            self.provider_alpha[k] = (
                retain * self.provider_alpha[k] + self.decay * self.provider_prior_alpha + n_corr
            )
            self.provider_beta[k] = (
                retain * self.provider_beta[k] + self.decay * self.provider_prior_beta + (n_obs - n_corr)
            )
        self.assurance_updates += 1

    def sample_provider_qualities(self, n_firms: int, rng: np.random.Generator) -> FloatArray:
        """Draw per-firm Thompson samples of shape ``(n_firms, n_providers)`` from Beta posteriors."""
        samples = np.empty((n_firms, self.n_providers), dtype=np.float64)
        for k in range(self.n_providers):
            a_k = max(1e-6, float(self.provider_alpha[k]))
            b_k = max(1e-6, float(self.provider_beta[k]))
            samples[:, k] = rng.beta(a_k, b_k, size=n_firms)
        return samples

    def choose_providers_thompson(
        self,
        *,
        mandated: BoolArray,
        current_providers: IntArray,
        value_fn,
        rng: np.random.Generator,
        draws: FloatArray | None = None,
    ) -> IntArray:
        """Discounted Thompson Sampling provider selection (zero access to true provider quality).

        Each mandated firm draws a Thompson sample ``a_tilde_{i,k} ~ Beta(alpha_k, beta_k)`` for
        every provider ``k``, evaluates ``value_fn(a_tilde[:, k])`` and switches if the best
        sampled provider strictly improves on its current provider's sampled value.
        """
        n_firms = mandated.size
        chosen = current_providers.copy()
        sampled = self.sample_provider_qualities(n_firms, rng) if draws is None else draws
        options = np.column_stack([value_fn(sampled[:, k]) for k in range(self.n_providers)])
        best = np.argmax(options, axis=1)
        rows = np.arange(n_firms)
        current = np.maximum(current_providers, 0)
        unassigned = mandated & (current_providers < 0)
        improved = mandated & (current_providers >= 0) & (options[rows, best] > options[rows, current] + 1e-12)
        switch = unassigned | improved
        chosen[switch] = best[switch]
        return chosen
