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
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .domain import DomainSpec

FIXED_FINE_BASES = ("case", "report")
AUDIT_ALLOCATIONS = ("uniform", "value", "jump")
ASSURANCE_SCOPES = ("none", "voluntary", "large", "all")
ASSURER_SELECTIONS = ("assigned", "firm")
# Assurance options and their defaults; at the defaults they are left out of a regime's identity.
ASSURANCE_DEFAULTS = {"assurance_scope": "none", "assurer_selection": "assigned", "assurer_liability": 0.0,
                      "assurance_fee": 0.0, "assurer_effort_cost": 0.0, "assurer_count": 4}


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

    def __post_init__(self) -> None:
        if self.assurance_scope not in ASSURANCE_SCOPES:
            raise ValueError(f"assurance_scope must be one of {ASSURANCE_SCOPES}")
        if self.assurer_selection not in ASSURER_SELECTIONS:
            raise ValueError(f"assurer_selection must be one of {ASSURER_SELECTIONS}")
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
