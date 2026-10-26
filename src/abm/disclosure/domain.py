"""Domain specification: what the disclosed attribute is and what it costs.

The market core works in *goodness* units: the attribute standardized by its
initial cross-sectional distribution and signed so that higher is better.
A domain maps raw scores (a rating, an intensity, ...) to goodness and supplies costs, delays and the baseline regulatory regime. All
costs are fractions of the firm's market capitalisation: quadratic resource
cost coefficients are per squared standard deviation of goodness, while
penalties and subsidies are per standard deviation. The exception is
``baseline_fixed_fine``, a money amount per detected case, so its weight
relative to market cap falls with firm size. Direct size differences
in resource costs enter through ``size_cost_multipliers``.

By default goodness is a linear standardization of the raw score, so the raw
scale's ends bound the attribute. With ``observation_link="empirical_quantile"``
goodness is instead a latent standard-normal attribute and the raw score is its
image under the empirical quantile map of ``initial_empirical_distribution``:
a raw score at the top of the scale is then a saturated observation, not a cap
on the latent attribute. ``initial_size_rank_correlation`` sets a Gaussian
copula between initial latent goodness and log intrinsic value.

With ``signal_saturation`` (quantile link only) investors and published
audits see the attribute through the raw scale: values above the goodness of
the best raw end look the same. Reports, and the changes firms choose, then
stop at that observable maximum, as the scale ends bound them under the
linear link; true values drawn above it stay latent.

``natural_reversion`` replaces a pure downward drift by reversion towards the
cross-sectional mean with stationary dispersion, so the attribute has no
aggregate trend but individual firms drift back towards the middle.
``reversion_permanent_share`` splits the attribute into a permanent firm
level and a reverting part; ``reversion_improvement`` says which of the two
real improvement joins.
``wash_cost_log_sd`` gives each firm its own wash cost ratio.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

OBSERVATION_LINKS = ("linear", "empirical_quantile")
# Latent goodness range under the quantile link; a numerical bound far in the
# normal tails (two-sided mass about 2e-9), not an empirical scale end.
LATENT_GOODNESS_BOUND = 6.0


@dataclass(frozen=True, slots=True)
class DomainSpec:
    name: str
    attribute_label: str
    unit: str
    # +1 when a higher raw score is better (a rating), -1 when lower is
    # better (an intensity).
    direction: int
    initial_mean: float
    initial_sd: float
    raw_lower: float
    raw_upper: float
    # Goodness lost per year without effort (standards tighten, assets age).
    natural_decay: float
    # Cost of one standard deviation of real improvement, as a fraction of
    # market cap, for a mid-size firm: median and log dispersion across firms.
    real_cost_scale: float
    real_cost_log_sd: float
    size_cost_multipliers: tuple[float, float, float]
    # Marginal cost rises with the level already reached, cost * exp(eta * g);
    # without it a constant benefit makes firms improve every period forever.
    cost_level_elasticity: float
    # Most a firm can change its true attribute or its report in one decision
    # period, in standard deviations (capacity, not cost, binds beyond it).
    max_change_per_period: float
    # Cost of inflating the report by one standard deviation, relative to the
    # real cost.
    wash_cost_ratio: float
    implementation_delay_days: int
    disclosure_delay_days: int
    baseline_audit_probability: float
    baseline_penalty: float
    baseline_transparency: float
    baseline_subsidy: float
    action_labels: dict[str, str] = field(default_factory=dict)
    initial_empirical_distribution: tuple[tuple[float, int], ...] = ()
    observation_link: str = "linear"
    # Gaussian-copula correlation between initial latent goodness and the
    # standardized log intrinsic value; only valid with the quantile link.
    initial_size_rank_correlation: float = 0.0
    # Lump-sum fine per detected case, in model money units (mean initial
    # firm value = 1), unlike the per-SD market-cap fraction baseline_penalty.
    baseline_fixed_fine: float = 0.0
    # Signals saturate at the best raw end; quantile link only.
    signal_saturation: bool = False
    # Annual speed at which true values revert to the current equal-weight
    # cross-sectional mean, with idiosyncratic shocks that keep the
    # cross-sectional variance at its initial unit level (an
    # Ornstein-Uhlenbeck process; see AttributeState.advance). Zero keeps
    # the uniform natural_decay alone.
    natural_reversion: float = 0.0
    # Log dispersion of firm-level wash cost ratios around wash_cost_ratio
    # (mean-preserving lognormal, like real_cost_log_sd). Zero gives every
    # firm the same ratio.
    wash_cost_log_sd: float = 0.0
    # Under natural_reversion, the share of the cross-sectional variance that is
    # a permanent firm level mu_i; only the rest, g_i - mu_i, reverts and is
    # shocked (research contract section 25.2). Zero is pure reversion.
    reversion_permanent_share: float = 0.0
    # Under natural_reversion, whether real improvement joins the reverting part
    # ("transitory", like a shock) or the permanent level ("permanent").
    reversion_improvement: str = "transitory"

    def __post_init__(self) -> None:
        if not self.name or not self.attribute_label:
            raise ValueError("domain name and attribute label are required")
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        if self.initial_sd <= 0 or self.raw_lower >= self.raw_upper:
            raise ValueError("initial_sd must be positive and raw bounds ordered")
        if not self.raw_lower <= self.initial_mean <= self.raw_upper:
            raise ValueError("initial_mean must lie within the raw bounds")
        for name in ("natural_decay", "real_cost_log_sd", "cost_level_elasticity",
                     "baseline_penalty", "baseline_subsidy", "baseline_fixed_fine",
                     "natural_reversion", "wash_cost_log_sd", "reversion_permanent_share"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.reversion_permanent_share >= 1:
            raise ValueError("reversion_permanent_share must be below 1")
        if self.reversion_improvement not in ("transitory", "permanent"):
            raise ValueError("reversion_improvement must be 'transitory' or 'permanent'")
        if (self.reversion_permanent_share or self.reversion_improvement != "transitory") and not self.natural_reversion:
            raise ValueError("a permanent level or permanent improvement requires natural_reversion")
        for name in ("real_cost_scale", "wash_cost_ratio", "max_change_per_period"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.size_cost_multipliers) != 3 or min(self.size_cost_multipliers) <= 0:
            raise ValueError("size_cost_multipliers needs three positive values (large, mid, small)")
        for name in ("implementation_delay_days", "disclosure_delay_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("baseline_audit_probability", "baseline_transparency"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        unknown = set(self.action_labels) - {"improve", "wash", "maintain"}
        if unknown:
            raise ValueError(f"unknown action labels: {sorted(unknown)}")
        if self.initial_empirical_distribution:
            support, counts = zip(*self.initial_empirical_distribution)
            if any(type(count) is not int or count <= 0 for count in counts):
                raise ValueError("empirical frequencies must be positive integers")
            if len(set(support)) != len(support) or any(
                not math.isfinite(value) or not self.raw_lower <= value <= self.raw_upper
                for value in support
            ):
                raise ValueError("empirical support must be unique, finite and within raw bounds")
            mean = float(np.average(support, weights=counts))
            sd = float(np.sqrt(np.average((np.asarray(support) - mean)**2, weights=counts)))
            if not math.isclose(mean, self.initial_mean, rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(
                sd, self.initial_sd, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("initial_mean and initial_sd must match the empirical population moments")
        if self.observation_link not in OBSERVATION_LINKS:
            raise ValueError(f"observation_link must be one of {OBSERVATION_LINKS}")
        if self.observation_link == "empirical_quantile" and len(self.initial_empirical_distribution) < 2:
            raise ValueError("the empirical quantile link needs at least two empirical support values")
        if not -1 < self.initial_size_rank_correlation < 1:
            raise ValueError("initial_size_rank_correlation must be in (-1, 1)")
        if self.initial_size_rank_correlation and self.observation_link != "empirical_quantile":
            raise ValueError("initial_size_rank_correlation requires the empirical quantile link")
        if type(self.signal_saturation) is not bool:
            raise ValueError("signal_saturation must be boolean")
        if self.signal_saturation:
            if self.observation_link != "empirical_quantile":
                raise ValueError("signal_saturation requires the empirical quantile link")
            worst, best = sorted(self.to_goodness(np.array([self.raw_lower, self.raw_upper])))
            if best >= LATENT_GOODNESS_BOUND:
                raise ValueError("signal_saturation needs empirical mass at the best scale end")
            if worst > -LATENT_GOODNESS_BOUND:
                # Improvements would then become visible only above the floor, which the decision rules lack.
                raise ValueError("signal saturation at the worst scale end is not modelled")

    @property
    def goodness_bounds(self) -> tuple[float, float]:
        if self.observation_link == "empirical_quantile":
            return -LATENT_GOODNESS_BOUND, LATENT_GOODNESS_BOUND
        ends = sorted(self.to_goodness(np.array([self.raw_lower, self.raw_upper])))
        return float(ends[0]), float(ends[1])

    @property
    def observation_ceiling(self) -> float | None:
        """Goodness of the best raw end when signals saturate there, else None."""
        if not self.signal_saturation:
            return None
        return float(max(self.to_goodness(np.array([self.raw_lower, self.raw_upper]))))

    @property
    def report_bound(self) -> float:
        """Highest report a firm can make; also bounds the changes it chooses."""
        ceiling = self.observation_ceiling
        return self.goodness_bounds[1] if ceiling is None else ceiling

    def quantile_knots(self) -> tuple[FloatArray, FloatArray]:
        """Increasing raw knots and their latent normal scores (mid-rank CDF).

        Scale ends outside the observed support map to the latent bound; an
        observed scale end keeps its mid-rank score, so latent values beyond it
        saturate at that end.
        """
        support, counts = np.asarray(sorted(self.initial_empirical_distribution), dtype=np.float64).T
        cdf = (np.cumsum(counts) - counts / 2) / counts.sum()
        latent = np.array([NormalDist().inv_cdf(float(p)) for p in cdf])
        raw, scores = list(support), list(latent)
        if support[0] > self.raw_lower:
            raw.insert(0, self.raw_lower)
            scores.insert(0, -LATENT_GOODNESS_BOUND)
        if support[-1] < self.raw_upper:
            raw.append(self.raw_upper)
            scores.append(LATENT_GOODNESS_BOUND)
        return np.asarray(raw), np.asarray(scores)

    def to_goodness(self, raw: FloatArray) -> FloatArray:
        if self.observation_link == "empirical_quantile":
            knots, scores = self.quantile_knots()
            return self.direction * np.interp(np.asarray(raw, dtype=np.float64), knots, scores)
        return self.direction * (np.asarray(raw, dtype=np.float64) - self.initial_mean) / self.initial_sd

    def to_raw(self, goodness: FloatArray) -> FloatArray:
        if self.observation_link == "empirical_quantile":
            knots, scores = self.quantile_knots()
            return np.interp(self.direction * np.asarray(goodness, dtype=np.float64), scores, knots)
        return self.initial_mean + self.direction * self.initial_sd * np.asarray(goodness, dtype=np.float64)

    def label(self, action: str) -> str:
        return self.action_labels.get(action, action)

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: list(getattr(self, item.name)) if item.name == "size_cost_multipliers"
            else [list(pair) for pair in self.initial_empirical_distribution] if item.name == "initial_empirical_distribution"
            else getattr(self, item.name)
            for item in fields(self)
            if (item.name != "initial_empirical_distribution" or self.initial_empirical_distribution)
            and (item.name != "observation_link" or self.observation_link != "linear")
            and (item.name != "initial_size_rank_correlation" or self.initial_size_rank_correlation)
            and (item.name != "baseline_fixed_fine" or self.baseline_fixed_fine)
            and (item.name != "signal_saturation" or self.signal_saturation)
            and (item.name != "natural_reversion" or self.natural_reversion)
            and (item.name != "wash_cost_log_sd" or self.wash_cost_log_sd)
            and (item.name != "reversion_permanent_share" or self.reversion_permanent_share)
            and (item.name != "reversion_improvement" or self.reversion_improvement != "transitory")
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DomainSpec":
        known = {item.name for item in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown domain keys: {sorted(unknown)}")
        values = {key: value for key, value in payload.items() if not key.startswith("_")}
        values["size_cost_multipliers"] = tuple(float(value) for value in values["size_cost_multipliers"])
        if "initial_empirical_distribution" in values:
            values["initial_empirical_distribution"] = tuple(
                (float(value), count) for value, count in values["initial_empirical_distribution"]
            )
        return cls(**values)


def load_domain(path: str | Path) -> DomainSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("domain file root must be a JSON object")
    return DomainSpec.from_dict({key: value for key, value in payload.items() if not key.startswith("_")})
