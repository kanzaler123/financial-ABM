"""Auditable post-estimation helpers, independent of licensed data and simulation.

The distance-profile calculation implements contract 25.15's J*_profile - min J*.
It does not change the historical point estimator, fit test or simulation parameters.
The returned intervals are envelopes of accepted grid points, not interpolated
endpoints; the full grid and contiguous accepted segments remain available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from math import isfinite

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

CHI2_95_ONE_DF = 3.841458820694124


class OptimizationFailure(RuntimeError):
    """A minimum could not be established; no interval should be reported."""


def _candidates(theta: np.ndarray, bounds: np.ndarray, extra) -> np.ndarray:
    middle = bounds.mean(axis=1)
    points = [theta, middle]
    for j in range(theta.size):
        for edge in bounds[j]:
            point = middle.copy()
            point[j] = edge
            points.append(point)
    if theta.size <= 6:
        points.extend(np.array(point) for point in product(*bounds))
    if extra is not None:
        extra = np.asarray(extra, dtype=float)
        if extra.ndim != 2 or extra.shape[1] != theta.size or not np.isfinite(extra).all():
            raise ValueError("start_points must be a finite matrix with one column per parameter")
        points.extend(extra)
    return np.unique(np.clip(np.asarray(points), bounds[:, 0], bounds[:, 1]), axis=0)


def _minimum(objective: Callable, bounds: np.ndarray, candidates: np.ndarray, starts: int) -> dict:
    """Best successful bounded multistart fit, with explicit failure handling."""
    candidates = np.unique(np.clip(candidates, bounds[:, 0], bounds[:, 1]), axis=0)
    values = np.array([objective(point) for point in candidates])
    ranked = candidates[np.argsort(values, kind="stable")[:starts]]
    free = np.flatnonzero(bounds[:, 0] != bounds[:, 1])
    if not free.size:
        point = bounds[:, 0].copy()
        return {"x": point, "fun": objective(point), "attempts": 0, "failed_attempts": 0}

    def expand(x):
        point = bounds[:, 0].copy()
        point[free] = x
        return point

    best, attempts, failures = None, 0, 0
    lowest_failed = float("inf")
    for start in ranked:
        # Powell is a bounded fallback for line-search failures, not permission
        # to accept a failed optimizer's x as a valid interval endpoint.
        for method in ("L-BFGS-B", "Powell"):
            attempts += 1
            fit = minimize(lambda x: objective(expand(x)), start[free], method=method,
                           bounds=bounds[free], options={"maxiter": 2000, "ftol": 1e-12})
            point = expand(np.asarray(fit.x, dtype=float))
            finite = np.isfinite(point).all() and isfinite(float(fit.fun))
            inside = np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1])
            if fit.success and finite and inside:
                value = objective(point)
                if best is None or value < best["fun"]:
                    best = {"x": point, "fun": value}
                break
            failures += 1
            if finite and inside:
                lowest_failed = min(lowest_failed, objective(point))
    if best is None:
        raise OptimizationFailure("all bounded optimization attempts failed")
    known_lowest = min(float(values.min()), lowest_failed)
    tolerance = 1e-7 * max(1.0, abs(best["fun"]), abs(known_lowest))
    if best["fun"] > known_lowest + tolerance:
        raise OptimizationFailure("a failed fit or sampled point is better than every converged fit")
    return {**best, "attempts": attempts, "failed_attempts": failures}


def distance_profile_intervals(
    model: Callable[[np.ndarray], np.ndarray],
    theta: Sequence[float],
    target: Sequence[float],
    covariance: np.ndarray,
    *,
    bounds: Sequence[Sequence[float]],
    names: Sequence[str],
    start_points=None,
    grid_points: int = 61,
    threshold: float = CHI2_95_ONE_DF,
    starts: int = 10,
) -> dict:
    """Profile the first len(names) parameters using one fixed covariance.

    ``theta`` may minimize a different criterion. It supplies an initial point
    only: the reference minimum is recomputed with this function's full-covariance
    distance. Remaining parameters are nuisance parameters. Equal bounds fix a
    parameter; endpoints are included. The historical chi-square cutoff is retained
    as an auxiliary convention, not a new claim of boundary-adjusted coverage.

    Deterministic multistart optimization is a numerical search, not a certificate
    of a global optimum. Any lower constrained fit triggers another unrestricted
    search. Failed/nonfinite fits raise instead of manufacturing an interval.
    """
    theta, target = np.asarray(theta, dtype=float), np.asarray(target, dtype=float)
    bounds, covariance = np.asarray(bounds, dtype=float), np.asarray(covariance, dtype=float)
    if theta.ndim != 1 or not theta.size or not np.isfinite(theta).all():
        raise ValueError("theta must be a nonempty finite vector")
    if bounds.shape != (theta.size, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("bounds must contain finite ordered pairs for every parameter")
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise ValueError("theta is outside the parameter bounds")
    if not names or len(names) > theta.size or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("names must uniquely name the leading profiled parameters")
    if target.ndim != 1 or not target.size or not np.isfinite(target).all():
        raise ValueError("target must be a nonempty finite vector")
    if covariance.shape != (target.size, target.size) or not np.isfinite(covariance).all():
        raise ValueError("covariance must be a finite square matrix matching the target")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    if type(grid_points) is not int or grid_points < 2 or type(starts) is not int or starts < 1:
        raise ValueError("grid_points must be at least 2 and starts must be positive integers")
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")

    def distance(point):
        value = np.asarray(model(point), dtype=float)
        if value.shape != target.shape or not np.isfinite(value).all():
            raise ValueError("model moments must be finite and match the target")
        residual = np.linalg.solve(root, target - value)
        result = float(residual @ residual)
        if not isfinite(result):
            raise ValueError("distance is nonfinite")
        return result

    candidates = _candidates(theta, bounds, start_points)
    minimum = _minimum(distance, bounds, candidates, starts)
    order = np.argsort([distance(point) for point in candidates], kind="stable")
    profile_starts = candidates[order[:starts]]
    profiles, all_fits = {}, []
    for j, name in enumerate(names):
        rows, previous = [], minimum["x"]
        for value in np.unique(np.linspace(*bounds[j], grid_points)):
            fixed = bounds.copy()
            fixed[j] = value
            seeds = np.vstack([profile_starts, minimum["x"], previous])
            fit = _minimum(distance, fixed, seeds, starts)
            previous = fit["x"]
            rows.append({"value": float(value), "objective": fit["fun"], "parameters": fit["x"].tolist(),
                         "failed_attempts": fit["failed_attempts"]})
            all_fits.append(fit)
        profiles[name] = rows
    if all_fits and min(fit["fun"] for fit in all_fits) < minimum["fun"]:
        minimum = _minimum(distance, bounds, np.vstack([candidates, *[fit["x"] for fit in all_fits]]), starts)
    tolerance = 1e-7 * max(1.0, abs(minimum["fun"]))
    intervals, segments, boundary = {}, {}, {}
    for j, (name, rows) in enumerate(profiles.items()):
        accepted, groups, group = [], [], []
        for row in rows:
            delta = row["objective"] - minimum["fun"]
            if delta < -tolerance:
                raise OptimizationFailure("a profile fit is below the unrestricted reference minimum")
            row.update(delta=max(0.0, delta), accepted=bool(delta <= threshold))
            if row["accepted"]:
                accepted.append(row["value"])
                group.append(row["value"])
            elif group:
                groups.append([group[0], group[-1]])
                group = []
        if group:
            groups.append([group[0], group[-1]])
        intervals[name] = [accepted[0], accepted[-1]] if accepted else None
        segments[name] = groups
        boundary[name] = {"lower": bool(rows[0]["accepted"]), "upper": bool(rows[-1]["accepted"])}
    return {"criterion": "g' covariance^-1 g", "reference_parameters": minimum["x"].tolist(),
            "minimum": minimum["fun"], "distance_at_original_estimate": distance(theta),
            "original_estimate": theta.tolist(), "threshold": threshold, "grid_points": grid_points,
            "intervals": intervals, "accepted_segments": segments, "touches_parameter_bounds": boundary,
            "profiles": profiles, "reference_failed_attempts": minimum["failed_attempts"],
            "coverage": "auxiliary grid profile; historical cutoff; no boundary or misspecification adjustment"}


def paired_mean_difference(plus: Mapping[int, float], minus: Mapping[int, float], seeds: Sequence[int]) -> dict:
    """Unadjusted 95% paired t interval on an explicit, complete seed set."""
    seeds = list(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct paired seeds are required")
    missing = [seed for seed in seeds if seed not in plus or seed not in minus]
    if missing:
        raise ValueError(f"incomplete pairs for seeds {missing}")
    values = np.asarray([plus[seed] - minus[seed] for seed in seeds], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf(0.975, len(values) - 1) * se)
    return {"mean": mean, "standard_error": se, "ci_low": mean - half, "ci_high": mean + half,
            "n": len(values), "seeds": seeds, "inference": "descriptive, unadjusted 95% paired t interval"}
