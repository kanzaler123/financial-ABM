"""Synthetic-data check of the contract 28.37 estimator (estimate_k3.py sizeaudit) before it reads any model run.

    .venv/Scripts/python.exe -I -X utf8 reports/disclosure_enforcement_costs_20261029/selftest_sizeaudit_k3.py

Each synthetic cell draws, for 30 seeds, the number of small (120 firms x 8 years) and large (20 x 8) firm-years
with a detection from binomials with known yearly rates (generator seed 28908, used only here). Checked against
calculations made here independently: the pooled m9; the bootstrap resampling, which must be that of
moments_k3.seed_bootstrap_cov; the percentile interval (inverse empirical distribution); the sign and data-interval
classes, the point rule and the threshold; the pairing of the public minus hidden draws; and monotonicity and
interpolation on a hand-made path. Writes selftest_sizeaudit_k3.json (LF).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ek, mk = _load("estimate_k3"), _load("moments_k3")
SEEDS, SMALL_YEARS, LARGE_YEARS = 30, 120 * 8, 20 * 8
DATA_CI = [1.052664586430521, 1.7798659311792209]          # k1_size_correction.json, T1-or-T2 ln ratio
# Yearly detection rates (small, large) and the classes they are built to fall in.
CELLS = {"A": ((0.04, 0.01), "clearly_positive", "cannot_exclude"),
         "B": ((0.025, 0.04), "clearly_negative", "clearly_below"),
         "C": ((0.05, 0.002), "clearly_positive", "clearly_above"),
         "D": ((0.03, 0.03), "sign_undetermined", "clearly_below"),
         "E": ((0.03, 0.0), "undefined", "undefined")}


class FakeStats:
    """What seed_bootstrap_cov reads from moments_k3.Stats: moments() for m1-m8 (zeros here) and the ratio sums."""

    def __init__(self, sums: np.ndarray):
        self.ratio = sums[None, :]

    def moments(self) -> np.ndarray:
        return np.zeros(9)


def main() -> None:
    rng = np.random.default_rng(28908)
    per_seed = {}
    for name, ((p_small, p_large), _, _) in CELLS.items():
        per_seed[name] = np.array([[rng.binomial(SMALL_YEARS, p_small), SMALL_YEARS,
                                    rng.binomial(LARGE_YEARS, p_large), LARGE_YEARS] for _ in range(SEEDS)], float)
    per_seed["A_copy"] = per_seed["A"].copy()
    boot = ek.bootstrap_m9(per_seed, mk)
    checks, cells = {}, {}
    # The resampled seeds: the same generator and calls as seed_bootstrap_cov, so the m9 variance must agree.
    cov = mk.seed_bootstrap_cov([FakeStats(row) for row in per_seed["A"]])
    checks["bootstrap_matches_seed_bootstrap_cov"] = bool(math.isclose(cov[8, 8], boot["A"].var(ddof=1), rel_tol=1e-12))
    picks_rng = np.random.default_rng(28906)
    picks = [picks_rng.integers(0, SEEDS, SEEDS) for _ in range(1000)]
    for name, (_, sign, data) in CELLS.items():
        cs, ys, cl, yl = per_seed[name].sum(0)
        with np.errstate(divide="ignore"):
            m9_hand = float(np.log((cs / ys) / (cl / yl)))
        m9 = mk.log_ratio(per_seed[name])
        draws_hand = []
        for pick in picks:
            s = per_seed[name][pick].sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                draws_hand.append(float(np.log((s[0] / s[1]) / (s[2] / s[3]))))
        draws_hand = np.array(draws_hand)
        valid = np.sort(draws_hand[~np.isnan(draws_hand)])
        # The smallest draw whose empirical distribution function reaches 2.5% (97.5%), in exact arithmetic.
        ci_hand = [float(valid[math.ceil(Fraction(1, 40) * valid.size) - 1]),
                   float(valid[math.ceil(Fraction(39, 40) * valid.size) - 1])]
        interval = ek.percentile_interval(boot[name])
        classes = ek.classify_m9(m9, interval["ci95"], DATA_CI)
        if math.isfinite(m9_hand):
            sign_hand = ("clearly_positive" if ci_hand[0] > 0 else "clearly_negative" if ci_hand[1] < 0
                         else "sign_undetermined")
            data_hand = ("clearly_below" if ci_hand[1] < DATA_CI[0] else "clearly_above" if ci_hand[0] > DATA_CI[1]
                         else "cannot_exclude")
        else:
            sign_hand = data_hand = "undefined"
        cells[name] = {"rates": CELLS[name][0], "true_m9": (math.log(CELLS[name][0][0] / CELLS[name][0][1])
                                                            if CELLS[name][0][1] else math.inf),
                       "m9": m9, "ci95": interval["ci95"], "se": interval["se"], **classes,
                       "large_cases": int(cl)}
        checks[f"{name}_m9"] = m9 == m9_hand
        checks[f"{name}_draws"] = bool(np.array_equal(boot[name], draws_hand, equal_nan=True))
        checks[f"{name}_ci95"] = interval["ci95"] == ci_hand
        checks[f"{name}_classes_by_hand"] = (classes["sign"], classes["data"]) == (sign_hand, data_hand)
        checks[f"{name}_classes_as_built"] = (classes["sign"], classes["data"]) == (sign, data)
        if math.isfinite(m9):
            checks[f"{name}_point_rule"] = classes["mean_in_data_ci"] == (DATA_CI[0] <= m9 <= DATA_CI[1])
    # Threshold: clearly positive and cannot exclude, beta != 0 only.
    rows = [{"name": n, "beta": -0.5, **{k: cells[n][k] for k in ("sign", "data")}} for n in CELLS]
    rows.append({"name": "A_at_beta0", "beta": 0.0, "sign": cells["A"]["sign"], "data": cells["A"]["data"]})
    checks["threshold_cells"] = ek.threshold_cells(rows) == ["A"]
    # Pairing: two arms with identical seeds differ by exactly zero in every draw.
    with np.errstate(invalid="ignore"):
        diff = ek.percentile_interval(boot["A"] - boot["A_copy"])
    checks["paired_difference_of_identical_cells_is_zero"] = diff["ci95"] == [0.0, 0.0] and diff["nan_draws"] == 0
    # A cell with no detection in either group in every draw: no interval, undefined class (no crash).
    empty = ek.percentile_interval(np.full(1000, np.nan))
    checks["all_nan_draws_no_interval"] = (empty["ci95"] is None and empty["se"] is None and empty["nan_draws"] == 1000
                                           and ek.classify_m9(math.nan, empty["ci95"], DATA_CI)["sign"] == "undefined")
    # Monotonicity and interpolation on a hand-made path.
    betas = [0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5]
    path = ek.arm_path(betas, [-0.5, 0.0, 0.6, 1.2, 1.1, 2.0, 2.5],
                       {"zero": 0.0, "data_ci_low": DATA_CI[0], "data_ci_high": DATA_CI[1]})
    expected = {"zero": [-0.25], "data_ci_low": [-0.5 - 0.25 * (DATA_CI[0] - 0.6) / 0.6],
                "data_ci_high": [-1.0 - 0.25 * (DATA_CI[1] - 1.1) / 0.9]}
    checks["rising_steps"] = [s["rising"] for s in path["steps"]] == [True, True, True, False, True, True]
    checks["all_steps_rising_false"] = path["all_steps_rising"] is False
    checks["crossings"] = all(len(path["crossings_on_rising_steps"][k]) == len(v) and
                              all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(path["crossings_on_rising_steps"][k], v))
                              for k, v in expected.items())
    out = {"contract": "28.37", "synthetic_seed": 28908, "bootstrap_seed": 28906, "data_ci": DATA_CI,
           "cells": cells, "path": path, "checks": checks, "passed": all(checks.values())}
    (HERE / "selftest_sizeaudit_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"passed": out["passed"], "failed": [k for k, v in checks.items() if not v],
                      "cells": {n: {k: c[k] for k in ("m9", "ci95", "sign", "data", "mean_in_data_ci")}
                                for n, c in cells.items()}}, indent=1))
    if not out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
