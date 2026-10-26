"""Check estimate.py on synthetic panels with known dynamics, before any model run (contract 25.15).

  python selftest.py

Replaces the model by a statistical stand-in on the same grid: a permanent level with variance s plus an AR(1) part
with quarterly persistence exp(-kappa / 4) (Q; R is s = 0), and for U a slowly drifting rank order whose quarterly
innovation variance rises with delta. Pseudo-data come from Q at (kappa_T, s_P, sigma_m) = (0.9, 0.35, 0.1) with
4,000 firms over 49 dates. Passing means: Q is selected, its estimate lies within 0.15 of each true value, and the
true values lie inside the reported 95% intervals. Writes selftest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import estimate as est

HERE = Path(__file__).resolve().parent
TRUTH = {"kappa_T": 0.9, "s_P": 0.35, "sigma_m": 0.1}


def panel(rng: np.random.Generator, firms: int, dates: int, kappa: float, share: float, drift: float = 0.0) -> np.ndarray:
    phi = np.exp(-kappa / 4)
    mu = np.sqrt(share) * rng.standard_normal(firms)
    x = np.sqrt(1 - share) * rng.standard_normal(firms)
    out = np.empty((dates, firms))
    for t in range(dates):
        if t:
            x = phi * x + np.sqrt((1 - phi**2) * (1 - share)) * rng.standard_normal(firms)
            mu = mu + drift * rng.standard_normal(firms)
        out[t] = mu + x
    return out


def stand_in(form: str, values: tuple, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, hash(form) % 1000, *[int(v * 1000) for v in values]])
    if form == "U":
        return panel(rng, 200, 36, 0.0, 1.0 - 1e-9, drift=0.05 + values[0])
    if form == "R":
        return panel(rng, 200, 36, values[0], 0.0)
    return panel(rng, 200, 36, values[0], values[1])


def main() -> None:
    seeds = list(range(48001, 48011))
    grid = {}
    for form, (_, axes, _) in est.FORMS.items():
        shape = tuple(len(axis) for axis in axes)
        values = np.empty(shape + (len(est.SIGMAS), len(seeds), len(est.LAGS)))
        for index in np.ndindex(*shape):
            point = tuple(axis[i] for axis, i in zip(axes, index))
            values[index] = est.point_moments({seed: stand_in(form, point, seed) for seed in seeds})
        grid[form] = values
    rng = np.random.default_rng(7)
    data = panel(rng, 4000, 49, TRUTH["kappa_T"], TRUTH["s_P"]) + TRUTH["sigma_m"] * rng.standard_normal((49, 4000))
    target, cov = est.pseudo_moments({0: data}, replications=300)
    result = est.estimate_all(grid, target, cov)
    q = result["forms"]["Q"]
    close = {name: bool(abs(q["estimate"][name] - value) <= 0.15) for name, value in TRUTH.items()}
    covered = {name: bool(q["ci95"][name][0] <= value <= q["ci95"][name][1]) for name, value in TRUTH.items()}
    passed = bool(result["selected"] == "Q" and all(close.values()) and all(covered.values()))
    out = {"truth": TRUTH, "selected": result["selected"], "fitting_forms": result["fitting_forms"],
           "estimates": {name: row["estimate"] for name, row in result["forms"].items()},
           "T_p_values": {name: row["T_p_value"] for name, row in result["forms"].items()},
           "Q_ci95": q["ci95"], "close": close, "covered": covered, "passed": passed}
    (HERE / "selftest.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    if not passed:
        raise SystemExit("estimator self-test failed")


if __name__ == "__main__":
    main()
