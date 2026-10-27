"""Stage I estimates: the Huazheng score and the cost of debt (contract section 26).

    python -I -X utf8 estimate.py --selftest    the estimator on synthetic panels with a known coefficient
    python -I -X utf8 estimate.py --smoke       every specification on the real panel, the score permuted within year
    python -I -X utf8 estimate.py               the pre-registered estimates (only after section 26 is committed)

Reads the panels written by build_panel.py (data/processed/cnrds_stage_i_20261027/, not committed). OLS with
absorbed fixed effects (alternating projections) and standard errors clustered by firm: CR1 with the factor
G/(G - 1) (N - 1)/(N - K), where K counts the regressors and the levels of absorbed effects not nested in the
clusters; 95% intervals use t with G - 1 degrees of freedom. Singletons are dropped first, repeatedly. Outcomes
are in basis points, so the score coefficient is basis points per SD of the score and the premium is minus it.

The smoke run replaces the score by a seeded permutation within each year, so it exercises the samples, fixed
effects and clustering of every specification without estimating any relation between the score and an outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT / "data/processed/cnrds_stage_i_20261027"
CONTROLS = ["size_w", "lev_w", "roa_w"]
EXTENDED = ["tang_w", "cash_w", "growth_w", "age_w"]
SMOKE_SEED = 26001
REPS = 1000                       # self-test replications per fixed-effect design


# -- estimator ---------------------------------------------------------------------------------------------------
def codes(values) -> np.ndarray:
    return pd.factorize(pd.Series(values).astype(str), sort=True)[0]


def drop_singletons(fes: list[np.ndarray]) -> np.ndarray:
    keep = np.ones(len(fes[0]) if fes else 0, dtype=bool)
    while fes:
        before = keep.sum()
        for g in fes:
            keep &= np.bincount(g, weights=keep.astype(float))[g] > 1
        if keep.sum() == before:
            break
    return keep


def demean(M: np.ndarray, fes: list[np.ndarray], tol: float = 1e-13, max_iter: int = 100_000) -> np.ndarray:
    """Residuals of M on the fixed effects: exact for one set, alternating projections for several."""
    M = M.astype(float).copy()
    counts = [np.bincount(g) for g in fes]
    scale = max(np.abs(M).max(), 1.0)
    for it in range(max_iter):
        change = 0.0
        for g, n in zip(fes, counts):
            for j in range(M.shape[1]):
                means = np.bincount(g, weights=M[:, j]) / n
                M[:, j] -= means[g]
                change = max(change, float(np.abs(means).max()))
        if len(fes) <= 1 or change < tol * scale:
            return M
    raise RuntimeError("fixed effects did not converge")


@dataclass
class Fit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    n: int
    clusters: int
    k: int
    absorbed: int
    singletons: int
    r2_within: float
    fe_levels: dict = field(default_factory=dict)

    def row(self, j: int = 0) -> dict:
        t = self.beta[j] / self.se[j]
        crit = stats.t.ppf(0.975, self.clusters - 1)
        return {"coef": float(self.beta[j]), "se": float(self.se[j]), "t": float(t),
                "p": float(2 * stats.t.sf(abs(t), self.clusters - 1)),
                "ci95": [float(self.beta[j] - crit * self.se[j]), float(self.beta[j] + crit * self.se[j])]}


def fe_ols(df: pd.DataFrame, y: str, xs: list[str], fes: list[str], cluster: str, nested: tuple[str, ...] = ()) -> Fit:
    d = df.dropna(subset=[y, *xs, *fes, cluster])
    fe_codes = [codes(d[f]) for f in fes]
    keep = drop_singletons(fe_codes)
    d = d[keep]
    fe_codes = [codes(d[f]) for f in fes]
    M = demean(d[[y, *xs]].to_numpy(float), fe_codes)
    yy, X = M[:, 0], M[:, 1:]
    cl = codes(d[cluster])
    n, k, G = len(yy), X.shape[1], cl.max() + 1
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ yy
    e = yy - X @ beta
    scores = np.zeros((G, k))
    np.add.at(scores, cl, X * e[:, None])
    levels = {f: int(c.max() + 1) for f, c in zip(fes, fe_codes)}
    absorbed = sum(v for f, v in levels.items() if f not in nested)
    factor = G / (G - 1) * (n - 1) / (n - k - absorbed)
    V = factor * XtX_inv @ (scores.T @ scores) @ XtX_inv
    return Fit(xs, beta, np.sqrt(np.diag(V)), n, int(G), k, absorbed, int((~keep).sum()),
               float(1 - e @ e / (yy @ yy)), levels)


# -- specifications ------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    name: str
    role: str                     # primary, robustness or secondary
    describe: str
    x: str = "ns"
    y: str = "y_bp"
    controls: list[str] = field(default_factory=lambda: list(CONTROLS))
    fes: list[str] = field(default_factory=lambda: ["ind_year"])
    sample: Callable[[pd.DataFrame], pd.Series] | None = None
    data: str = "firm_year"
    cluster: str = "Scode"


def debt_specs(ownership: bool) -> list[Spec]:
    soe = ["soe"] if ownership else []
    base = CONTROLS + soe
    return [
        Spec("main", "primary", "debt cost of t+1 on the October score of t; size, leverage, ROA" +
             (", state ownership" if ownership else "") + "; industry x year effects", controls=base),
        Spec("firm_fe", "robustness", "adds firm fixed effects", controls=base, fes=["Scode", "ind_year"]),
        Spec("regime_R1", "robustness", "score dates 2009-10 to 2020-10 (first rating method)", controls=base,
             sample=lambda d: d.regime == "R1"),
        Spec("regime_R2", "robustness", "score dates 2021-10 to 2023-10 (second method)", controls=base,
             sample=lambda d: d.regime == "R2"),
        Spec("regime_R3", "robustness", "score date 2024-10 (third method)", controls=base,
             sample=lambda d: d.regime == "R3"),
        Spec("zscore", "robustness", "score standardised by mean and SD within date", x="z", controls=base),
        Spec("year_average", "robustness", "mean of the four quarterly normal scores of t", x="ns_avg", controls=base),
        Spec("outcome_to_2018", "robustness", "outcome years 2010-2018 (note-based interest, no lease standard)",
             controls=base, sample=lambda d: d.year + 1 <= 2018),
        Spec("outcome_from_2019", "robustness", "outcome years 2019-2025 (income-statement interest)", controls=base,
             sample=lambda d: d.year + 1 >= 2019),
        Spec("no_lease", "robustness", "denominator without lease liabilities", y="y_nolease_bp", controls=base),
        Spec("no_real_estate", "robustness", "without real estate (CSRC K70)", controls=base,
             sample=lambda d: d.IndcodeB != "K70"),
        Spec("debt_5pct", "robustness", "mean interest-bearing debt at least 5% of assets", controls=base,
             sample=lambda d: d.debt_to_assets >= 0.05),
        Spec("extended_controls", "robustness", "adds tangibility, cash, sales growth, log listing age",
             controls=base + EXTENDED),
        Spec("csi300", "robustness", "CSI 300 constituents of 2026-10-09 only", controls=base, sample=lambda d: d.csi300),
        Spec("grade_notches", "robustness", "score as rating notches (C = 1 ... AAA = 9), per notch", x="grade_num",
             controls=base),
    ]


def load_firm_year(smoke: bool) -> tuple[pd.DataFrame, bool]:
    d = pd.read_csv(PANEL / "firm_year.csv.gz", dtype={"Scode": str, "IndcodeB": str, "industry": str})
    d = d[d.main_sample].copy()
    d["ind_year"] = d.industry + "_" + d.year.astype(str)
    d["y_bp"] = d.debt_cost_w * 1e4
    d["y_nolease_bp"] = d.debt_cost_nolease_w * 1e4
    ownership = "soe" in d.columns and d.soe.notna().any()
    if smoke:
        rng = np.random.default_rng(SMOKE_SEED)
        perm = d.groupby("year").indices
        for col in ("ns", "z", "ns_avg", "grade_num"):
            v = d[col].to_numpy().copy()
            for idx in perm.values():
                v[idx] = v[rng.permutation(idx)]
            d[col] = v
    return d, ownership


def run(specs: list[Spec], data: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for s in specs:
        d = data[s.data]
        if s.sample is not None:
            d = d[s.sample(d).fillna(False)]
        fit = fe_ols(d, s.y, [s.x, *s.controls], s.fes, s.cluster, nested=(s.cluster,))
        r = fit.row(0)
        out.append({"name": s.name, "role": s.role, "describe": s.describe, "x": s.x, "y": s.y, "controls": s.controls,
                    "fixed_effects": s.fes, "n": fit.n, "clusters": fit.clusters, "singletons_dropped": fit.singletons,
                    "fe_levels": fit.fe_levels, "absorbed_df": fit.absorbed, "r2_within": fit.r2_within,
                    "score_coef_bp_per_sd": r, "premium_bp_per_sd": -r["coef"],
                    "controls_coef": {c: fit.row(j + 1) for j, c in enumerate(s.controls)}})
    return out


# -- self-test -------------------------------------------------------------------------------------------------------
def synthetic(rng: np.random.Generator, firms: int, years: int, industries: int, beta: float,
              confound: bool) -> pd.DataFrame:
    """A firm effect in the outcome (always) that is correlated with the score only if confound is set, so that
    industry x year effects alone are a correct specification exactly when confound is False."""
    firm = np.repeat(np.arange(firms), years)
    year = np.tile(np.arange(years), firms)
    ind = rng.integers(0, industries, firms)[firm]
    keep = rng.random(firm.size) < 0.85                                   # unbalanced
    a = rng.normal(0, 1, firms)
    x = (0.6 * a[firm] if confound else 0.6 * rng.normal(0, 1, firms)[firm]) + rng.normal(0, 1, firm.size)
    for t in range(1, years):                                             # persistent score
        x[year == t] = 0.7 * x[year == t - 1] + 0.3 * x[year == t]
    c1 = 0.5 * x + rng.normal(0, 1, firm.size)
    iy = rng.normal(0, 30, (industries, years))[ind, year]
    u = np.zeros(firm.size)
    eps = rng.normal(0, 40, firm.size)
    for t in range(years):                                                # serially correlated errors within firm
        u[year == t] = (0.6 * u[year == t - 1] if t else 0) + eps[year == t]
    y = 450 + beta * x + 20 * c1 + 25 * a[firm] + iy + u
    d = pd.DataFrame({"Scode": firm, "year": year, "ind": ind, "x": x, "c1": c1, "y": y})[keep]
    d["ind_year"] = d.ind.astype(str) + "_" + d.year.astype(str)
    return d


def dummy_ols(d: pd.DataFrame, y: str, xs: list[str], fes: list[str], nested: tuple[str, ...]) -> tuple:
    """Brute force: explicit dummies, the same CR1 factor (nested dummies not counted in K)."""
    blocks = [d[xs].to_numpy(float)]
    for f in fes:
        D = pd.get_dummies(d[f].astype(str), drop_first=bool(len(blocks) > 1)).to_numpy(float)
        blocks.append(D)
    X = np.hstack(blocks)
    yy = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ beta
    cl = codes(d.Scode)
    G, n = cl.max() + 1, len(yy)
    XtX_inv = np.linalg.pinv(X.T @ X)
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, cl, X * e[:, None])
    k_eff = len(xs) + sum(d[f].nunique() for f in fes if f not in nested)
    V = G / (G - 1) * (n - 1) / (n - k_eff) * XtX_inv @ (scores.T @ scores) @ XtX_inv
    return beta[: len(xs)], np.sqrt(np.diag(V))[: len(xs)]


def selftest() -> dict:
    rng = np.random.default_rng(26002)
    out = {"true_beta": -8.0, "exact_vs_dummies": [], "coverage": {}}
    for fes in (["ind_year"], ["Scode", "ind_year"]):
        d = synthetic(rng, 150, 6, 5, -8.0, confound=True)
        keep = drop_singletons([codes(d[f]) for f in fes])
        d = d[keep]
        fit = fe_ols(d, "y", ["x", "c1"], fes, "Scode", nested=("Scode",))
        b, se = dummy_ols(d, "y", ["x", "c1"], fes, nested=("Scode",))
        out["exact_vs_dummies"].append({"fixed_effects": fes, "max_abs_beta_diff": float(np.abs(fit.beta - b).max()),
                                        "max_rel_se_diff": float(np.abs(fit.se / se - 1).max())})
        assert np.allclose(fit.beta, b, atol=1e-8) and np.allclose(fit.se, se, rtol=1e-6), (fit.beta, b, fit.se, se)
    for fes in (["ind_year"], ["Scode", "ind_year"]):
        hits, est = 0, []
        for _ in range(REPS):
            d = synthetic(rng, 1500, 12, 30, -8.0, confound=len(fes) > 1)
            r = fe_ols(d, "y", ["x", "c1"], fes, "Scode", nested=("Scode",)).row(0)
            hits += r["ci95"][0] <= -8.0 <= r["ci95"][1]
            est.append(r["coef"])
        out["coverage"]["+".join(fes)] = {"replications": REPS, "ci95_coverage": hits / REPS,
                                          "mean_estimate": float(np.mean(est)), "sd_estimate": float(np.std(est))}
        assert 0.93 <= hits / REPS <= 0.97 and abs(np.mean(est) + 8.0) < 0.2, out["coverage"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        result = selftest()
        name = "selftest.json"
    else:
        firm_year, ownership = load_firm_year(smoke=args.smoke)
        specs = debt_specs(ownership)
        result = {"mode": "smoke: score permuted within year (seed %d); no score-outcome relation" % SMOKE_SEED
                  if args.smoke else "final", "ownership_control": ownership,
                  "estimates": run(specs, {"firm_year": firm_year})}
        name = "smoke.json" if args.smoke else "estimates.json"
    (HERE / name).write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if args.selftest:
        print(json.dumps(result, indent=1))
    else:
        print(json.dumps([{k: r[k] for k in ("name", "n", "clusters", "singletons_dropped", "fe_levels")}
                          for r in result["estimates"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
