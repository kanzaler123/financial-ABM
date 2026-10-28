"""Stage J estimators (contract 27): returns after Huazheng rating changes (J1) and year-end fund ownership (J2).

    python -I -X utf8 estimate.py --smoke      # rating changes permuted within date or year: sizes and SEs only

The smoke run permutes the rating change within each rating date (J1) or year (J2) with seed 27901 before any
estimation, so it carries no information about the relation between rating changes and returns or holdings. It
records sample sizes, cluster counts and fixed-effect cells, and the standard errors under the permutation, from
which the minimum detectable effects follow. The --final mode is added after contract 27 is approved.

J1: car_1_2 (months +1 and +2 after the rating-date month, percent) on the notch change, with size, log book-to-
market, the return of the rating quarter (months -2..0), momentum (months -11..-3) and the previous rating as
controls and industry fixed effects within each rating date. Fama-MacBeth over rating dates (t with D - 1 degrees of
freedom); pooled OLS with date x industry effects and two-way (firm, date) clustered errors as a robustness check.
J2: change in year-end fund ownership (percentage points of tradable shares) on the October-to-October notch change,
with size, log book-to-market, the return of November y-1 to October y and the ownership at December y-1 as controls
and year x industry effects; firm-clustered (CR1) errors, t with G - 1 degrees of freedom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANELS = ROOT / "data/processed/stage_j_20261028"
SMOKE_SEED = 27901
NULL_SEED = 27903
WINSOR = (0.01, 0.99)
J1_CONTROLS = ["size", "log_bm", "ret_q", "mom", "notch_prev"]
J2_CONTROLS = ["size", "log_bm", "ret_12", "fund_own_prev"]


def winsorize_within(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        b = df.groupby(by)[c].quantile(list(WINSOR)).unstack()
        df[c] = df[c].clip(df[by].map(b[WINSOR[0]]), df[by].map(b[WINSOR[1]]))
    return df


def demean(df: pd.DataFrame, cols: list[str], group: pd.Series) -> np.ndarray:
    x = df[cols].to_numpy(float)
    codes = pd.factorize(group)[0]
    sums = np.zeros((codes.max() + 1, x.shape[1]))
    np.add.at(sums, codes, x)
    counts = np.bincount(codes)
    return x - (sums / counts[:, None])[codes]


def drop_singletons(df: pd.DataFrame, cell: pd.Series) -> pd.DataFrame:
    size = cell.map(cell.value_counts())
    return df[size > 1]


def ols_cluster(y: np.ndarray, X: np.ndarray, clusters: list[np.ndarray], n_fe: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS on demeaned data; CR1 cluster-robust covariance, two-way by inclusion-exclusion when two clusterings."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta

    def meat(codes: np.ndarray) -> tuple[np.ndarray, int]:
        g = pd.factorize(codes)[0]
        s = np.zeros((g.max() + 1, k))
        np.add.at(s, g, X * u[:, None])
        G = g.max() + 1
        return s.T @ s, G

    def cr1(m: np.ndarray, G: int) -> np.ndarray:
        return G / (G - 1) * (n - 1) / (n - k - n_fe) * xtx_inv @ m @ xtx_inv

    if len(clusters) == 1:
        m, G = meat(clusters[0])
        return beta, cr1(m, G)
    (m1, g1), (m2, g2) = meat(clusters[0]), meat(clusters[1])
    inter = pd.factorize(pd.Series(clusters[0]).astype(str) + "|" + pd.Series(clusters[1]).astype(str))[0]
    m12, g12 = meat(inter)
    v = cr1(m1, g1) + cr1(m2, g2) - cr1(m12, g12)
    return beta, v


def fama_macbeth(df: pd.DataFrame, y: str, x: str, controls: list[str], date: str, fe: str) -> dict:
    slopes = []
    for d, g in df.groupby(date):
        g = drop_singletons(g, g[fe])
        Z = demean(g, [y, x, *controls], g[fe])
        b = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
        slopes.append((d, float(b[0]), int(len(g))))
    b = np.array([s[1] for s in slopes])
    D = len(b)
    se = float(b.std(ddof=1) / np.sqrt(D))
    return {"dates": D, "obs": int(sum(s[2] for s in slopes)), "mean": float(b.mean()), "se": se, "df": D - 1,
            "per_date": {s[0]: {"slope": s[1], "n": s[2]} for s in slopes}}


def pooled(df: pd.DataFrame, y: str, x: str, controls: list[str], cell: pd.Series, clusters: list[str]) -> dict:
    df = drop_singletons(df, cell.loc[df.index])
    c = cell.loc[df.index]
    Z = demean(df, [y, x, *controls], c)
    n_fe = int(c.nunique())
    beta, v = ols_cluster(Z[:, 0], Z[:, 1:], [df[k].to_numpy() for k in clusters], n_fe)
    G = min(int(df[k].nunique()) for k in clusters)
    return {"obs": int(len(df)), "fe_cells": n_fe, "clusters": {k: int(df[k].nunique()) for k in clusters},
            "coef": float(beta[0]), "se": float(np.sqrt(v[0, 0])), "df": G - 1}


def mde(se: float, df: int, alpha: float, power: float = 0.8) -> float:
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se)


def load_j1(period: str) -> pd.DataFrame:
    e = pd.read_csv(PANELS / "events.csv.gz", dtype={"Scode": str})
    e = e[e.in_sample & (e.period == period)].copy()
    e["car"] = 100 * e.car_1_2
    e = winsorize_within(e, ["car", "size", "log_bm", "ret_q", "mom"], "date")
    e["cell"] = e.date + "|" + e.industry
    return e


def load_j2(period: str) -> pd.DataFrame:
    a = pd.read_csv(PANELS / "annual_fund_ownership.csv.gz", dtype={"Scode": str})
    a = a[a.in_sample & (a.period == period)].copy()
    a = winsorize_within(a, ["dfund_own", "size", "log_bm", "ret_12", "fund_own_prev"], "year")
    a["cell"] = a.year.astype(str) + "|" + a.industry
    return a


def smoke() -> dict:
    rng = np.random.default_rng(SMOKE_SEED)
    out: dict = {"seed": SMOKE_SEED, "note": "rating changes permuted within date (J1) or year (J2); only sizes, "
                 "cluster counts and standard errors are read"}
    for period in ("pit", "prelaunch", "restated"):
        e = load_j1(period)
        e["dnotch"] = e.groupby("date").dnotch.transform(lambda s: rng.permutation(s.to_numpy()))
        e["dns"] = e.groupby("date").dns.transform(lambda s: rng.permutation(s.to_numpy()))
        fm = fama_macbeth(e, "car", "dnotch", J1_CONTROLS, "date", "industry")
        fm_ns = fama_macbeth(e, "car", "dns", J1_CONTROLS, "date", "industry")
        po = pooled(e, "car", "dnotch", J1_CONTROLS, e.cell, ["Scode", "date"])
        out[f"J1_{period}"] = {
            "events": int(len(e)), "firms": int(e.Scode.nunique()), "dates": int(e.date.nunique()),
            "fama_macbeth_dnotch": {"dates": fm["dates"], "obs": fm["obs"], "se": fm["se"], "df": fm["df"],
                                    "mde_alpha_0.05": mde(fm["se"], fm["df"], 0.05),
                                    "mde_alpha_0.0125": mde(fm["se"], fm["df"], 0.0125)},
            "fama_macbeth_dns": {"se": fm_ns["se"], "mde_alpha_0.05": mde(fm_ns["se"], fm_ns["df"], 0.05)},
            "pooled_two_way": {k: po[k] for k in ("obs", "fe_cells", "clusters", "se", "df")},
        }
        a = load_j2(period)
        a["dnotch"] = a.groupby("year").dnotch.transform(lambda s: rng.permutation(s.to_numpy()))
        p2 = pooled(a, "dfund_own", "dnotch", J2_CONTROLS, a.cell, ["Scode"])
        out[f"J2_{period}"] = {"firm_years": int(len(a)), "years": sorted(int(y) for y in a.year.unique()),
                               **{k: p2[k] for k in ("obs", "fe_cells", "clusters", "se", "df")},
                               "mde_alpha_0.05": mde(p2["se"], p2["df"], 0.05),
                               "mde_alpha_0.0125": mde(p2["se"], p2["df"], 0.0125)}
    return out


def null_check(reps: int) -> dict:
    """Size of each inference method under the sharp null: the rating change permuted within date (J1) or year
    (J2), reps times. Records the spread of the standard errors and the share of two-sided rejections at 5%; the
    permuted coefficients themselves are not kept."""
    rng = np.random.default_rng(NULL_SEED)
    out: dict = {"seed": NULL_SEED, "reps": reps}
    e = load_j1("pit")
    a = load_j2("pit")
    rows = {"J1_fama_macbeth": [], "J1_pooled_two_way": [], "J1_pooled_firm": [], "J1_pooled_date": [],
            "J2_pooled_firm": []}
    for _ in range(reps):
        e["x"] = e.groupby("date").dnotch.transform(lambda s: rng.permutation(s.to_numpy()))
        fm = fama_macbeth(e, "car", "x", J1_CONTROLS, "date", "industry")
        rows["J1_fama_macbeth"].append((fm["mean"] / fm["se"], fm["se"], fm["df"]))
        for name, cl in (("J1_pooled_two_way", ["Scode", "date"]), ("J1_pooled_firm", ["Scode"]),
                         ("J1_pooled_date", ["date"])):
            p = pooled(e, "car", "x", J1_CONTROLS, e.cell, cl)
            rows[name].append((p["coef"] / p["se"], p["se"], p["df"]))
        a["x"] = a.groupby("year").dnotch.transform(lambda s: rng.permutation(s.to_numpy()))
        p = pooled(a, "dfund_own", "x", J2_CONTROLS, a.cell, ["Scode"])
        rows["J2_pooled_firm"].append((p["coef"] / p["se"], p["se"], p["df"]))
    for name, r in rows.items():
        t, se, df = (np.array(v) for v in zip(*r))
        crit = stats.t.ppf(0.975, df)
        out[name] = {"df": int(df[0]), "rejection_rate_5pct": float(np.mean(np.abs(t) > crit)),
                     "se_quantiles_5_50_95": [float(x) for x in np.quantile(se, [0.05, 0.5, 0.95])],
                     "sd_of_t": float(t.std(ddof=1))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--null", type=int, default=0, help="permutation size check with this many replications")
    args = ap.parse_args()
    if args.null:
        res, name = null_check(args.null), "null_check.json"
    elif args.smoke:
        res, name = smoke(), "smoke.json"
    else:
        raise SystemExit("only --smoke and --null exist before contract 27 is approved")
    text = json.dumps(res, ensure_ascii=False, indent=2)
    (HERE / name).write_bytes((text + "\n").encode("utf-8"))
    print(text)


if __name__ == "__main__":
    main()
