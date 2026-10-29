"""Stage J estimators (contract 27): returns after Huazheng rating changes (J1), year-end fund ownership (J2) and
the implied cost of equity (J4). J3 is not done: the fund holdings were not obtained (contract 27.17).

    python -I -X utf8 estimate.py --smoke      # rating changes permuted within date or year: sizes and SEs only
    python -I -X utf8 estimate.py --smoke-j4   # J4 with the score permuted within year: sizes, SEs, size check
    python -I -X utf8 estimate.py --dry        # the --final path on permuted treatments: which estimates exist
    python -I -X utf8 estimate.py --final      # once (contract 27.11): estimates.json

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
J4: PEG implied cost of equity at the end of June t+1 (basis points) on the 31 October t rank-normal score, with
size, leverage, ROA, state ownership and log book-to-market as controls and industry x year effects; firm-clustered
(CR1) errors, t with G - 1 degrees of freedom; pi_E = -beta (contract 27.6, 27.17).
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
J4_CONTROLS = ["size", "lev", "roa", "soe", "log_bm"]
J4_VARIANTS = ("brokers3", "days90", "unadjusted")       # contract 27.8 and 27.17
FAMILY = ("J1", "J2", "J4")                              # Holm family (contract 27.17); J3 not done
J4_PANEL_SHA = "79d039316ccea3188276c2f3f7ad81384e9c9bc20b07b3a0bd7206059abdccb5"


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


def fama_macbeth(df: pd.DataFrame, y: str, x: str | list[str], controls: list[str], date: str, fe: str,
                 weight: str | None = None) -> dict:
    """Per-date OLS (WLS with weight) with fe effects absorbed; the mean of the per-date slopes on x, its standard
    error SD / sqrt(D) and D - 1 degrees of freedom. With a list x, one entry per regressor under "terms"."""
    xs = [x] if isinstance(x, str) else list(x)
    slopes = []
    for d, g in df.groupby(date):
        g = drop_singletons(g, g[fe])
        if weight is None:
            Z = demean(g, [y, *xs, *controls], g[fe])
            b = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
        else:
            w = g[weight].to_numpy(float)
            cols = [y, *xs, *controls]
            codes = pd.factorize(g[fe])[0]
            v = g[cols].to_numpy(float)
            sw = np.bincount(codes, weights=w)
            means = np.column_stack([np.bincount(codes, weights=w * v[:, i]) for i in range(v.shape[1])]) / sw[:, None]
            Z = (v - means[codes]) * np.sqrt(w)[:, None]
            b = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
        slopes.append((d, [float(v) for v in b[:len(xs)]], int(len(g))))
    B = np.array([s[1] for s in slopes])
    D = len(B)
    terms = {}
    for i, name in enumerate(xs):
        terms[name] = {"mean": float(B[:, i].mean()), "se": float(B[:, i].std(ddof=1) / np.sqrt(D))}
    out = {"dates": D, "obs": int(sum(s[2] for s in slopes)), "df": D - 1,
           "per_date": {s[0]: {"slope": s[1][0], "n": s[2]} for s in slopes}}
    if isinstance(x, str):
        out.update(terms[x])
    else:
        out["terms"] = terms
    return out


def cross_section(df: pd.DataFrame, y: str, x: str, controls: list[str], fe: str) -> dict:
    """One date: OLS with fe effects absorbed and HC1 errors; t with n - k - (fe cells) degrees of freedom."""
    df = drop_singletons(df, df[fe])
    Z = demean(df, [y, x, *controls], df[fe])
    n_fe = int(df[fe].nunique())
    beta, v = ols_cluster(Z[:, 0], Z[:, 1:], [np.arange(len(df))], n_fe)
    k = Z.shape[1] - 1
    return {"obs": int(len(df)), "coef": float(beta[0]), "se": float(np.sqrt(v[0, 0])), "df": int(len(df) - k - n_fe)}


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


def load_j4(period: str, variant: str = "main") -> pd.DataFrame:
    """Firm-years of the period with the variant's cost of equity, in basis points; winsorized within year."""
    a = pd.read_csv(PANELS / "j4_icc.csv.gz", dtype={"Scode": str})
    a = a[a.in_base & a[f"r_{variant}"].notna() & (a.period == period)].copy()
    a["icc"] = 1e4 * a[f"r_{variant}"]
    a = winsorize_within(a, ["icc", "size", "lev", "roa", "log_bm"], "year")
    a["cell"] = a.year.astype(str) + "|" + a.industry
    return a


def inference(est: float, se: float, df: int) -> dict:
    t = est / se
    h = stats.t.ppf(0.975, df) * se
    return {"estimate": est, "se": se, "df": int(df), "ci95": [est - h, est + h], "p": float(2 * stats.t.sf(abs(t), df))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[name]))
        out[name] = running
    return out


def fm_result(fm: dict) -> dict:
    return {**inference(fm["mean"], fm["se"], fm["df"]), "dates": fm["dates"], "obs": fm["obs"],
            "per_date": fm["per_date"]}


def pooled_result(p: dict) -> dict:
    return {**inference(p["coef"], p["se"], p["df"]), **{k: p[k] for k in ("obs", "fe_cells", "clusters")}}


def j1_block(e: pd.DataFrame, robustness: bool) -> dict:
    """J1 main specification (Fama-MacBeth, contract 27.3) and, for the point-in-time sample, the 27.8 variants."""
    out = {"main": fm_result(fama_macbeth(e, "car", "dnotch", J1_CONTROLS, "date", "industry")),
           "events": int(len(e)), "firms": int(e.Scode.nunique())}
    if not robustness:
        return out
    r = {}
    r["pooled_two_way"] = pooled_result(pooled(e, "car", "dnotch", J1_CONTROLS, e.cell, ["Scode", "date"]))
    r["treatment_dns_per_sd"] = fm_result(fama_macbeth(e, "car", "dns", J1_CONTROLS, "date", "industry"))
    ud = e.assign(up=e.dnotch.clip(lower=0), down=e.dnotch.clip(upper=0))
    split = fama_macbeth(ud, "car", ["up", "down"], J1_CONTROLS, "date", "industry")
    r["upgrades_and_downgrades"] = {k: {**inference(v["mean"], v["se"], split["df"])} for k, v in split["terms"].items()}
    r["without_2024_04"] = fm_result(fama_macbeth(e[e.date != "2024-04-30"], "car", "dnotch", J1_CONTROLS, "date", "industry"))
    one = e[e.date == "2024-04-30"]
    cs = cross_section(one, "car", "dnotch", J1_CONTROLS, "industry")
    r["only_2024_04_cross_section_hc1"] = {**inference(cs["coef"], cs["se"], cs["df"]), "obs": cs["obs"]}
    for col, name in (("r+1", "month_plus_1"), ("r+2", "month_plus_2"), ("car_3_6", "months_plus_3_to_6")):
        m = e[e[col].notna()].assign(y=lambda x: 100 * x[col])
        m = winsorize_within(m, ["y"], "date")
        r[name] = fm_result(fama_macbeth(m, "y", "dnotch", J1_CONTROLS, "date", "industry"))
    r["value_weighted"] = fm_result(fama_macbeth(e, "car", "dnotch", J1_CONTROLS, "date", "industry", weight="mcap"))
    out["robustness"] = r
    return out


def j2_block(a: pd.DataFrame, robustness: bool) -> dict:
    out = {"main": pooled_result(pooled(a, "dfund_own", "dnotch", J2_CONTROLS, a.cell, ["Scode"])),
           "firm_years": int(len(a)), "firms": int(a.Scode.nunique()), "years": sorted(int(y) for y in a.year.unique())}
    if robustness:
        out["robustness"] = {
            "treatment_dns_per_sd": pooled_result(pooled(a, "dfund_own", "dns", J2_CONTROLS, a.cell, ["Scode"])),
            "without_2024": pooled_result(pooled(a[a.year != 2024], "dfund_own", "dnotch", J2_CONTROLS,
                                                 a.cell, ["Scode"])),
        }
    return out


def j4_result(a: pd.DataFrame) -> dict:
    """pi_E = -beta: basis points lower cost of equity per SD of the score."""
    r = pooled_result(pooled(a, "icc", "ns", J4_CONTROLS, a.cell, ["Scode"]))
    pi = inference(-r["estimate"], r["se"], r["df"])
    return {**r, **pi, "beta_bp_per_sd": r["estimate"], "firm_years": int(len(a)), "firms": int(a.Scode.nunique()),
            "years": sorted(int(y) for y in a.year.unique()),
            "ci95_contains_28bp": bool(pi["ci95"][0] <= 28 <= pi["ci95"][1])}


def difference(a: dict, b: dict) -> dict:
    """Point-in-time minus placebo, independent samples; Welch-Satterthwaite degrees of freedom."""
    d, se = a["estimate"] - b["estimate"], float(np.hypot(a["se"], b["se"]))
    nu = se ** 4 / (a["se"] ** 4 / a["df"] + b["se"] ** 4 / b["df"])
    return inference(d, se, int(np.floor(nu)))


def verdict(estimate: float, p_holm: float) -> str:
    """Contract 27.7: every prediction is positive."""
    if p_holm >= 0.05:
        return "unresolved"
    return "supported" if estimate > 0 else "unsupported"


def final(rng: np.random.Generator | None = None) -> dict:
    """Contract 27.7-27.8 for J1, J2 and J4. With rng the treatments are permuted within date or year first (the
    --dry check of the code path); without, the real estimates."""

    def perm(df: pd.DataFrame, by: str, cols: tuple[str, ...]) -> pd.DataFrame:
        if rng is not None:
            for c in cols:
                df[c] = df.groupby(by)[c].transform(lambda s: rng.permutation(s.to_numpy()))
        return df

    out: dict = {"family": list(FAMILY),
                 "not_done": {"J3": "fund holdings and fund information not obtained (contract 27.17)",
                              "J2_half_yearly": "needs the fund holdings (contract 27.4, 27.17)"}}
    j1, j2, j4 = {}, {}, {}
    for period in ("pit", "prelaunch", "restated"):
        j1[period] = j1_block(perm(load_j1(period), "date", ("dnotch", "dns")), robustness=period == "pit")
        j2[period] = j2_block(perm(load_j2(period), "year", ("dnotch", "dns")), robustness=period == "pit")
        j4[period] = {"main": j4_result(perm(load_j4(period), "year", ("ns",)))}
    j4["pit"]["robustness"] = {v: j4_result(perm(load_j4("pit", v), "year", ("ns",))) for v in J4_VARIANTS}
    for j in (j1, j2, j4):
        j["difference_pit_minus_prelaunch"] = difference(j["pit"]["main"], j["prelaunch"]["main"])
    primary = {"J1": j1["pit"]["main"], "J2": j2["pit"]["main"], "J4": j4["pit"]["main"]}
    adj = holm({k: v["p"] for k, v in primary.items()})
    out["primary"] = {k: {"estimate": v["estimate"], "ci95": v["ci95"], "p": v["p"], "p_holm": adj[k],
                          "verdict": verdict(v["estimate"], adj[k])} for k, v in primary.items()}
    out["J1"], out["J2"], out["J4"] = j1, j2, j4
    return out


def smoke_j4(reps: int = 200) -> dict:
    """J4 with the score permuted within year (stream [27901, 2]): sizes, fixed-effect cells, clusters and standard
    errors; then a size check with reps permutations (stream [27903, 4]) that keeps only the standard errors and
    whether each rejects at 5%."""
    rng = np.random.default_rng([SMOKE_SEED, 2])
    out: dict = {"seed_stream": [SMOKE_SEED, 2], "panel_sha256": J4_PANEL_SHA}
    for period in ("pit", "restated", "prelaunch"):
        for variant in ("main", *J4_VARIANTS) if period == "pit" else ("main",):
            a = load_j4(period, variant)
            a["ns"] = a.groupby("year").ns.transform(lambda s: rng.permutation(s.to_numpy()))
            p = pooled(a, "icc", "ns", J4_CONTROLS, a.cell, ["Scode"])
            out[f"J4_{period}_{variant}"] = {
                "firm_years": int(len(a)), "firms": int(a.Scode.nunique()),
                "years": sorted(int(y) for y in a.year.unique()),
                **{k: p[k] for k in ("obs", "fe_cells", "clusters", "se", "df")},
                "mde_bp_alpha_0.05": mde(p["se"], p["df"], 0.05),
                "mde_bp_alpha_0.0167": mde(p["se"], p["df"], 0.05 / 3)}
    rng = np.random.default_rng([NULL_SEED, 4])
    a = load_j4("pit")
    t, se = [], []
    for _ in range(reps):
        a["x"] = a.groupby("year").ns.transform(lambda s: rng.permutation(s.to_numpy()))
        p = pooled(a, "icc", "x", J4_CONTROLS, a.cell, ["Scode"])
        t.append(p["coef"] / p["se"])
        se.append(p["se"])
    t, se = np.array(t), np.array(se)
    out["size_check_pit"] = {"seed_stream": [NULL_SEED, 4], "reps": reps, "df": int(p["df"]),
                             "rejection_rate_5pct": float(np.mean(np.abs(t) > stats.t.ppf(0.975, p["df"]))),
                             "se_quantiles_5_50_95": [float(x) for x in np.quantile(se, [0.05, 0.5, 0.95])],
                             "sd_of_t": float(t.std(ddof=1))}
    return out


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


def dry() -> dict:
    """The --final code path on permuted treatments (stream [27901, 3]): which estimates exist and their sample
    sizes; no coefficient, interval, p-value or verdict is kept."""
    drop = {"estimate", "se", "ci95", "p", "p_holm", "per_date", "mean", "beta_bp_per_sd", "ci95_contains_28bp",
            "verdict"}

    def shape(node):
        if isinstance(node, dict):
            return {k: shape(v) for k, v in node.items() if k not in drop}
        return node

    return shape(final(np.random.default_rng([SMOKE_SEED, 3])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-j4", action="store_true")
    ap.add_argument("--dry", action="store_true", help="the --final path on permuted treatments, sizes only")
    ap.add_argument("--final", action="store_true", help="run once (contract 27.11)")
    ap.add_argument("--null", type=int, default=0, help="permutation size check with this many replications")
    args = ap.parse_args()
    if args.final:
        import hashlib
        for f, want in (("events.csv.gz", "83d8a81cd89629f74a3f95dedcce15081b28abda8d25b87ced7b798e152ffe74"),
                        ("annual_fund_ownership.csv.gz",
                         "f62d517aba35f288fba355c870481dea2119d05518b291f1fe0079df050817b2"),
                        ("j4_icc.csv.gz", J4_PANEL_SHA)):                     # contract 27.2 and 27.17
            got = hashlib.sha256((PANELS / f).read_bytes()).hexdigest()
            assert got == want, (f, got)
        if (HERE / "estimates.json").exists():
            raise SystemExit("estimates.json exists: --final runs once (contract 27.11)")
        res, name = final(), "estimates.json"
    elif args.dry:
        res, name = dry(), "dry_final.json"
    elif args.smoke_j4:
        res, name = smoke_j4(), "smoke_j4.json"
    elif args.null:
        res, name = null_check(args.null), "null_check.json"
    elif args.smoke:
        res, name = smoke(), "smoke.json"
    else:
        raise SystemExit("choose --smoke, --smoke-j4, --null N, --dry or --final")
    text = json.dumps(res, ensure_ascii=False, indent=2)
    (HERE / name).write_bytes((text + "\n").encode("utf-8"))
    print(text)


if __name__ == "__main__":
    main()
