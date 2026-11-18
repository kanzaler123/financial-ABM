"""Stage I estimates: the Huazheng score and the cost of debt (contract section 26).

    python -I -X utf8 estimate.py --selftest    the estimator on synthetic panels with a known coefficient
    python -I -X utf8 estimate.py --smoke       every specification on the real panel, the score permuted within year
    python -I -X utf8 estimate.py --final       the pre-registered estimates (only after section 26 is committed)

Reads the panels written by build_panel.py (data/processed/cnrds_stage_i_20261027/, not committed). OLS with
absorbed fixed effects (alternating projections) and standard errors clustered by firm: CR1 with the factor
G/(G - 1) (N - 1)/(N - K), where K counts the regressors and the levels of absorbed effects not nested in the
clusters; 95% intervals use t with G - 1 degrees of freedom. Singletons are dropped first, repeatedly. Outcomes
are in basis points, so the score coefficient is basis points per SD of the score and the premium is minus it.

Bond spreads (contract 26.6) need the MOF treasury curve downloaded by download_curve.py: each file is checked
against curve_manifest.json, the curve of the last trading day on or before the value date is interpolated
linearly between standard tenors at the bond's maturity (3 months below 3 months, 30 years above 30 years), and
the spread, coupon minus that yield in basis points, is clipped at the 1st and 99th percentiles of its issue year.

The smoke run replaces the score by a seeded permutation within each year (issue year for bonds, from a separate
random stream), so it exercises the samples, fixed effects and clustering of every specification without
estimating any relation between the score and an outcome.
"""

from __future__ import annotations

import argparse
import hashlib
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
CURVE_MANIFEST = HERE / "curve_manifest.json"
CONTROLS = ["size_w", "lev_w", "roa_w"]
EXTENDED = ["tang_w", "cash_w", "growth_w", "age_w"]
BOND_CONTROLS = ["size", "lev", "roa", "soe_issuer", "ln_maturity", "ln_amount"]
BOND_FES = ["Bdtp", "Listplc", "ind_year"]          # bond type, trading market, industry x issue year
TENORS = {"3月": 0.25, "6月": 0.5, "1年": 1.0, "2年": 2.0, "3年": 3.0, "5年": 5.0, "7年": 7.0, "10年": 10.0,
          "30年": 30.0}
WINSOR = (0.01, 0.99)
# Contract 26.1 and 26.17: the final estimates read only the panels rebuilt once with the ownership column, whose
# SHA-256 values were recorded in the contract before any estimate.
FINAL_PANELS: dict[str, str] | None = {
    "firm_year.csv.gz": "a12c769709db3a88c008160c58181ec3eea3ee670b70cbe13059e082fa52fa57",
    "bond_issues.csv.gz": "7301eedf5aae83fb06147e78c3c006009fa8ad8bdc5494d4bdaefed75fc7b759",
}
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
    ] + ([Spec("reported_controller", "robustness", "drops firm-years whose actual-controller category is empty, "
               "which the main specification codes as not state-owned (contract 26.17)", controls=base,
               sample=lambda d: ~d.controller_missing)] if ownership else [])


def bond_specs() -> list[Spec]:
    """Contract 26.6: the secondary result and its three robustness checks."""
    common = dict(y="spread_w_bp", data="bonds")
    within_issuer = [c for c in BOND_CONTROLS if c != "soe_issuer"]   # constant within issuer, absorbed
    return [
        Spec("bond_main", "secondary", "issue spread over the treasury curve on the last score before the value "
             "date; issuer size, leverage, ROA of the previous year, state-owned issuer, log maturity, log amount; "
             "bond type, market and industry x issue year effects", controls=list(BOND_CONTROLS), fes=list(BOND_FES),
             **common),
        Spec("bond_issuer_fe", "secondary_robustness", "adds issuer fixed effects (the state-owned indicator is "
             "constant within issuer and drops out)", controls=within_issuer, fes=["Scode", *BOND_FES], **common),
        Spec("bond_rating_fe", "secondary_robustness", "adds fixed effects of the issuer rating at issue",
             controls=list(BOND_CONTROLS), fes=[*BOND_FES, "rating"], **common),
        Spec("bond_maturity_1y", "secondary_robustness", "bonds with a maturity of at least one year",
             controls=list(BOND_CONTROLS), fes=list(BOND_FES), sample=lambda d: d.maturity >= 1 - 1e-9, **common),
    ]


def describe(s: pd.Series) -> dict:
    s = s.dropna()
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).tolist() if len(s) else []
    return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None, "sd": float(s.std()) if len(s) > 1 else None,
            "quantiles_0_1_5_25_50_75_95_99_100": [float(x) for x in q]}


def read_curve() -> tuple[pd.DataFrame, dict]:
    """The MOF treasury curve in percent: one row per trading day (ascending), one column per tenor in years.
    Every file must match the size and SHA-256 recorded by download_curve.py."""
    import warnings

    import openpyxl

    manifest = json.loads(CURVE_MANIFEST.read_text(encoding="utf-8"))
    directory = ROOT / manifest["directory"]
    frames = []
    for entry in manifest["files"]:
        path = directory / entry["file"]
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path.name} differs from curve_manifest.json")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)              # "no default style"
            book = openpyxl.load_workbook(path, data_only=True)       # read-only mode sees one row (bad dimension tag)
        rows = list(book.worksheets[0].iter_rows(values_only=True))
        if rows[0] != ("日期", *TENORS):
            raise ValueError(f"{path.name}: unexpected header {rows[0]}")
        frames.append(pd.DataFrame(rows[1:], columns=["date", *TENORS.values()]))
    curve = pd.concat(frames).set_index("date").astype(float).sort_index()
    if not curve.index.is_unique or curve.isna().any().any():
        raise ValueError("curve dates repeat or yields are missing")
    info = {"manifest_sha256": hashlib.sha256(CURVE_MANIFEST.read_bytes()).hexdigest(), "files": len(manifest["files"]),
            "trading_days": int(len(curve)), "first_date": str(curve.index[0]), "last_date": str(curve.index[-1])}
    return curve, info


def treasury_yield(curve: pd.DataFrame, value_dates: pd.Series, maturity: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Yield in percent of the last curve on or before each value date at the bond's maturity in years: linear
    between standard tenors, the 3-month yield below 3 months and the 30-year yield above 30 years."""
    dates = curve.index.to_numpy(dtype=str)
    idx = np.searchsorted(dates, value_dates.to_numpy(dtype=str), side="right") - 1
    if (idx < 0).any():
        raise ValueError("a value date precedes the first curve date")
    tenors = np.array(list(TENORS.values()))
    rows = curve.to_numpy()[idx]
    m = np.clip(maturity.to_numpy(float), tenors[0], tenors[-1])
    return np.array([np.interp(mi, tenors, row) for mi, row in zip(m, rows)]), dates[idx]


def load_bonds(smoke: bool) -> tuple[pd.DataFrame, dict]:
    b = pd.read_csv(PANEL / "bond_issues.csv.gz", dtype={"Bdcd": str, "Scode": str, "IndcodeB": str, "industry": str})
    curve, info = read_curve()
    b["treasury_yield"], b["curve_date"] = treasury_yield(curve, b.Valdt, b.maturity)
    b["spread_bp"] = (b.coupon - b.treasury_yield) * 100
    bounds = b.groupby("issue_year").spread_bp.quantile(list(WINSOR)).unstack()
    b["spread_w_bp"] = b.spread_bp.clip(b.issue_year.map(bounds[WINSOR[0]]), b.issue_year.map(bounds[WINSOR[1]]))
    b["ln_maturity"], b["ln_amount"] = np.log(b.maturity), np.log(b.amount)
    b["soe_issuer"] = b.soe_issuer.astype(float)
    b["ind_year"] = b.industry + "_" + b.issue_year.astype(int).astype(str)
    b["rating"] = b.Issbjrt.str.replace(r"(pi|sti)$", "", regex=True)    # qualifiers dropped: AAApi -> AAA
    gap = (pd.to_datetime(b.Valdt) - pd.to_datetime(b.curve_date)).dt.days
    info.update({
        "bonds": int(len(b)), "curve_date_before_value_date": {"bonds": int((gap > 0).sum()), "max_days": int(gap.max())},
        "maturity_below_3_months": int((b.maturity < 0.25).sum()), "maturity_above_30_years": int((b.maturity > 30).sum()),
        "spread_bp": describe(b.spread_bp), "spread_w_bp": describe(b.spread_w_bp),
        "winsor_bounds_by_issue_year": {int(y): [float(r[WINSOR[0]]), float(r[WINSOR[1]])] for y, r in bounds.iterrows()},
        "market_missing": int(b.Listplc.isna().sum()), "rating_missing": int(b.rating.isna().sum()),
        "rating_levels": b.rating.value_counts().to_dict(),
    })
    if smoke:
        rng = np.random.default_rng([SMOKE_SEED, 6])   # its own stream: the firm-year permutation is unchanged
        v = b.ns.to_numpy().copy()
        for idx in b.groupby("issue_year").indices.values():
            v[idx] = v[rng.permutation(idx)]
        b["ns"] = v
    return b, info


def load_firm_year(smoke: bool) -> tuple[pd.DataFrame, bool]:
    d = pd.read_csv(PANEL / "firm_year.csv.gz", dtype={"Scode": str, "IndcodeB": str, "industry": str})
    d = d[d.main_sample].copy()
    d["ind_year"] = d.industry + "_" + d.year.astype(str)
    d["y_bp"] = d.debt_cost_w * 1e4
    d["y_nolease_bp"] = d.debt_cost_nolease_w * 1e4
    ownership = bool("soe" in d.columns and d.soe.notna().any())
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
        missing = int(d[[s.y, s.x, *s.controls, *s.fes, s.cluster]].isna().any(axis=1).sum())
        fit = fe_ols(d, s.y, [s.x, *s.controls], s.fes, s.cluster, nested=(s.cluster,))
        r = fit.row(0)
        out.append({"name": s.name, "role": s.role, "describe": s.describe, "x": s.x, "y": s.y, "controls": s.controls,
                    "fixed_effects": s.fes, "n": fit.n, "clusters": fit.clusters, "missing_dropped": missing,
                    "singletons_dropped": fit.singletons,
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
    # The bond designs absorb three or four sets of effects (bond type, market, industry x year, and rating or the
    # issuer): alternating projections against explicit dummies, on a stream of their own.
    rng = np.random.default_rng(26003)
    for fes in (["kind", "market", "ind_year"], ["kind", "market", "ind_year", "grade"],
                ["Scode", "kind", "market", "ind_year"]):
        d = synthetic(rng, 150, 6, 5, -8.0, confound=True)
        d["kind"], d["market"], d["grade"] = (rng.integers(0, k, len(d)) for k in (4, 3, 6))
        keep = drop_singletons([codes(d[f]) for f in fes])
        d = d[keep]
        fit = fe_ols(d, "y", ["x", "c1"], fes, "Scode", nested=("Scode",))
        b, se = dummy_ols(d, "y", ["x", "c1"], fes, nested=("Scode",))
        out["exact_vs_dummies"].append({"fixed_effects": fes, "max_abs_beta_diff": float(np.abs(fit.beta - b).max()),
                                        "max_rel_se_diff": float(np.abs(fit.se / se - 1).max())})
        assert np.allclose(fit.beta, b, atol=1e-8) and np.allclose(fit.se, se, rtol=1e-6), (fit.beta, b, fit.se, se)
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
        if args.final:
            if FINAL_PANELS is None:
                raise SystemExit("--final needs the rebuilt panels' SHA-256 values in FINAL_PANELS (contract 26.1)")
            for file, digest in FINAL_PANELS.items():
                if hashlib.sha256((PANEL / file).read_bytes()).hexdigest() != digest:
                    raise SystemExit(f"{file} is not the panel recorded in contract 26.1")
        firm_year, ownership = load_firm_year(smoke=args.smoke)
        if args.final and not ownership:
            raise SystemExit("the main specification needs the ownership column (contract 26.1)")
        bonds, bond_info = load_bonds(smoke=args.smoke)
        specs = debt_specs(ownership) + bond_specs()
        result = {"mode": "smoke: score permuted within year (seed %d); no score-outcome relation" % SMOKE_SEED
                  if args.smoke else "final", "ownership_control": ownership,
                  "estimates": run(specs, {"firm_year": firm_year, "bonds": bonds}),
                  "bond_panel": {**({"permutation": "score permuted within issue year, stream [%d, 6]" % SMOKE_SEED}
                                    if args.smoke else {}), **bond_info}}
        name = "smoke.json" if args.smoke else "estimates.json"
    (HERE / name).write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if args.selftest:
        print(json.dumps(result, indent=1))
    else:
        print(json.dumps([{k: r[k] for k in ("name", "n", "clusters", "singletons_dropped", "fe_levels")}
                          for r in result["estimates"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
