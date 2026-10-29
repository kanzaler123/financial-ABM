"""Stage J, J4 panel: implied cost of equity (PEG, Easton 2004) from analyst forecasts (contract 27.6, 27.17).

    python -I -X utf8 build_j4.py

Runs on the system Python (pandas, scipy). Inputs are local and not committed: the CNRDS analyst forecasts exported
2026-10-29 (data/raw/cnrds_20261029/分析师盈利预测/, docs/data_inventory.md), the CNRDS monthly returns and balance
sheet of 2026-10-26, and the stage I firm-year panel (data/processed/cnrds_stage_i_20261027/firm_year.csv.gz,
contract 26.2 and 26.17), which supplies the 31 October score, the controls and the industry. Writes
data/processed/stage_j_20261028/j4_icc.csv.gz (not committed) and j4_panel_summary.json here: input and panel hashes,
the sample flow, coverage and the distribution of every variable, each on its own. Nothing here relates a score to
the cost of equity.

Firm-year t. At P0 = the closing price at the end of June t+1, EPS1 and EPS2 are the consensus forecasts for fiscal
years t+1 and t+2: each broker's latest forecast for that year in the 180 days up to 30 June t+1, then the median
over brokers, with at least two brokers. r = sqrt((EPS2 - EPS1) / P0), defined when EPS2 > EPS1 > 0.

Share basis (contract 27.17). A forecast's EPS is on the shares outstanding when it was issued; bonus and
capitalisation issues, mostly ex-dated in May to July, change the share count before 30 June. Each forecast is put on
the June share basis: EPS x (total shares at the end of the forecast month) / (total shares at the end of June t+1).
The unadjusted EPS (the contract 27.6 text) is kept as a robustness variant.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORECASTS = ROOT / "data/raw/cnrds_20261029/分析师盈利预测"
CNRDS = ROOT / "data/processed/cnrds_20261026"
FIRM_YEAR = ROOT / "data/processed/cnrds_stage_i_20261027/firm_year.csv.gz"
OUT = ROOT / "data/processed/stage_j_20261028"
ARCHIVE_SHA = "58870f6c2bab28bd23fd0c08cff8f43b02d1ef48666eab924464ee888cc36b72"   # 分析师盈利预测.7z

SHSZ = r"^(000|001|002|003|300|301|302|600|601|603|605|688|689)\d{3}$"
YEARS = range(2009, 2025)
# Variants: (window in days, minimum brokers, share-basis adjustment). "main" is contract 27.6 with 27.17.
VARIANTS = {"main": (180, 2, True), "brokers3": (180, 3, True), "days90": (90, 2, True), "unadjusted": (180, 2, False)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def describe(s: pd.Series) -> dict:
    s = pd.Series(s).dropna()
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).tolist() if len(s) else []
    return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None, "sd": float(s.std()) if len(s) > 1 else None,
            "quantiles_0_1_5_25_50_75_95_99_100": [float(x) for x in q]}


def period(t: int) -> str:
    """Vintage of the 31 October t score (contract 27.1): as released from October 2022."""
    return "prelaunch" if t <= 2017 else "restated" if t <= 2021 else "pit"


def forecast_files() -> list[Path]:
    files = [p for p in FORECASTS.glob("分析师盈利预测_*.csv") if p.stem.rsplit("_", 1)[1].isdigit()]
    files.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    assert len(files) == 30, len(files)
    return files


def forecasts(summary: dict) -> pd.DataFrame:
    parts = []
    for p in forecast_files():
        x = pd.read_csv(p, encoding="utf-8-sig", dtype=str, usecols=["Scode", "Anndats", "Fpenddt", "B_code", "Feps"])
        assert x.iloc[0].Scode == "股票代码", p                 # the export's second header row holds the Chinese labels
        parts.append(x.iloc[1:])
    f = pd.concat(parts, ignore_index=True)
    rows = len(f)
    f["date"] = pd.to_datetime(f.Anndats, errors="raise")
    f["fy"] = f.Fpenddt.str[:4].astype(int)
    assert f.Fpenddt.str[5:].eq("12-31").all()
    f["eps"] = num(f.Feps)
    summary["forecasts"] = {"rows": rows, "firms": int(f.Scode.nunique()), "brokers": int(f.B_code.nunique()),
                            "report_dates": [str(f.date.min().date()), str(f.date.max().date())],
                            "eps_missing": int(f.eps.isna().sum()), "exact_duplicates": int(f.duplicated().sum())}
    f = f[f.eps.notna()]
    # Same broker, firm, fiscal year and date with several values: their mean is that day's forecast.
    f = f.groupby(["Scode", "fy", "B_code", "date"], as_index=False).eps.mean()
    summary["forecasts"]["broker_firm_year_dates"] = int(len(f))
    return f[["Scode", "fy", "B_code", "date", "eps"]]


def month_end(mr: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mr = mr[mr.Trdmt != "2017-03-10"]                         # mid-month snapshot (data_inventory.md)
    key = [mr.Scode, mr.Trdmt.str[:7]]
    shares = pd.Series(num(mr.Mts).to_numpy(), index=pd.MultiIndex.from_arrays(key))
    price = pd.Series(num(mr.Mclsprc).to_numpy(), index=pd.MultiIndex.from_arrays(key))
    assert not shares.index.duplicated().any()
    return shares, price


def consensus(f: pd.DataFrame, shares: pd.Series, t: int, days: int, min_brokers: int, adjust: bool) -> pd.DataFrame:
    end = pd.Timestamp(f"{t + 1}-06-30")
    w = f[(f.date > end - pd.Timedelta(days=days)) & (f.date <= end) & f.fy.isin([t + 1, t + 2])]
    w = w.sort_values("date").groupby(["Scode", "fy", "B_code"]).tail(1).copy()
    if adjust:
        s_f = shares.reindex(pd.MultiIndex.from_arrays([w.Scode, w.date.dt.strftime("%Y-%m")])).to_numpy()
        s_j = shares.reindex(pd.MultiIndex.from_arrays([w.Scode, np.full(len(w), f"{t + 1}-06")])).to_numpy()
        w["eps"] = w.eps * s_f / s_j                           # missing when either share count is missing
        w = w[w.eps.notna()]
    c = w.groupby(["Scode", "fy"]).agg(eps=("eps", "median"), brokers=("B_code", "nunique")).reset_index()
    c = c[c.brokers >= min_brokers]
    one = c[c.fy == t + 1].set_index("Scode")
    two = c[c.fy == t + 2].set_index("Scode")
    out = pd.DataFrame({"eps1": one.eps, "brokers1": one.brokers}).join(
        pd.DataFrame({"eps2": two.eps, "brokers2": two.brokers}), how="outer")
    return out.reset_index().assign(year=t)


def book_equity() -> pd.DataFrame:
    bs = pd.read_csv(CNRDS / "balance_sheet.csv.gz", dtype=str, usecols=["Scode", "Date", "EAPC", "EQU"])
    bs = bs[bs.Date.str.endswith("12-31")].copy()
    bs["year"] = bs.Date.str[:4].astype(int)
    bs["book"] = num(bs.EAPC).fillna(num(bs.EQU))           # as build_panel.book_equity (J1, J2)
    assert not bs.duplicated(["Scode", "year"]).any()
    return bs[["Scode", "year", "book"]]


def share_basis_check(f: pd.DataFrame, shares: pd.Series) -> dict:
    """How many forecasts in the main window precede a change in total shares before 30 June (contract 27.17)."""
    out = {}
    for t in YEARS:
        end = pd.Timestamp(f"{t + 1}-06-30")
        w = f[(f.date > end - pd.Timedelta(days=180)) & (f.date <= end) & f.fy.isin([t + 1, t + 2])]
        w = w.sort_values("date").groupby(["Scode", "fy", "B_code"]).tail(1)
        s_f = shares.reindex(pd.MultiIndex.from_arrays([w.Scode, w.date.dt.strftime("%Y-%m")])).to_numpy()
        s_j = shares.reindex(pd.MultiIndex.from_arrays([w.Scode, np.full(len(w), f"{t + 1}-06")])).to_numpy()
        r = s_j / s_f
        ok = np.isfinite(r)
        out[str(t)] = {"forecasts": int(ok.sum()), "share_count_missing": int((~ok).sum()),
                       "share_growth_gt_5pct": float((r[ok] > 1.05).mean()),
                       "share_growth_gt_20pct": float((r[ok] > 1.2).mean()),
                       "share_fall_gt_5pct": float((r[ok] < 0.95).mean()),
                       "firms_with_growth_gt_5pct": int(w.Scode[ok & (r > 1.05)].nunique()),
                       "median_growth_when_gt_5pct": float(np.median(r[ok & (r > 1.05)])) if (ok & (r > 1.05)).any() else None}
    return out


def main() -> None:
    files = forecast_files()
    summary: dict = {"inputs": {"archive_7z_sha256": ARCHIVE_SHA,
                                **{p.relative_to(ROOT).as_posix(): sha(p) for p in
                                   [*files, CNRDS / "monthly_returns.csv.gz", CNRDS / "balance_sheet.csv.gz", FIRM_YEAR]}}}
    f = forecasts(summary)
    mr = pd.read_csv(CNRDS / "monthly_returns.csv.gz", dtype=str, usecols=["Scode", "Trdmt", "Mclsprc", "Mts"])
    shares, price = month_end(mr)
    summary["share_basis_check"] = share_basis_check(f, shares)

    fy = pd.read_csv(FIRM_YEAR, dtype={"Scode": str})
    fy = fy[fy.year.isin(YEARS)]
    p = fy[["Scode", "year", "industry", "IndcodeB", "ns", "score", "size", "lev", "roa", "soe", "mcap", "AT", "LB"]].copy()
    p = p.merge(book_equity(), on=["Scode", "year"], how="left")
    p["log_bm"] = np.log((p.book / p.mcap).where((p.book > 0) & (p.mcap > 0)))
    p["p0"] = price.reindex(pd.MultiIndex.from_arrays([p.Scode, (p.year + 1).astype(str) + "-06"])).to_numpy()
    p["period"] = p.year.map(period)

    for name, (days, k, adjust) in VARIANTS.items():
        c = pd.concat([consensus(f, shares, t, days, k, adjust) for t in YEARS], ignore_index=True)
        c = c.rename(columns={col: f"{col}_{name}" for col in ("eps1", "eps2", "brokers1", "brokers2")})
        p = p.merge(c, on=["Scode", "year"], how="left")
        e1, e2 = p[f"eps1_{name}"], p[f"eps2_{name}"]
        ok = (e2 > e1) & (e1 > 0) & (p.p0 > 0)
        p[f"r_{name}"] = np.sqrt(((e2 - e1) / p.p0).where(ok))

    steps = [
        ("Shanghai or Shenzhen A share", p.Scode.str.match(SHSZ)),
        ("industry code available, not financial", p.industry.notna() & (p.industry != "J")),
        ("total assets > 0 and liabilities <= assets", (p.AT > 0) & (p.LB <= p.AT)),
        ("31 October score available", p.ns.notna()),
        ("controls available (size, leverage, ROA, state ownership)", p[["size", "lev", "roa", "soe"]].notna().all(axis=1)),
        ("book equity > 0 and December market value", p.log_bm.notna()),
        ("closing price at the end of June t+1", p.p0 > 0),
        (">= 2 brokers for both fiscal years t+1 and t+2", p.eps1_main.notna() & p.eps2_main.notna()),
        ("EPS2 > EPS1 > 0", p.r_main.notna()),
    ]
    keep, flows = pd.Series(True, index=p.index), {}
    for per in ("pit", "restated", "prelaunch"):
        flows[per] = []
    for label, cond in steps:
        before = {per: int((keep & (p.period == per)).sum()) for per in flows}
        keep &= cond.fillna(False)
        for per in flows:
            now = int((keep & (p.period == per)).sum())
            flows[per].append({"step": label, "kept": now, "removed": before[per] - now})
    base = pd.Series(True, index=p.index)
    for _, cond in steps[:-2]:
        base &= cond.fillna(False)
    p["in_base"] = base                                       # every condition except the forecast ones
    p["in_sample"] = keep                                     # main variant
    summary["sample_flow"] = flows

    s = p[p.in_sample]
    summary["sample"] = {
        per: {"firm_years": int(len(g)), "firms": int(g.Scode.nunique()), "years": sorted(int(y) for y in g.year.unique()),
              "by_year": g.groupby("year").size().to_dict(),
              "industry_year_cells": int(g.groupby(["year", "industry"]).ngroups),
              "soe_share": float(g.soe.mean()),
              "variables": {c: describe(g[c]) for c in ["r_main", "eps1_main", "eps2_main", "brokers1_main",
                                                        "brokers2_main", "p0", "ns", "size", "lev", "roa", "log_bm"]}}
        for per, g in s.groupby("period")}
    summary["variants"] = {
        name: {per: {"firm_years": int((p.in_base & p[f"r_{name}"].notna() & (p.period == per)).sum()),
                     "r": describe(p.loc[p.in_base & (p.period == per), f"r_{name}"])}
               for per in ("pit", "restated", "prelaunch")}
        for name in VARIANTS}
    b = p[p.in_base & (p.period == "pit")]
    has_two = b.eps1_main.notna() & b.eps2_main.notna()
    summary["pit_coverage"] = {
        "base_firm_years": int(len(b)), "with_two_broker_consensus": int(has_two.sum()),
        "eps1_le_0": int((has_two & (b.eps1_main <= 0)).sum()),
        "eps2_le_eps1_with_eps1_gt_0": int((has_two & (b.eps1_main > 0) & (b.eps2_main <= b.eps1_main)).sum()),
        "adjusted_vs_unadjusted_r_main_sample": describe((p.r_main / p.r_unadjusted)[p.in_sample & (p.period == "pit")])}

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["Scode", "year", "period", "industry", "IndcodeB", "ns", "score", "size", "lev", "roa", "soe", "log_bm",
            "mcap", "book", "p0", *[f"{c}_{n}" for n in VARIANTS for c in ("eps1", "eps2", "brokers1", "brokers2", "r")],
            "in_base", "in_sample"]
    buf = io.StringIO()
    p[cols].to_csv(buf, index=False, lineterminator="\n")
    raw = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
    (OUT / "j4_icc.csv.gz").write_bytes(raw)
    summary["outputs"] = {"directory": OUT.relative_to(ROOT).as_posix(),
                          "j4_icc.csv.gz": {"sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(p))}}
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    (HERE / "j4_panel_summary.json").write_bytes((text + "\n").encode("utf-8"))
    print(json.dumps({"flow": flows, "sample": {k: v["firm_years"] for k, v in summary["sample"].items()},
                      "pit_coverage": summary["pit_coverage"], "outputs": summary["outputs"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
