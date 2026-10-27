"""Stage I panels: firm-year debt cost and bond issues with Huazheng scores and controls (contract section 26).

    python -I -X utf8 build_panel.py

Runs on the system Python (pandas, openpyxl); the project venv has neither. Inputs are local and not committed:
the CNRDS tables converted to data/processed/cnrds_20261026/*.csv.gz (docs/data_inventory.md), the netdisk
Huazheng quarterly panel and the CSI 300 list of 2026-10-09. Writes the panels to data/processed/
cnrds_stage_i_20261027/ (not committed) and panel_summary.json here: input and panel hashes, the sample flow,
coverage, missing rates and the distribution of every variable. Nothing here relates a score to an outcome.

Firm-year t carries the Huazheng score of 31 October of year t, balance-sheet controls at the end of t and the
debt cost of year t + 1: interest expense of t + 1 over the mean of interest-bearing debt at the ends of t and
t + 1. Interest expense is the income-statement line INTEXP from 2019 (when it exists and is not negative) and
otherwise the finance-expense note, mapped by NOTE_RULE below; a missing current-period note is filled with the
prior-period column of the next year's note.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CNRDS = ROOT / "data/processed/cnrds_20261026"
HUAZHENG = ROOT / "data/raw/huazheng_panel_netdisk_20261026/huazheng_quarterly_2009_2025.xlsx"
CSI300 = ROOT / "data/processed/disclosure_public_20261011/csi300_constituents_20261009.csv"
OUT = ROOT / "data/processed/cnrds_stage_i_20261027"

YEARS = range(2009, 2025)            # score year t; the outcome year is t + 1 <= 2025
SHSZ = re.compile(r"^(000|001|002|003|300|301|302|600|601|603|605|688|689)\d{3}$")
REGIMES = (("R1", "2009-01-31", "2021-01-31"), ("R2", "2021-04-30", "2024-01-31"), ("R3", "2024-04-30", "2025-10-31"))
DEBT = ("STBL", "NCLDOY", "LTBL", "LTBP")          # short-term loans, current portion, long-term loans, bonds
LEASE = "LSOB"                                     # lease liabilities (new lease standard, 2019/2021 on)
MIN_DEBT_TO_ASSETS = 0.01                          # see contract 26: mean interest-bearing debt >= 1% of assets
WINSOR = (0.01, 0.99)

# Finance-expense note lines (names normalised by norm_item). A: interest expense; L: lease interest shown as its
# own line; C: interest capitalised and deducted. Sub-items ("其中:") and net lines (利息净支出) are not used.
NOTE_RULE = {
    "A": {"利息支出", "利息费用", "借款利息支出", "利息费用总额", "贷款及应付款项的利息支出", "贷款利息支出", "借款利息",
          "贷款的利息支出", "费用化利息支出", "借款利息费用", "加:利息支出", "加:利息费用", "利息支出总额", "利息开支",
          "其他利息支出", "债券利息", "债券利息支出", "银行借款利息支出", "银行借款利息"},
    "L": {"租赁负债利息支出", "租赁负债利息费用", "租赁负债的利息支出", "租赁负债的利息费用", "加:租赁负债利息支出",
          "加:租赁负债利息费用", "租赁负债利息", "租赁利息"},
    "C": {"减:利息资本化", "减:利息资本化金额", "减:资本化利息", "减:资本化的利息支出", "减:已资本化的利息费用",
          "减:已资本化的利息支出", "减:资本化利息支出", "减:利息资本化的金额", "减:资本化的利息", "利息资本化金额",
          "利息资本化"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(CNRDS / f"{name}.csv.gz", dtype=str, usecols=cols)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip(), errors="coerce")


def norm_item(s: str) -> str:
    s = (s or "").strip()
    for a, b in (("：", ":"), ("﹕", ":"), (" ", ""), ("　", ""), ("（", "("), ("）", ")")):
        s = s.replace(a, b)
    return s


def note_class(item: str) -> str:
    for k, names in NOTE_RULE.items():
        if item in names:
            return k
    return "C" if item.startswith("减:资本化利息") else ""


def note_interest(lines: pd.DataFrame, col: str) -> float:
    """Interest expense of one note column: one interest line less capitalisation plus lease lines; with several
    interest lines, the smallest when a capitalisation line shows that one of them is gross, otherwise the largest
    (component lines plus their total)."""
    a = lines.loc[lines.k == "A", col].dropna()
    if a.empty:
        return np.nan
    c = lines.loc[lines.k == "C", col].dropna().abs().sum()
    if len(a) == 1:
        return float(a.iloc[0] - c + lines.loc[lines.k == "L", col].dropna().sum())
    return float(a.min() if c > 0 else a.max())


def notes_interest() -> tuple[pd.DataFrame, dict]:
    d = read("finance_expense_notes")
    d = d[d.Date.str.endswith("12-31")].copy()
    d["k"] = d.Itemnm.fillna("").map(norm_item).map(note_class)
    d = d[d.k != ""].copy()
    d["cp"], d["pp"] = num(d.Fiexpcp), num(d.Fiexppp)
    d["year"] = d.Date.str[:4].astype(int)
    cur = d.groupby(["Scode", "year"]).apply(lambda x: note_interest(x, "cp"), include_groups=False).rename("note_cur")
    prior = d.groupby(["Scode", "year"]).apply(lambda x: note_interest(x, "pp"), include_groups=False).rename("note_prior")
    prior = prior.reset_index()
    prior["year"] -= 1                    # the prior-period column of year y's note describes year y - 1
    out = cur.reset_index().merge(prior, on=["Scode", "year"], how="outer")
    return out, {"note_lines_used": int(len(d)), "by_class": d.k.value_counts().to_dict()}


def normal_scores(values: pd.Series) -> pd.Series:
    ok = values.notna()
    out = pd.Series(np.nan, index=values.index)
    out[ok] = ndtri((rankdata(values[ok]) - 0.5) / ok.sum())
    return out


def huazheng() -> pd.DataFrame:
    import openpyxl

    rows = list(openpyxl.load_workbook(HUAZHENG, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    assert rows[0][:5] == ("证券代码", "季度日期", "证券简称", "综合评级", "综合得分"), rows[0]
    h = pd.DataFrame([(f"{r[0]:06d}", r[1].date().isoformat(), r[3], float(r[4])) for r in rows[1:]],
                     columns=["Scode", "date", "grade", "score"])
    g = h.groupby("date").score
    h["ns"] = h.groupby("date", group_keys=False).score.apply(normal_scores)   # within date = within method
    h["z"] = (h.score - g.transform("mean")) / g.transform("std")
    h["grade_num"] = h.grade.map({x: i + 1 for i, x in enumerate(["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"])})
    h["regime"] = ""
    for name, a, b in REGIMES:
        h.loc[(h.date >= a) & (h.date <= b), "regime"] = name
    assert (h.regime != "").all() and not h.duplicated(["Scode", "date"]).any()
    return h


def winsorize(df: pd.DataFrame, cols: list[str], sample: pd.Series) -> pd.DataFrame:
    """Clip at the 1st and 99th percentiles of the main sample in the same year (bounds applied to every row)."""
    for c in cols:
        bounds = df[sample].groupby("year")[c].quantile(list(WINSOR)).unstack()
        lo, hi = df.year.map(bounds[WINSOR[0]]), df.year.map(bounds[WINSOR[1]])
        df[c + "_w"] = df[c].clip(lo, hi)
    return df


def describe(s: pd.Series) -> dict:
    s = s.dropna()
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).tolist() if len(s) else []
    return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None, "sd": float(s.std()) if len(s) > 1 else None,
            "quantiles_0_1_5_25_50_75_95_99_100": [float(x) for x in q]}


def firm_year(h: pd.DataFrame, summary: dict) -> pd.DataFrame:
    info = read("company_info", ["Scode", "Coname_cn", "Listdt", "IndcodeB"])
    bs = read("balance_sheet", ["Scode", "Date", "AT", "LB", "EQU", "CH", "PPE", "STBP", *DEBT, LEASE])
    bs = bs[bs.Date.str.endswith("12-31")].copy()
    inc = read("income_statement", ["Scode", "Date", "NI", "OPREV", "INTEXP"])
    inc = inc[inc.Date.str.endswith("12-31")].copy()
    for df in (bs, inc):
        df["year"] = df.Date.str[:4].astype(int)
        for c in df.columns.difference(["Scode", "Date", "year"]):
            df[c] = num(df[c])
    assert not bs.duplicated(["Scode", "year"]).any() and not inc.duplicated(["Scode", "year"]).any()
    for c in (*DEBT, LEASE, "STBP"):
        bs[c + "_blank"] = bs[c].isna()
        bs[c] = bs[c].fillna(0.0)                      # a blank line is an item the report does not show
    bs["ibd"] = bs[list(DEBT)].sum(axis=1) + bs[LEASE]
    bs["ibd_nolease"] = bs[list(DEBT)].sum(axis=1)

    notes, note_info = notes_interest()
    inc = inc.merge(notes, on=["Scode", "year"], how="outer")
    cur_ok, prior_ok = inc.note_cur >= 0, inc.note_prior >= 0
    inc["note"] = inc.note_cur.where(cur_ok, inc.note_prior.where(prior_ok))
    use_intexp = (inc.year >= 2019) & (inc.INTEXP >= 0)
    inc["interest"] = inc.INTEXP.where(use_intexp, inc.note)
    inc["interest_source"] = np.select([use_intexp, cur_ok, prior_ok],
                                       ["INTEXP", "note_current", "note_prior_of_next_year"], "missing")
    both = inc[(inc.year >= 2019) & inc.INTEXP.notna() & inc.note_cur.notna()]
    close = (both.note_cur - both.INTEXP).abs() <= 0.01 * both.INTEXP.abs().clip(lower=1.0)
    summary["interest_sources"] = {
        **note_info,
        "note_vs_INTEXP_2019_2025": {"firm_years": int(len(both)), "within_1pct": float(close.mean())},
        "note_vs_INTEXP_2019_2020": {"firm_years": int((both.year <= 2020).sum()),
                                     "within_1pct": float(close[both.year <= 2020].mean())},
    }

    mr = read("monthly_returns", ["Scode", "Trdmt", "Mmkecap"])
    mr = mr[(mr.Trdmt != "2017-03-10") & (mr.Trdmt.str[5:7] == "12")].copy()
    mr["year"], mr["mcap"] = mr.Trdmt.str[:4].astype(int), num(mr.Mmkecap)
    assert not mr.duplicated(["Scode", "year"]).any()

    cur = bs.merge(inc[["Scode", "year", "NI", "OPREV", "interest", "interest_source"]], on=["Scode", "year"], how="left")
    cur = cur.merge(mr[["Scode", "year", "mcap"]], on=["Scode", "year"], how="left")
    nxt = cur[["Scode", "year", "AT", "ibd", "ibd_nolease", "interest", "interest_source"]].copy()
    nxt["year"] -= 1
    p = cur.merge(nxt, on=["Scode", "year"], how="left", suffixes=("", "_next"))
    prev = cur[["Scode", "year", "OPREV"]].assign(year=lambda x: x.year + 1)
    p = p.merge(prev, on=["Scode", "year"], how="left", suffixes=("", "_prev"))

    info["industry"] = np.where(info.IndcodeB.str[0] == "C", info.IndcodeB.str[:3], info.IndcodeB.str[0])
    p = p.merge(info[["Scode", "IndcodeB", "industry", "Listdt"]], on="Scode", how="left")
    oct31 = h[h.date.str[5:] == "10-31"].assign(year=lambda x: x.date.str[:4].astype(int))
    p = p.merge(oct31[["Scode", "year", "score", "ns", "z", "grade_num", "regime"]], on=["Scode", "year"], how="left")
    q = h.assign(year=h.date.str[:4].astype(int)).groupby(["Scode", "year"]).agg(ns_avg=("ns", "mean"), n_dates=("ns", "size"))
    p = p.merge(q.reset_index(), on=["Scode", "year"], how="left")
    csi = pd.read_csv(CSI300, dtype=str, encoding="utf-8-sig")
    p["csi300"] = p.Scode.isin(set(csi.iloc[:, 4]))

    debt_mean = (p.ibd + p.ibd_next) / 2
    nolease_mean = (p.ibd_nolease + p.ibd_nolease_next) / 2
    p["debt_cost"] = p.interest_next / debt_mean.where(debt_mean > 0)
    p["debt_cost_nolease"] = p.interest_next / nolease_mean.where(nolease_mean > 0)
    p["debt_to_assets"] = debt_mean / p.AT
    p["size"] = np.log(p.AT.where(p.AT > 0))
    p["lev"] = p.LB / p.AT
    p["roa"] = p.NI / p.AT
    p["tang"] = p.PPE / p.AT
    p["cash"] = p.CH / p.AT
    p["growth"] = p.OPREV / p.OPREV_prev.where(p.OPREV_prev > 0) - 1
    listed = p.year - pd.to_datetime(p.Listdt, errors="coerce").dt.year
    p["age"] = np.log1p(listed.where(listed >= 0))                 # years before listing have no age
    p["ibd_to_mcap"] = p.ibd / p.mcap.where(p.mcap > 0)
    p["stbp_share"] = p.STBP / p.ibd.where(p.ibd > 0)
    controls = p[["Scode", "year", "size", "lev", "roa"]].copy()     # every year, for the bond issues
    p = p[p.year.isin(YEARS)].copy()

    # Sample flow of the main specification (contract 26): each step removes firm-years and is counted.
    steps = [
        ("Shanghai or Shenzhen A share", p.Scode.str.match(SHSZ)),
        ("industry code available", p.industry.notna()),
        ("not financial (CSRC J)", p.industry != "J"),
        ("assets > 0 and liabilities <= assets at t", (p.AT > 0) & (p.LB >= 0) & (p.LB <= p.AT)),
        ("balance sheet at t + 1", p.AT_next > 0),
        ("mean interest-bearing debt > 0", debt_mean > 0),
        (f"mean interest-bearing debt >= {MIN_DEBT_TO_ASSETS:.0%} of assets", p.debt_to_assets >= MIN_DEBT_TO_ASSETS),
        ("interest expense of t + 1 available", p.interest_next.notna()),
        ("Huazheng score on 31 October of t", p.ns.notna()),
        ("controls available (size, leverage, ROA)", p[["size", "lev", "roa"]].notna().all(axis=1)),
    ]
    keep, flow = pd.Series(True, index=p.index), []
    for label, cond in steps:
        before = int(keep.sum())
        keep &= cond.fillna(False)
        flow.append({"step": label, "kept": int(keep.sum()), "removed": before - int(keep.sum())})
        if label == "mean interest-bearing debt > 0":
            # Why the 1% floor: the ratio is noisy when the denominator is small (outcome only, no score).
            bins = pd.cut(p.debt_to_assets[keep & p.interest_next.notna()], [0, .005, .01, .02, .05, .1, .2, .4, np.inf],
                          right=False)
            summary["debt_cost_by_debt_to_assets_before_floor"] = {
                str(k): describe(g) for k, g in p.debt_cost[bins.index].groupby(bins, observed=True)}
    summary["sample_flow_main"] = flow
    p["main_sample"] = keep
    p = winsorize(p, ["debt_cost", "debt_cost_nolease", "size", "lev", "roa", "tang", "cash", "growth", "age"], keep)
    return p, controls


def bond_issues(h: pd.DataFrame, controls: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Bonds of listed non-financial firms, matched by the issuer's full name; the spread needs the treasury curve."""
    bb = read("bond_basic", ["Bdcd", "Bdtp", "Valdt", "Bdprd", "Inrttp", "Fisirt", "Amtisn", "Listplc"])
    bi = read("bond_issuer", ["Bdcd", "Sbjd", "Lstcmp", "Ntrcmp"])
    bc = read("bond_class", ["Bdcd", "Mbdorn"])
    br = read("bond_issue_rating", ["Bdcd", "Issbjrt"])
    info = read("company_info", ["Scode", "Coname_cn", "IndcodeB"])
    b = bb.merge(bi, on="Bdcd", how="left").merge(bc, on="Bdcd", how="left").merge(br, on="Bdcd", how="left")
    info = info[info.Scode.str.match(SHSZ)]
    names = info.dropna(subset=["Coname_cn"]).drop_duplicates("Coname_cn", keep=False)
    b = b.merge(names.rename(columns={"Coname_cn": "Sbjd"}), on="Sbjd", how="left")
    b["coupon"], b["maturity"], b["amount"] = num(b.Fisirt), num(b.Bdprd), num(b.Amtisn)
    b["issue_year"] = b.Valdt.str[:4].astype(float)
    b["soe_issuer"] = b.Ntrcmp.fillna("").str.startswith("国有企业")
    steps = [
        ("corporate bond type (企业债/公司债/中期票据/短期融资券)", b.Bdtp.isin(["企业债", "公司债", "中期票据", "短期融资券"])),
        ("issued 2009-02 to 2025-12", (b.Valdt > "2009-01-31") & (b.Valdt <= "2025-12-31")),
        ("fixed coupon", b.Inrttp == "固定利率"),
        ("issuer matched to a Shanghai or Shenzhen A share by full name", b.Scode.notna()),
        ("not financial (CSRC J)", b.IndcodeB.fillna("J").str[0] != "J"),
        ("not a local-government financing bond (城投)", b.Mbdorn != "是"),
        ("coupon, maturity and amount positive", (b.coupon > 0) & (b.maturity > 0) & (b.amount > 0)),
    ]
    keep, flow = pd.Series(True, index=b.index), []
    for label, cond in steps:
        before = int(keep.sum())
        keep &= cond.fillna(False)
        flow.append({"step": label, "kept": int(keep.sum()), "removed": before - int(keep.sum())})
    b = b[keep].copy()
    # Score: the last Huazheng date strictly before the issue date.
    dates = np.array(sorted(h.date.unique()))
    idx = np.searchsorted(dates, b.Valdt.to_numpy(), side="left") - 1
    b["score_date"] = np.where(idx >= 0, dates[np.clip(idx, 0, None)], None)
    b = b.merge(h[["Scode", "date", "ns", "z", "regime"]].rename(columns={"date": "score_date"}),
                on=["Scode", "score_date"], how="left")
    # Controls: the issuer's last annual balance sheet before the issue year.
    b["industry"] = np.where(b.IndcodeB.str[0] == "C", b.IndcodeB.str[:3], b.IndcodeB.str[0])
    b["ctrl_year"] = b.issue_year - 1
    b = b.merge(controls.rename(columns={"year": "ctrl_year"}), on=["Scode", "ctrl_year"], how="left")
    for label, test in [("score before the issue date", lambda x: x.ns.notna()),
                        ("controls of the previous year", lambda x: x[["size", "lev", "roa"]].notna().all(axis=1))]:
        cond = test(b)
        flow.append({"step": label, "kept": int(cond.sum()), "removed": int((~cond).sum())})
        b = b[cond].copy()
    summary["sample_flow_bonds"] = flow
    summary["bonds"] = {
        "by_type": b.Bdtp.value_counts().to_dict(), "by_market": b.Listplc.value_counts().to_dict(),
        "issuers": int(b.Scode.nunique()), "by_issue_year": b.issue_year.astype(int).value_counts().sort_index().to_dict(),
        "issuer_attribute": b.Ntrcmp.value_counts().head(12).to_dict(),
        "issuer_rating_at_issue": b.Issbjrt.fillna("missing").value_counts().head(12).to_dict(),
        "coupon_pct": describe(b.coupon), "maturity_years": describe(b.maturity), "amount_yuan": describe(b.amount),
        "score_regime": b.regime.value_counts().to_dict(),
        "spread": "pending: needs the treasury yield curve (contract 26)",
    }
    return b


def main() -> None:
    summary: dict = {"inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                sorted(CNRDS.glob("*.csv.gz")) + [HUAZHENG, CSI300]}}
    h = huazheng()
    summary["huazheng"] = {
        "rows": int(len(h)), "firms": int(h.Scode.nunique()), "dates": int(h.date.nunique()),
        "normal_score_by_date_mean_sd_range": [float(h.groupby("date").ns.mean().abs().max()),
                                               float(h.groupby("date").ns.std().min()), float(h.groupby("date").ns.std().max())],
        "october_firms_by_year": h[h.date.str[5:] == "10-31"].groupby(h.date.str[:4]).size().to_dict(),
    }
    fy, controls = firm_year(h, summary)
    m = fy[fy.main_sample]
    summary["main_sample"] = {
        "firm_years": int(len(m)), "firms": int(m.Scode.nunique()),
        "by_year": m.groupby("year").size().to_dict(), "by_regime": m.regime.value_counts().to_dict(),
        "industries": int(m.industry.nunique()), "industry_year_cells": int(m.groupby(["industry", "year"]).ngroups),
        "csi300_firms": int(m[m.csi300].Scode.nunique()), "csi300_firm_years": int(m.csi300.sum()),
        "interest_source_of_outcome": m.interest_source_next.value_counts().to_dict(),
        "real_estate_K70_firm_years": int((m.IndcodeB == "K70").sum()),
        "lease_liabilities_positive_share_by_year": m.groupby("year").apply(
            lambda x: float((x[LEASE] > 0).mean()), include_groups=False).to_dict(),
        "stbp_positive_share": float((m.STBP > 0).mean()), "stbp_share_of_debt_when_positive": describe(m.stbp_share[m.STBP > 0]),
        "variables": {c: describe(m[c]) for c in ["debt_cost", "debt_cost_w", "debt_cost_nolease_w", "debt_to_assets",
                                                  "ns", "z", "ns_avg", "score", "size_w", "lev_w", "roa_w", "tang_w",
                                                  "cash_w", "growth_w", "age_w", "ibd_to_mcap"]},
        "debt_cost_by_outcome_year": {int(y) + 1: describe(g.debt_cost) for y, g in m.groupby("year")},
        "missing_extended_controls": {c: float(m[c].isna().mean()) for c in ["tang", "cash", "growth", "age", "mcap"]},
        "ownership": "pending: no ownership field in the CNRDS export (docs/data_inventory.md)",
    }
    # Leverage of the model's reference firms: CSI 300 non-financial constituents at the end of 2025.
    bs25 = read("balance_sheet", ["Scode", "Date", *DEBT, LEASE])
    bs25 = bs25[bs25.Date == "2025-12-31"].copy()
    for c in (*DEBT, LEASE):
        bs25[c] = num(bs25[c]).fillna(0.0)
    bs25["ibd"] = bs25[list(DEBT)].sum(axis=1) + bs25[LEASE]
    mr = read("monthly_returns", ["Scode", "Trdmt", "Mmkecap"])
    mr = mr[mr.Trdmt.str[:7] == "2025-12"].assign(mcap=lambda x: num(x.Mmkecap))
    info = read("company_info", ["Scode", "IndcodeB"])
    csi = set(pd.read_csv(CSI300, dtype=str, encoding="utf-8-sig").iloc[:, 4])
    lam = bs25.merge(mr[["Scode", "mcap"]], on="Scode").merge(info, on="Scode", how="left")
    lam = lam[lam.Scode.isin(csi) & (lam.IndcodeB.str[0] != "J") & (lam.mcap > 0)]
    ratio = lam.ibd / lam.mcap
    summary["leverage_reference_2025"] = {
        "definition": "interest-bearing debt (incl. lease liabilities) at 2025-12-31 / total market value in 2025-12",
        "csi300_nonfinancial": {"firms": int(len(lam)), "median": float(ratio.median()), "mean": float(ratio.mean()),
                                "aggregate": float(lam.ibd.sum() / lam.mcap.sum()), **describe(ratio)},
        "main_sample_median_by_year": m.groupby("year").ibd_to_mcap.median().to_dict(),
    }
    bonds = bond_issues(h, controls, summary)

    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, df in (("firm_year", fy), ("bond_issues", bonds)):
        buf = io.StringIO()
        df.to_csv(buf, index=False, lineterminator="\n")
        raw = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
        (OUT / f"{name}.csv.gz").write_bytes(raw)
        hashes[f"{name}.csv.gz"] = {"sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(df))}
    summary["outputs"] = {"directory": OUT.relative_to(ROOT).as_posix(), **hashes}
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    (HERE / "panel_summary.json").write_bytes((text + "\n").encode("utf-8"))
    print(json.dumps({"main_sample": summary["main_sample"]["firm_years"], "flow": summary["sample_flow_main"],
                      "bonds": summary["sample_flow_bonds"][-1]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
