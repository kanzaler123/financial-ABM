"""Stage J panels: Huazheng rating changes, the stock returns after them and year-end fund ownership (contract 27).

    python -I -X utf8 build_panel.py

Runs on the system Python (pandas, scipy, openpyxl). Inputs are local and not committed: the CNRDS tables in
data/processed/cnrds_20261026/ (docs/data_inventory.md), the CNRDS year-end institutional holdings of 2026-10-27 and
the netdisk Huazheng quarterly panel. Writes the panels to data/processed/stage_j_20261028/ (not committed) and
panel_summary.json here: input and panel hashes, the sample flow, coverage and the distribution of every variable,
each on its own. Nothing here relates a rating change to a return or to a holding.

Dates. The panel's dates (31 January, 30 April, 31 July, 31 October) are Huazheng's rating dates; Huazheng releases
the update "within 20 trading days" of the rating date (official FAQ, chindices.com/esg-faq.html, read 2026-10-28).
So the month of the rating date (event month 0) ends before the release, and the release falls in month +1 or +2.

Vintage. The netdisk panel is not a point-in-time record for every date (contract 27.1): Huazheng launched the
product in 2018, its own tracking report for 2018-10-31 shows a different distribution from the panel's 2018-10-31,
and the older netdisk copy holds other ratings for 2021-04 to 2022-07. PERIOD labels each rating change accordingly.
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
HOLDINGS = ROOT / "data/raw/cnrds_20261027/机构持股/机构持股.csv"      # CNRDS, year-end institutional holdings
OUT = ROOT / "data/processed/stage_j_20261028"

SHSZ = re.compile(r"^(000|001|002|003|300|301|302|600|601|603|605|688|689)\d{3}$")
GRADES = ["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
METHOD_CHANGE_DATES = {"2021-04-30", "2024-04-30"}       # breaks in the panel (huazheng_panel_dynamics_20261026)


def period(date: str) -> str:
    """Vintage class of a rating change dated `date` (the change from the previous rating date to this one)."""
    if date <= "2017-10-31":
        return "prelaunch"            # both ratings back-calculated; Huazheng launched the product in 2018
    if date <= "2018-07-31":
        return "launch_2018"          # launch month unknown: left out of every sample
    if date <= "2022-07-31":
        return "restated"             # public, but the panel's ratings are not the ones released at the time
    if date == "2022-10-31":
        return "switch_2022"          # 2022-07 restated (V2.0 back-calculation), 2022-10 released under V2.0
    return "pit"                      # 2023-01 to 2025-10: both ratings as released (contract 27.1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(CNRDS / f"{name}.csv.gz", dtype=str, usecols=cols)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def normal_scores(values: pd.Series) -> pd.Series:
    ok = values.notna()
    out = pd.Series(np.nan, index=values.index)
    out[ok] = ndtri((rankdata(values[ok]) - 0.5) / ok.sum())
    return out


def describe(s: pd.Series) -> dict:
    s = pd.Series(s).dropna()
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).tolist() if len(s) else []
    return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None, "sd": float(s.std()) if len(s) > 1 else None,
            "quantiles_0_1_5_25_50_75_95_99_100": [float(x) for x in q]}


def huazheng() -> pd.DataFrame:
    import openpyxl

    rows = list(openpyxl.load_workbook(HUAZHENG, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    assert rows[0][:5] == ("证券代码", "季度日期", "证券简称", "综合评级", "综合得分"), rows[0]
    h = pd.DataFrame([(f"{r[0]:06d}", r[1].date().isoformat(), r[3], float(r[4])) for r in rows[1:]],
                     columns=["Scode", "date", "grade", "score"])
    h["ns"] = h.groupby("date", group_keys=False).score.apply(normal_scores)
    h["notch"] = h.grade.map({g: i + 1 for i, g in enumerate(GRADES)})
    assert h.notch.notna().all() and not h.duplicated(["Scode", "date"]).any()
    return h


def rating_changes(h: pd.DataFrame) -> pd.DataFrame:
    """One row per firm and rating date d with a rating on the previous rating date as well."""
    dates = sorted(h.date.unique())
    prev = dict(zip(dates[1:], dates[:-1]))
    cur = h[h.date.isin(prev)].copy()
    cur["date_prev"] = cur.date.map(prev)
    old = h.rename(columns={"date": "date_prev", "grade": "grade_prev", "score": "score_prev", "ns": "ns_prev",
                            "notch": "notch_prev"})
    e = cur.merge(old, on=["Scode", "date_prev"], how="inner")
    e["dnotch"] = e.notch - e.notch_prev
    e["dscore"] = e.score - e.score_prev
    e["dns"] = e.ns - e.ns_prev
    e["period"] = e.date.map(period)
    e["method_change"] = e.date.isin(METHOD_CHANGE_DATES)
    return e


def monthly() -> pd.DataFrame:
    """Firm x month: total return (Mret includes cash dividends, data_inventory.md) and total market value."""
    mr = read("monthly_returns", ["Scode", "Trdmt", "Mret", "Mmkecap"])
    mr = mr[mr.Trdmt != "2017-03-10"].copy()                 # mid-month snapshot (data_inventory.md)
    mr["month"] = pd.PeriodIndex(mr.Trdmt.str[:7], freq="M")
    mr["ret"], mr["mcap"] = num(mr.Mret) / 100, num(mr.Mmkecap)
    assert not mr.duplicated(["Scode", "month"]).any()
    return mr[["Scode", "month", "ret", "mcap"]]


def compound(ret: pd.DataFrame, offsets: range, min_months: int) -> pd.Series:
    """Compounded return over the given month offsets; missing when fewer than min_months are available."""
    block = ret[[f"r{k:+d}" for k in offsets]]
    out = (1 + block).prod(axis=1, min_count=1) - 1
    return out.where(block.notna().sum(axis=1) >= min_months)


def book_equity() -> pd.DataFrame:
    bs = read("balance_sheet", ["Scode", "Date", "EAPC", "EQU"])
    bs = bs[bs.Date.str.endswith("12-31")].copy()
    bs["fy"] = bs.Date.str[:4].astype(int)
    bs["book"] = num(bs.EAPC).fillna(num(bs.EQU))           # equity attributable to the parent; total equity if blank
    assert not bs.duplicated(["Scode", "fy"]).any()
    return bs[["Scode", "fy", "book"]]


def event_panel(e: pd.DataFrame, mr: pd.DataFrame, summary: dict) -> pd.DataFrame:
    info = read("company_info", ["Scode", "Listdt", "IndcodeB"])
    info["industry"] = np.where(info.IndcodeB.str[0] == "C", info.IndcodeB.str[:3], info.IndcodeB.str[0])
    e = e.merge(info[["Scode", "Listdt", "IndcodeB", "industry"]], on="Scode", how="left")
    e["m0"] = pd.PeriodIndex(e.date.str[:7], freq="M")
    ret = mr.set_index(["Scode", "month"]).ret
    mcap = mr.set_index(["Scode", "month"]).mcap
    for k in range(-11, 7):
        key = pd.MultiIndex.from_arrays([e.Scode, e.m0 + k])
        e[f"r{k:+d}"] = ret.reindex(key).to_numpy()
    e["mcap"] = mcap.reindex(pd.MultiIndex.from_arrays([e.Scode, e.m0])).to_numpy()
    e["car_1_2"] = compound(e, range(1, 3), 2)          # release window: months +1 and +2
    e["car_3_6"] = compound(e, range(3, 7), 4)          # drift after the release (descriptive)
    e["ret_q"] = compound(e, range(-2, 1), 3)           # the rating quarter: months -2 to 0
    e["mom"] = compound(e, range(-11, -2), 6)           # months -11 to -3, at least 6 available
    e["size"] = np.log(e.mcap.where(e.mcap > 0))
    # Book equity of the latest fiscal year published at least five months before the rating date.
    e["fy"] = e.date.str[:4].astype(int) - np.where(e.date.str[5:7].astype(int) >= 6, 1, 2)
    e = e.merge(book_equity(), on=["Scode", "fy"], how="left")
    e["bm"] = e.book / e.mcap.where(e.mcap > 0)
    e["log_bm"] = np.log(e.bm.where(e.bm > 0))
    listed = (pd.to_datetime(e.date) - pd.to_datetime(e.Listdt, errors="coerce")).dt.days

    steps = [
        ("Shanghai or Shenzhen A share", e.Scode.str.match(SHSZ)),
        ("industry code available", e.industry.notna()),
        ("not financial (CSRC J)", e.industry != "J"),
        ("listed at least 365 days before the rating date", listed >= 365),
        ("market value at month 0", e["size"].notna()),
        ("book equity > 0", e.log_bm.notna()),
        ("returns of months -2..0 and >= 6 of months -11..-3", e.ret_q.notna() & e.mom.notna()),
        ("returns of months +1 and +2", e.car_1_2.notna()),
    ]
    keep, flow, flow_pit = pd.Series(True, index=e.index), [], []
    pit = e.period == "pit"
    for label, cond in steps:
        before, before_pit = int(keep.sum()), int((keep & pit).sum())
        keep &= cond.fillna(False)
        flow.append({"step": label, "kept": int(keep.sum()), "removed": before - int(keep.sum())})
        flow_pit.append({"step": label, "kept": int((keep & pit).sum()), "removed": before_pit - int((keep & pit).sum())})
    e["in_sample"] = keep
    summary["event_sample_flow_all_periods"] = flow
    summary["event_sample_flow_pit"] = flow_pit
    return e


def holdings() -> pd.DataFrame:
    """Year-end fund ownership as a percentage of tradable A shares; no fund holder counts as 0."""
    o = pd.read_csv(HOLDINGS, dtype=str, encoding="utf-8-sig")
    o = o[o.SCode != "股票代码"]                       # the export's second header row holds the Chinese labels
    assert o.SCode.str.fullmatch(r"\d{6}").all() and not o.duplicated(["SCode", "Date"]).any()
    assert o.Date.str.endswith("12-31").all()
    n, rt = num(o.FShrhN), num(o.FCSRt)
    fo = rt.where(~(n == 0), 0.0)
    return pd.DataFrame({"Scode": o.SCode, "year": o.Date.str[:4].astype(int), "fund_holders": n, "fund_own": fo,
                         "fund_mv": num(o.FPosnMV), "float_shares": num(o.TlCShrN)})


def annual_panel(h: pd.DataFrame, mr: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Firm-year y: change in year-end fund ownership (December y-1 to December y) and the rating change from
    31 October y-1 to 31 October y (released by early December y)."""
    octo = h[h.date.str[5:] == "10-31"].assign(year=lambda x: x.date.str[:4].astype(int))
    cur = octo[["Scode", "year", "notch", "ns", "score"]]
    prv = cur.assign(year=cur.year + 1).rename(columns={"notch": "notch_prev", "ns": "ns_prev", "score": "score_prev"})
    a = cur.merge(prv, on=["Scode", "year"], how="inner")
    a["dnotch"], a["dns"] = a.notch - a.notch_prev, a.ns - a.ns_prev
    # Year y compares October y-1 with October y: prelaunch through 2017, launch year 2018 (October 2017 is
    # back-calculated), restated 2019-2022 (October 2021 is a V2.0 back-calculation), as released from 2023.
    a["period"] = np.select([a.year <= 2017, a.year == 2018, a.year <= 2022], ["prelaunch", "launch_2018", "restated"], "pit")
    ho = holdings()
    a = a.merge(ho, on=["Scode", "year"], how="left")
    a = a.merge(ho[["Scode", "year", "fund_own"]].assign(year=lambda x: x.year + 1)
                .rename(columns={"fund_own": "fund_own_prev"}), on=["Scode", "year"], how="left")
    a["dfund_own"] = a.fund_own - a.fund_own_prev
    info = read("company_info", ["Scode", "Listdt", "IndcodeB"])
    info["industry"] = np.where(info.IndcodeB.str[0] == "C", info.IndcodeB.str[:3], info.IndcodeB.str[0])
    a = a.merge(info[["Scode", "Listdt", "industry"]], on="Scode", how="left")
    m = mr.set_index(["Scode", "month"])
    dec_prev = pd.PeriodIndex([f"{y - 1}-12" for y in a.year], freq="M")
    a["mcap_prev"] = m.mcap.reindex(pd.MultiIndex.from_arrays([a.Scode, dec_prev])).to_numpy()
    a["size"] = np.log(a.mcap_prev.where(a.mcap_prev > 0))
    rets = np.column_stack([m.ret.reindex(pd.MultiIndex.from_arrays([a.Scode, pd.PeriodIndex([f"{y}-10" for y in a.year], freq="M") - k])).to_numpy()
                            for k in range(12)])
    ok = np.isfinite(rets).sum(axis=1) >= 9
    a["ret_12"] = np.where(ok, np.nanprod(1 + rets, axis=1) - 1, np.nan)    # November y-1 to October y
    be = book_equity().rename(columns={"fy": "year"})
    a = a.merge(be.assign(year=be.year + 2), on=["Scode", "year"], how="left")   # fiscal year y-2: known by Dec y-1
    a["log_bm"] = np.log((a.book / a.mcap_prev).where((a.book > 0) & (a.mcap_prev > 0)))
    listed = (pd.to_datetime(pd.Series([f"{y - 1}-12-31" for y in a.year], index=a.index))
              - pd.to_datetime(a.Listdt, errors="coerce")).dt.days
    steps = [
        ("Shanghai or Shenzhen A share", a.Scode.str.match(SHSZ)),
        ("industry code available, not financial", a.industry.notna() & (a.industry != "J")),
        ("listed at least 365 days before December y-1", listed >= 365),
        ("fund ownership at December y-1 and y", a.fund_own.notna() & a.fund_own_prev.notna()),
        ("size, book-to-market and >= 9 of 12 monthly returns", a[["size", "log_bm", "ret_12"]].notna().all(axis=1)),
    ]
    keep, flow = pd.Series(True, index=a.index), []
    for label, cond in steps:
        before = int(keep.sum())
        keep &= cond.fillna(False)
        flow.append({"step": label, "kept": int(keep.sum()), "removed": before - int(keep.sum())})
    a["in_sample"] = keep
    summary["annual_sample_flow"] = flow
    return a


def main() -> None:
    summary: dict = {"inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                [CNRDS / "monthly_returns.csv.gz", CNRDS / "company_info.csv.gz",
                                 CNRDS / "balance_sheet.csv.gz", HUAZHENG, HOLDINGS]}}
    h = huazheng()
    e = rating_changes(h)
    summary["rating_changes"] = {
        "rows": int(len(e)), "dates": int(e.date.nunique()),
        "dates_by_period": e.groupby("period").date.agg(lambda x: sorted(set(x))).to_dict(),
        "by_period": {p: {"changes": int(len(g)), "firms": int(g.Scode.nunique()),
                          "dnotch_counts": g.dnotch.value_counts().sort_index().to_dict(),
                          "share_changed": float((g.dnotch != 0).mean()), "dns": describe(g.dns),
                          "dscore": describe(g.dscore), "corr_dnotch_dns": float(g.dnotch.corr(g.dns))}
                      for p, g in e.groupby("period")},
        "pit_by_date": {d: {"n": int(len(g)), "up": int((g.dnotch > 0).sum()), "down": int((g.dnotch < 0).sum()),
                            "sd_dnotch": float(g.dnotch.std()), "sd_dns": float(g.dns.std())}
                        for d, g in e[e.period == "pit"].groupby("date")},
    }
    mr = monthly()
    summary["monthly_returns"] = {"last_month": str(mr.month.max()), "rows": int(len(mr))}
    ev = event_panel(e, mr, summary)
    s = ev[ev.in_sample]
    summary["event_sample"] = {
        p: {"events": int(len(g)), "firms": int(g.Scode.nunique()), "dates": int(g.date.nunique()),
            "industries": int(g.industry.nunique()), "date_industry_cells": int(g.groupby(["date", "industry"]).ngroups),
            "share_changed": float((g.dnotch != 0).mean()), "up": int((g.dnotch > 0).sum()), "down": int((g.dnotch < 0).sum()),
            "dnotch_counts": g.dnotch.value_counts().sort_index().to_dict(),
            "variables": {c: describe(g[c]) for c in ["car_1_2", "car_3_6", "r+1", "r+2", "ret_q", "mom", "size", "log_bm",
                                                      "dnotch", "dns", "notch_prev"]},
            "car_1_2_missing_months_6_coverage": float(g.car_3_6.notna().mean())}
        for p, g in s.groupby("period")}
    summary["event_sample_pit_by_date"] = {
        d: {"events": int(len(g)), "up": int((g.dnotch > 0).sum()), "down": int((g.dnotch < 0).sum()),
            "car_1_2_sd": float(g.car_1_2.std())}
        for d, g in s[s.period == "pit"].groupby("date")}
    an = annual_panel(h, mr, summary)
    sa = an[an.in_sample]
    summary["annual_sample"] = {
        p: {"firm_years": int(len(g)), "firms": int(g.Scode.nunique()), "years": sorted(int(y) for y in g.year.unique()),
            "variables": {c: describe(g[c]) for c in ["dfund_own", "fund_own_prev", "fund_own", "dnotch", "dns", "size",
                                                      "log_bm", "ret_12"]},
            "zero_fund_own_prev_share": float((g.fund_own_prev == 0).mean())}
        for p, g in sa.groupby("period")}

    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    keep_cols = ["Scode", "date", "date_prev", "period", "method_change", "grade", "grade_prev", "notch", "notch_prev",
                 "dnotch", "score", "score_prev", "dscore", "ns", "ns_prev", "dns", "industry", "IndcodeB", "mcap", "size",
                 "book", "log_bm", "ret_q", "mom", "car_1_2", "car_3_6", *[f"r{k:+d}" for k in range(-2, 7)], "in_sample"]
    for name, df in (("events", ev[keep_cols].assign(m0=ev.m0.astype(str))), ("annual_fund_ownership", an)):
        buf = io.StringIO()
        df.to_csv(buf, index=False, lineterminator="\n")
        raw = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
        (OUT / f"{name}.csv.gz").write_bytes(raw)
        hashes[f"{name}.csv.gz"] = {"sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(df))}
    summary["outputs"] = {"directory": OUT.relative_to(ROOT).as_posix(), **hashes}
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    (HERE / "panel_summary.json").write_bytes((text + "\n").encode("utf-8"))
    print(json.dumps({"flow": summary["event_sample_flow_all_periods"], "flow_pit": summary["event_sample_flow_pit"],
                      "events": {p: v["events"] for p, v in summary["event_sample"].items()},
                      "annual": summary["annual_sample_flow"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
