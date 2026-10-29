"""Stage K1: disclosure-violation enforcement events and the firm-year population (contract 28.2).

    python -I -X utf8 k1_build.py

Inputs are local and not committed (docs/data_inventory.md): the CNRDS violations table, monthly returns and company
information in data/processed/cnrds_20261026/. Writes data/processed/stage_k_20261029/k1_events.csv.gz (one row per
enforcement event against a listed company itself) and k1_firm_year.csv.gz (the population), and k1_build.json here:
input and output hashes, keyword hit counts and tier counts by year. Nothing here computes an incidence by size, a
fine distribution or any relation to ratings; those are k1_estimate.py --final.

Implementation notes fixed before any estimate (recorded in contract 28.13):
- Handler. Contract 28.2 lists "监管局" (CSRC local offices). Every CSRC office is named "中国证券监督管理委员会…监管局",
  so the securities-regulator test is the regex SECURITIES below; a bare "监管局" would also match 4,353 rows handled
  by the banking and insurance regulators' local offices, which are not CSRC offices.
- Event. Rows sharing Itemid are one event; the tier is the highest over the company's own rows of that event.
- T3 is every remaining measure that is neither T1 nor T2 and not an investigation only, including the "其他" label.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CNRDS = ROOT / "data/processed/cnrds_20261026"
OUT = ROOT / "data/processed/stage_k_20261029"

SHSZ = re.compile(r"^(000|001|002|003|300|301|302|600|601|603|605|688|689)\d{3}$")
SECURITIES = r"证券监督管理委员会|证监会|证监局"                    # the CSRC and its local offices
EXCHANGE = r"证券交易所"
DISCLOSURE = ["披露", "虚假记载", "误导性陈述", "重大遗漏", "业绩预告", "业绩快报", "定期报告"]
ESG_TEXT = ["社会责任报告", "ESG", "可持续发展报告", "环境信息", "排污信息", "碳排放"]
T1_TEXT = r"罚款|警告|没收"
T2_TEXT = r"公开谴责|通报批评|公开批评"
INVESTIGATION = r"立案调查"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def write_csv(df: pd.DataFrame, path: Path) -> str:
    # Fixed gzip header (mtime 0), so the hash depends only on the content.
    import gzip
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    path.write_bytes(buf.getvalue())
    return sha(path)


def events(summary: dict) -> pd.DataFrame:
    v = pd.read_csv(CNRDS / "violations.csv.gz", dtype=str)
    v["text"] = v.VioAct.fillna("")
    v["handler"] = v.Handler.fillna("")
    v["measure"] = v.DispTp.fillna("") + "；" + v.DispMeas.fillna("")
    v["fine"] = num(v.Penalty)                                  # 10,000 yuan
    own = v.Relationship == "公司本身"
    sec = v.handler.str.contains(SECURITIES)
    exch = v.handler.str.contains(EXCHANGE)
    disc = v.text.str.contains("|".join(DISCLOSURE))
    summary["rows"] = {"all": int(len(v)), "company_itself": int(own.sum()),
                       "company_itself_and_csrc_or_exchange": int((own & (sec | exch)).sum()),
                       "company_itself_csrc_or_exchange_disclosure_text": int((own & (sec | exch) & disc).sum()),
                       "bare_监管局_not_securities": int((v.handler.str.contains("监管局") & ~sec & ~exch).sum())}
    base = own & (sec | exch)
    summary["keyword_hits_among_company_itself_csrc_or_exchange"] = {
        k: int((base & v.text.str.contains(re.escape(k))).sum()) for k in DISCLOSURE}
    summary["keyword_unique_hits"] = {          # rows hit by this keyword and by no other
        k: int((base & v.text.str.contains(re.escape(k))
                & ~v.text.str.contains("|".join(re.escape(x) for x in DISCLOSURE if x != k))).sum()) for k in DISCLOSURE}

    rows = v[base & disc].copy()
    rows["is_sec"] = rows.handler.str.contains(SECURITIES)
    rows["t1"] = rows.is_sec & ((rows.fine > 0) | rows.measure.str.contains(T1_TEXT))
    rows["t1_amount"] = rows.is_sec & (rows.fine > 0)
    rows["t2"] = rows.measure.str.contains(T2_TEXT)
    rows["investigation_only"] = (rows.measure.str.contains(INVESTIGATION) & ~rows.t1 & ~rows.t2
                                  & (rows.DispTp.fillna("").str.replace(r"立案调查|,|，|其他", "", regex=True) == ""))
    rows["no_periodic"] = rows.text.str.contains("|".join(x for x in DISCLOSURE if x != "定期报告"))
    rows["esg"] = rows.text.str.contains("|".join(ESG_TEXT))
    rows["fine_company"] = rows.fine.where(rows.fine > 0, 0.0)
    g = rows.groupby("Itemid")
    e = pd.DataFrame({
        "Scode": g.SCode.first(), "n_codes": g.SCode.nunique(), "date": g.AnncDate.min(),
        "t1": g.t1.any(), "t1_amount": g.t1_amount.any(), "t2": g.t2.any(),
        "investigation_only": g.investigation_only.all(), "no_periodic": g.no_periodic.any(), "esg": g.esg.any(),
        "fine_wan": g.fine_company.sum(), "rows": g.size(),
        "handler_exchange": g.handler.apply(lambda s: bool(s.str.contains(EXCHANGE).any())),
        "disp_types": g.DispTp.apply(lambda s: "|".join(sorted(set(s.fillna(""))))),
    }).reset_index()
    assert (e.n_codes == 1).all(), "an event spans several stock codes"
    e["tier"] = np.select([e.t1, e.t2, e.investigation_only], ["T1", "T2", "investigation"], "T3")
    e["year"] = e.date.str[:4].astype(int)
    summary["events"] = {"rows_used": int(len(rows)), "events": int(len(e)), "events_with_several_rows": int((e.rows > 1).sum())}
    summary["tier_counts_by_year"] = {str(y): {t: int(n) for t, n in grp.tier.value_counts().sort_index().items()}
                                      for y, grp in e.groupby("year")}
    summary["tier_counts_total"] = {t: int(n) for t, n in e.tier.value_counts().sort_index().items()}
    summary["t3_disp_types_top"] = {k: int(n) for k, n in e[e.tier == "T3"].disp_types.value_counts().head(25).items()}
    summary["esg_events"] = e[e.esg][["Itemid", "Scode", "date", "tier", "fine_wan"]].to_dict("records")
    return e.drop(columns=["n_codes"])


def population(summary: dict) -> pd.DataFrame:
    mr = pd.read_csv(CNRDS / "monthly_returns.csv.gz", dtype=str, usecols=["Scode", "Trdmt", "Mret", "Mmkecap"])
    mr = mr[mr.Trdmt != "2017-03-10"].copy()                     # mid-month snapshot (data_inventory.md)
    mr["year"], mr["month"] = mr.Trdmt.str[:4].astype(int), mr.Trdmt.str[5:7].astype(int)
    mr["ret"], mr["mcap"] = num(mr.Mret), num(mr.Mmkecap)
    assert not mr.duplicated(["Scode", "year", "month"]).any()
    fy = mr[mr.ret.notna()].groupby(["Scode", "year"]).size().rename("months_with_return").reset_index()
    dec = mr[mr.month == 12][["Scode", "year", "mcap"]].rename(columns={"mcap": "mcap_dec"})
    dec["year"] += 1                                             # previous year-end market value
    fy = fy.merge(dec.rename(columns={"mcap_dec": "mcap_prev_dec"}), on=["Scode", "year"], how="left")
    info = pd.read_csv(CNRDS / "company_info.csv.gz", dtype=str, usecols=["Scode", "IndcodeB"])
    fy = fy.merge(info, on="Scode", how="left")
    fy["industry"] = np.where(fy.IndcodeB.str[0] == "C", fy.IndcodeB.str[:3], fy.IndcodeB.str[0])
    fy.loc[fy.IndcodeB.isna(), "industry"] = np.nan
    fy["shsz_a"] = fy.Scode.str.match(SHSZ)
    fy["financial"] = fy.industry == "J"
    fy["active"] = fy.months_with_return >= 6
    keep = fy.shsz_a & fy.active & fy.year.between(2008, 2025) & fy.industry.notna()
    fy = fy[keep].drop(columns=["IndcodeB", "shsz_a", "active"]).reset_index(drop=True)
    summary["population"] = {str(y): {"firms": int(len(g)), "non_financial": int((~g.financial).sum()),
                                      "with_prev_dec_mcap": int(g.mcap_prev_dec.notna().sum())}
                             for y, g in fy.groupby("year")}
    return fy


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"inputs": {name: sha(CNRDS / f"{name}.csv.gz")
                                for name in ("violations", "monthly_returns", "company_info")}}
    e = events(summary)
    fy = population(summary)
    inpop = e.merge(fy[["Scode", "year", "financial"]], on=["Scode", "year"], how="left", indicator=True)
    summary["events_in_population"] = {
        "in_population": int((inpop._merge == "both").sum()),
        "in_population_non_financial": int(((inpop._merge == "both") & (inpop.financial == False)).sum()),  # noqa: E712
        "not_in_population": int((inpop._merge == "left_only").sum())}
    summary["outputs"] = {"k1_events.csv.gz": write_csv(e, OUT / "k1_events.csv.gz"),
                          "k1_firm_year.csv.gz": write_csv(fy, OUT / "k1_firm_year.csv.gz")}
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (HERE / "k1_build.json").write_bytes(text.encode("utf-8"))
    print(json.dumps({k: summary[k] for k in ("rows", "keyword_hits_among_company_itself_csrc_or_exchange",
                                              "keyword_unique_hits", "events", "tier_counts_total",
                                              "events_in_population", "outputs")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
