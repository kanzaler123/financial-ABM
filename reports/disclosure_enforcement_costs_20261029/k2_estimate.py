"""Stage K2 (c): size gradient of ex-post confirmed disclosure violations (contract 28.3, 28.19; descriptive).

    python -I -X utf8 k2_estimate.py --final

Unit: firm x year t, t = 2015-2020, firms with a Huazheng rating on 31 October t that belong to the K1 main
population in year t (Shanghai or Shenzhen A share, non-financial, at least 6 months of returns). D_c = 1 when the
firm has a T1 or T2 event (k1_events.csv.gz) announced after 31 October t and on or before 31 October t + 2. Size
groups: top 10% / next 30% / rest of total market value at the end of October t among the sample firms of year t.
Estimate: D_c on large and mid dummies (small is the base) with year x industry fixed effects (within transform),
firm-clustered CR1 standard errors, t distribution with G - 1 degrees of freedom. Reported: large minus small (the
gradient), mid minus small, and the group means. Runs once: refuses when k2_estimates.json exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT / "data/processed/stage_k_20261029"
CNRDS = ROOT / "data/processed/cnrds_20261026"
SOURCE = ROOT / "data/raw/huazheng_panel_netdisk_20261026/huazheng_quarterly_2009_2025.xlsx"
PANELS = {"k1_events.csv.gz": "9e18ade4478d9a7036a9bba2e7056b0c600115157a9ef947da7219e8cb7670a4",
          "k1_firm_year.csv.gz": "502873098264df5cea39d996778f1a817976ccf07f724cdbadd2be50a021d3a9"}
OUTPUT = HERE / "k2_estimates.json"
YEARS = (2015, 2020)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel() -> pd.DataFrame:
    import openpyxl

    rows = list(openpyxl.load_workbook(SOURCE, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    h = pd.DataFrame([(f"{r[0]:06d}", r[1].date().isoformat()) for r in rows[1:]], columns=["Scode", "date"])
    h = h[h.date.str.endswith("-10-31")].copy()
    h["year"] = h.date.str[:4].astype(int)
    h = h[h.year.between(*YEARS)]
    fy = pd.read_csv(PANEL / "k1_firm_year.csv.gz", dtype={"Scode": str})
    fy = fy[~fy.financial][["Scode", "year", "industry"]]
    df = h.merge(fy, on=["Scode", "year"], how="inner")
    mr = pd.read_csv(CNRDS / "monthly_returns.csv.gz", dtype=str, usecols=["Scode", "Trdmt", "Mmkecap"])
    mr = mr[mr.Trdmt.str[5:7] == "10"]
    mr = pd.DataFrame({"Scode": mr.Scode, "year": mr.Trdmt.str[:4].astype(int),
                       "mcap": pd.to_numeric(mr.Mmkecap, errors="coerce")})
    df = df.merge(mr, on=["Scode", "year"], how="left")
    df = df[df.mcap > 0].copy()
    r = df.groupby("year").mcap.rank(ascending=False, method="first") / df.groupby("year").Scode.transform("size")
    df["group"] = np.where(r <= 0.10, "large", np.where(r <= 0.40, "mid", "small"))
    ev = pd.read_csv(PANEL / "k1_events.csv.gz", dtype={"Scode": str})
    ev = ev[ev.tier.isin(["T1", "T2"])][["Scode", "date"]]
    m = df[["Scode", "year"]].merge(ev, on="Scode", how="left")
    m["hit"] = (m.date > m.year.astype(str) + "-10-31") & (m.date <= (m.year + 2).astype(str) + "-10-31")
    hit = m.groupby(["Scode", "year"]).hit.any().rename("D_c").reset_index()
    return df.merge(hit, on=["Scode", "year"], how="left")


def within(df: pd.DataFrame, cols: list[str], cell: pd.Series) -> pd.DataFrame:
    return df[cols] - df.groupby(cell)[cols].transform("mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    if not ap.parse_args().final:
        raise SystemExit("run with --final (once)")
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT.name} exists; the final run is done")
    for name, digest in PANELS.items():
        if sha(PANEL / name) != digest:
            raise SystemExit(f"{name}: hash differs from the recorded panel")
    df = panel()
    df["y"] = df.D_c.astype(float)
    df["large"] = (df.group == "large").astype(float)
    df["mid"] = (df.group == "mid").astype(float)
    cell = df.year.astype(str) + "_" + df.industry.astype(str)
    df = df[cell.map(cell.value_counts()) > 1].copy()            # drop singleton cells
    cell = df.year.astype(str) + "_" + df.industry.astype(str)
    X = within(df, ["large", "mid"], cell).to_numpy()
    y = within(df, ["y"], cell).to_numpy()[:, 0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    groups = df.Scode.to_numpy()
    scores = pd.DataFrame(X * e[:, None]).groupby(groups).sum().to_numpy()
    G, n, k = len(scores), len(y), X.shape[1] + cell.nunique()
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    tcrit = stats.t.ppf(0.975, G - 1)
    out = {"contract": "28.3, 28.19", "years": list(YEARS), "firm_years": n, "firms": G, "cells": int(cell.nunique()),
           "group_means": {g: {"rate": float(d.y.mean()), "firm_years": int(len(d))} for g, d in df.groupby("group")},
           "large_minus_small": {"estimate": float(beta[0]), "se": float(se[0]),
                                 "ci95": [float(beta[0] - tcrit * se[0]), float(beta[0] + tcrit * se[0])],
                                 "p": float(2 * stats.t.sf(abs(beta[0] / se[0]), G - 1))},
           "mid_minus_small": {"estimate": float(beta[1]), "se": float(se[1]),
                               "ci95": [float(beta[1] - tcrit * se[1]), float(beta[1] + tcrit * se[1])]},
           "panels": PANELS, "huazheng_sha256": sha(SOURCE)}
    OUTPUT.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
