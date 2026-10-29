"""Correction of K1 (ii): size groups ranked within the main population (contract 28.15).

    python -I -X utf8 k1_size_correction.py

k1_estimate.py --final ranked firms by previous year-end market value among all A shares, financial firms included,
and then kept the non-financial firms, so its size groups are not the top 10% / next 30% / rest of the main
population that contract 28.2 (ii) describes. This script repeats (ii) with the groups ranked within the
non-financial population, with the same panels, events, bootstrap draws and functions as k1_estimate.py; nothing
else changes. Writes k1_size_correction.json (LF).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("k1_estimate", HERE / "k1_estimate.py")
k1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k1)


def main() -> None:
    for name, digest in k1.PANELS.items():
        assert k1.sha(k1.PANEL / name) == digest, name
    e = pd.read_csv(k1.PANEL / "k1_events.csv.gz", dtype={"Scode": str, "Itemid": str})
    fy = pd.read_csv(k1.PANEL / "k1_firm_year.csv.gz", dtype={"Scode": str})
    boot = k1.Boot(pd.Index(sorted(fy.Scode.unique())))          # the same draws as the final run
    nonfin = fy[~fy.financial].copy()
    nonfin["size_group"] = k1.size_groups(nonfin)
    out = {"main": k1.incidence_block(nonfin, e, boot),
           "by_law_period": {"2016-2019": k1.incidence_block(nonfin, e, boot, years=(2016, 2019)),
                             "2020-2025": k1.incidence_block(nonfin, e, boot, years=(2020, 2025))}}
    allfy = fy.copy()
    allfy["size_group"] = k1.size_groups(allfy)
    out["including_financials"] = k1.incidence_block(allfy, e, boot)
    text = json.dumps(k1.clean(out), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (HERE / "k1_size_correction.json").write_bytes(text.encode("utf-8"))
    m = out["main"]
    for g in ("large", "mid", "small"):
        r = m["by_size"][g]["T1_T2"]
        print(g, r["firm_years"], f"{100 * r['rate']:.2f}", [round(100 * x, 2) for x in r["ci95"]])
    for t, r in m["log_ratio_small_over_large"].items():
        print(t, round(r["ratio"], 2), round(r["log_ratio"], 3), [round(x, 3) for x in r["ci95_log"]])
    for p, b in out["by_law_period"].items():
        r = b["log_ratio_small_over_large"]["T1_T2"]
        print(p, round(r["ratio"], 2), [round(x, 3) for x in r["ci95_log"]])


if __name__ == "__main__":
    main()
