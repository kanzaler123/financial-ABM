"""Rating dynamics of the Huazheng ESG panel (2009-01 to 2025-10), split by rating method.

Input: the netdisk copy in data/raw/huazheng_panel_netdisk_20261026/ (not committed); the older 2009-2022 copy in
data/raw/huazheng_panel_netdisk_20261013/ for the comparison; the official-site CSI 300 snapshot of 2026-09-24 for the
spot check. Output: aggregate statistics only (summary.json). Grades are mapped C=1 ... AAA=9.

The panel has three rating methods, separated by breaks at 2021-01 -> 2021-04 and 2024-01 -> 2024-04 (see README).
Persistence is measured inside each method only. kappa follows the paper's estimator (contract section 584):
kappa_lag1 = -4 ln(rho_1) and the noise-corrected kappa_ratio = -(4/3) ln(rho_4 / rho_1), where rho_k is the pooled
correlation between ratings k observation dates (quarters) apart.
"""
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/huazheng_panel_netdisk_20261026"
QUARTERLY = RAW / "huazheng_quarterly_2009_2025.xlsx"
ANNUAL = RAW / "huazheng_annual_2009_2025.xlsx"
OLD = ROOT / "data/raw/huazheng_panel_netdisk_20261013/huazheng_2009_2022.xlsx"
SNAPSHOT = ROOT / "reports/disclosure_public_data_20261011/csi300_huazheng_latest/audited_observations.csv"
GRADES = ["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
MAP = {g: i + 1 for i, g in enumerate(GRADES)}
REGIMES = {"R1": ("2009-01-31", "2021-01-31"), "R2": ("2021-04-30", "2024-01-31"), "R3": ("2024-04-30", "2025-10-31")}
LAGS = (1, 2, 4, 8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path, sheet=None):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    return list((book[sheet] if sheet else book.worksheets[0]).iter_rows(values_only=True))


def corr(X, lag):
    a, b = X[:, :-lag], X[:, lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else None, int(ok.sum())


def kappas(r1, r4):
    if r1 is None or r4 is None:
        return None
    k1, k2 = -4 * math.log(r1), -4 / 3 * math.log(r4 / r1)
    return {"kappa_lag1": k1, "kappa_ratio": k2, "half_life_years_lag1": math.log(2) / k1,
            "half_life_years_ratio": math.log(2) / k2, "implied_reliability": r1 / (r4 / r1) ** (1 / 3)}


def regime_stats(G, S, dates, cols):
    g, s = G[:, cols], S[:, cols]
    change = g[:, 1:] - g[:, :-1]
    ok = np.isfinite(change)
    d, level = change[ok], g[:, :-1][ok]
    slope = np.polyfit(level, g[:, 1:][ok], 1)[0]
    grade_acf = {lag: corr(g, lag) for lag in LAGS if lag < len(cols)}
    score_acf = {lag: corr(s, lag) for lag in LAGS if lag < len(cols)}
    years = sorted({dates[j].year for j in cols})
    return {
        "first": dates[cols[0]].isoformat(), "last": dates[cols[-1]].isoformat(), "dates": len(cols),
        "firm_dates": int(np.isfinite(g).sum()), "mean_grade": float(np.nanmean(g)),
        "mean_score": float(np.nanmean(s)), "sd_score_mean_by_date": float(np.nanmean(np.nanstd(s, axis=0))),
        "grade_counts": {GRADES[k - 1]: int((g == k).sum()) for k in range(1, 10)},
        "quarter_pairs": int(d.size), "p_change": float(np.mean(d != 0)), "p_up": float(np.mean(d > 0)),
        "p_down": float(np.mean(d < 0)), "mean_change_notches": float(d.mean()), "sd_change_notches": float(d.std()),
        "abs_change_counts": {int(k): int(v) for k, v in sorted(collections.Counter(np.abs(d).astype(int)).items())},
        "ols_slope_next_on_current": float(slope),
        "grade_acf": {lag: {"rho": v[0], "pairs": v[1]} for lag, v in grade_acf.items()},
        "score_acf": {lag: {"rho": v[0], "pairs": v[1]} for lag, v in score_acf.items()},
        "kappa_grade": kappas(grade_acf.get(1, (None,))[0], grade_acf.get(4, (None,))[0]),
        "kappa_score": kappas(score_acf.get(1, (None,))[0], score_acf.get(4, (None,))[0]),
        "expected_change_by_grade": {GRADES[k - 1]: float(d[level == k].mean()) for k in range(1, 10) if (level == k).sum() > 100},
        "yearly_mean_grade": {y: float(np.nanmean(g[:, [i for i, j in enumerate(cols) if dates[j].year == y]])) for y in years},
    }


def main():
    q = rows(QUARTERLY)
    header, body = q[0], q[1:]
    assert len(header) == 14 and all(r[3] in MAP for r in body)
    firms = sorted({r[0] for r in body})
    dates = sorted({r[1].date() for r in body})
    fi, di = {c: i for i, c in enumerate(firms)}, {d: j for j, d in enumerate(dates)}
    G = np.full((len(firms), len(dates)), np.nan)
    S = G.copy()
    for r in body:
        G[fi[r[0]], di[r[1].date()]] = MAP[r[3]]
        S[fi[r[0]], di[r[1].date()]] = r[4]
    duplicates = len(body) - int(np.isfinite(G).sum())
    month_days = collections.Counter((d.month, d.day) for d in dates)

    # Breaks: every adjacent pair of observation dates.
    adjacent = []
    for j in range(len(dates) - 1):
        ok = np.isfinite(G[:, j]) & np.isfinite(G[:, j + 1])
        adjacent.append({"from": dates[j].isoformat(), "to": dates[j + 1].isoformat(), "pairs": int(ok.sum()),
                         "same_grade": float(np.mean(G[ok, j] == G[ok, j + 1])),
                         "score_corr": float(np.corrcoef(S[ok, j], S[ok, j + 1])[0, 1]),
                         "score_sd_to": float(np.nanstd(S[:, j + 1]))})
    regime_cols = {k: [di[dt.date.fromisoformat(a)] + i for i in range(di[dt.date.fromisoformat(b)] - di[dt.date.fromisoformat(a)] + 1)]
                   for k, (a, b) in REGIMES.items()}
    regimes = {k: regime_stats(G, S, dates, cols) for k, cols in regime_cols.items()}
    r1 = regime_cols["R1"]
    season = collections.defaultdict(list)
    for j in range(r1[0], r1[-1]):
        season[f"{dates[j].month:02d}->{dates[j + 1].month:02d}"].append(adjacent[j]["same_grade"])
    bands = {}
    for k, cols in regime_cols.items():
        g, s = G[:, cols], S[:, cols]
        bands[k] = {GRADES[m - 1]: [float(np.nanmin(s[g == m])), float(np.nanmax(s[g == m]))] for m in range(1, 10) if (g == m).any()}

    # The annual file should be the October rows of the quarterly file.
    annual = rows(ANNUAL)[1:]
    annual_match = sum(MAP[r[3]] == G[fi[r[0]], di[dt.date(r[1], 10, 31)]] and r[4] == S[fi[r[0]], di[dt.date(r[1], 10, 31)]]
                       for r in annual)

    # The older 2009-2022 copy: which new date does each of its columns reproduce?
    old = rows(OLD, "季度数据")
    old_cols = [str(h).strip()[:10] for h in old[0][2:]]
    old_firms = {int(str(r[0]).split(".")[0]): r for r in old[1:] if str(r[0]).split(".")[0].isdigit()}
    old_vs_new = []
    for j, label in enumerate(old_cols):
        label_date = dt.date.fromisoformat(label)
        best = None
        for d in dates:
            if abs((d - label_date).days) > 400:
                continue
            pairs = [(G[fi[c], di[d]], MAP[r[j + 2]]) for c, r in old_firms.items() if c in fi and r[j + 2] in MAP and np.isfinite(G[fi[c], di[d]])]
            share = float(np.mean([a == b for a, b in pairs]))
            if best is None or share > best[1]:
                best = (d.isoformat(), share, len(pairs))
        old_vs_new.append({"old_label": label, "best_new_date": best[0], "same_grade": best[1], "pairs": best[2]})

    # Spot check of the last date against the official-site CSI 300 snapshot of 2026-09-24.
    with open(SNAPSHOT, encoding="utf-8-sig") as f:
        snap = {int(r["ticker"][:6]): r for r in csv.DictReader(f) if r["status"] == "observed"}
    last = di[dates[-1]]
    common = [c for c in snap if c in fi and np.isfinite(G[fi[c], last])]
    a = np.array([G[fi[c], last] for c in common])
    b = np.array([MAP[snap[c]["grade"]] for c in common])
    sa = np.array([S[fi[c], last] for c in common])
    sb = np.array([float(snap[c]["score_native"]) for c in common])

    summary = {
        "source": {"quarterly": {"file": QUARTERLY.relative_to(ROOT).as_posix(), "sha256": sha(QUARTERLY)},
                   "annual": {"file": ANNUAL.relative_to(ROOT).as_posix(), "sha256": sha(ANNUAL)},
                   "old_copy": {"file": OLD.relative_to(ROOT).as_posix(), "sha256": sha(OLD)},
                   "provenance": "Netdisk copies; not exported by us from Wind or CSMAR."},
        "coverage": {"rows": len(body), "firms": len(firms), "dates": len(dates), "first": dates[0].isoformat(),
                     "last": dates[-1].isoformat(), "month_days": {f"{m:02d}-{d:02d}": n for (m, d), n in sorted(month_days.items())},
                     "duplicate_firm_dates": duplicates},
        "checks": {"annual_equals_october_rows": [int(annual_match), len(annual)], "score_bands_by_regime": bands},
        "breaks": {"adjacent_dates": adjacent,
                   "same_grade_by_transition_R1": {k: float(np.mean(v)) for k, v in sorted(season.items())}},
        "regimes": regimes,
        "old_copy_vs_new": old_vs_new,
        "snapshot_check": {"snapshot": SNAPSHOT.relative_to(ROOT).as_posix(), "snapshot_date": "2026-09-24",
                           "panel_date": dates[-1].isoformat(), "firms": len(common),
                           "same_grade": float(np.mean(a == b)), "within_one_notch": float(np.mean(np.abs(a - b) <= 1)),
                           "grade_corr": float(np.corrcoef(a, b)[0, 1]), "score_corr": float(np.corrcoef(sa, sb)[0, 1]),
                           "mean_grade_panel": float(a.mean()), "mean_grade_snapshot": float(b.mean())},
    }
    with open(Path(__file__).parent / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {"kappa_grade": v["kappa_grade"], "kappa_score": v["kappa_score"]} for k, v in regimes.items()}, indent=1))


if __name__ == "__main__":
    main()
