"""Target moments for estimating the persistence of ESG ratings (research contract section 25).

Input: the netdisk Huazheng quarterly panel in data/raw/huazheng_panel_netdisk_20261026/ (not committed) and the
CSI 300 constituent list of 2026-10-09. Output: data_moments.json (moments and their firm-cluster bootstrap
covariance), committed before any estimation run.

On every observation date the composite scores are replaced by normal scores, Phi^-1((rank - 0.5) / n) with average
ranks for ties, so the data and the model's public signal are compared on the same latent scale whatever the score
distribution. The moments are the pooled correlations between a firm's normal scores k observation dates (quarters)
apart, k in LAGS, using every firm and pair of dates inside one rating method (see
reports/huazheng_panel_dynamics_20261026). The covariance resamples firms with replacement; the normal scores are
held fixed.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
from scipy.special import ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/huazheng_panel_netdisk_20261026/huazheng_quarterly_2009_2025.xlsx"
CONSTITUENTS = ROOT / "data/processed/disclosure_public_20261011/csi300_constituents_20261009.csv"
REGIMES = {"R1": ("2009-01-31", "2021-01-31"), "R2": ("2021-04-30", "2024-01-31")}
LAGS = (1, 2, 4, 8, 12)
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 25026


def normal_scores(S):
    Z = np.full(S.shape, np.nan)
    for j in range(S.shape[1]):
        ok = np.isfinite(S[:, j])
        if ok.sum() > 1:
            Z[ok, j] = ndtri((rankdata(S[ok, j]) - 0.5) / ok.sum())
    return Z


def pair_sums(Z, lags):
    """Per firm and lag: number of pairs and the sums needed for a pooled correlation."""
    out = {}
    for k in lags:
        a, b = Z[:, :-k], Z[:, k:]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        out[k] = np.stack([ok.sum(1), a.sum(1), b.sum(1), (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)], axis=1)
    return out


def pooled(sums, weights=None):
    n, sa, sb, saa, sbb, sab = (sums if weights is None else weights[:, None] * sums).sum(0)
    cov = sab / n - sa * sb / n**2
    return cov / np.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2))


def moments(Z, label):
    lags = [k for k in LAGS if k < Z.shape[1]]
    sums = pair_sums(Z, lags)
    value = np.array([pooled(sums[k]) for k in lags])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP, len(lags)))
    for b in range(BOOTSTRAP):
        weights = rng.multinomial(Z.shape[0], np.full(Z.shape[0], 1 / Z.shape[0])).astype(float)
        draws[b] = [pooled(sums[k], weights) for k in lags]
    return {"label": label, "lags": lags, "moments": value.tolist(), "pairs": [int(sums[k][:, 0].sum()) for k in lags],
            "firms": int(np.isfinite(Z).any(1).sum()), "dates": Z.shape[1],
            "bootstrap_se": draws.std(0, ddof=1).tolist(), "bootstrap_cov": np.cov(draws, rowvar=False).tolist()}


def main():
    rows = list(openpyxl.load_workbook(SOURCE, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))[1:]
    firms = sorted({r[0] for r in rows})
    dates = sorted({r[1].date() for r in rows})
    fi, di = {c: i for i, c in enumerate(firms)}, {d: j for j, d in enumerate(dates)}
    S = np.full((len(firms), len(dates)), np.nan)
    for r in rows:
        S[fi[r[0]], di[r[1].date()]] = r[4]
    with open(CONSTITUENTS, encoding="utf-8-sig") as f:
        csi300 = {int(r["成份券代码Constituent Code"]) for r in csv.DictReader(f)}
    in_csi300 = np.array([c in csi300 for c in firms])
    out = {"source": {"file": SOURCE.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "constituents": CONSTITUENTS.relative_to(ROOT).as_posix(),
                      "constituents_sha256": hashlib.sha256(CONSTITUENTS.read_bytes()).hexdigest()},
           "bootstrap": {"replications": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "unit": "firm"}, "targets": {}}
    for name, (first, last) in REGIMES.items():
        cols = list(range(di[dt.date.fromisoformat(first)], di[dt.date.fromisoformat(last)] + 1))
        Z = normal_scores(S[:, cols])
        out["targets"][name] = moments(Z, f"all A-share firms, {first} to {last}")
        # Current CSI 300 members, normal scores still ranked among all firms on each date.
        out["targets"][f"{name}_csi300_current"] = moments(Z[in_csi300], f"2026-10-09 CSI 300 members, {first} to {last}")
    with open(Path(__file__).parent / "data_moments.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    for name, t in out["targets"].items():
        print(name, t["firms"], "firms", " ".join(f"lag{k} {m:.4f}({s:.4f})" for k, m, s in zip(t["lags"], t["moments"], t["bootstrap_se"])))


if __name__ == "__main__":
    main()
