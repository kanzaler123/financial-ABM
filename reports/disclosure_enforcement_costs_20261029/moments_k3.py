"""Stage K3 target moments m1-m9, for the data and for the model (contract 28.4).

    python -I -X utf8 moments_k3.py        # data moments and their joint firm-bootstrap covariance

The same functions compute the model moments from sampled archives (estimate_k3.py). Moments:
  m1-m5  pooled correlations of normal scores k = 1, 2, 4, 8, 12 quarters apart (contract 25.1, data_moments.py);
  m6     share of adjacent-quarter pairs whose rank notch changes;
  m7     share of adjacent-quarter pairs whose rank notch changes by at least 2;
  m8     per date, mean normal score of the large group minus the small group, averaged over dates;
  m9     ln(small / large) of the yearly share of firm-years with a detected case (data: K1's T1 or T2 events).
Rank notch: on every date the firms are ranked by score and cut into 9 notches with the pooled R1 shares of the
official grades (C ... AAA), so data and model have the same notch margins. Size groups: top 10% (large) and bottom
60% (small) of market value on the date (data; model: size_class 0 and 2, fixed at the start).

Data: the Huazheng quarterly panel (R1 = 2009-01-31 to 2021-01-31, all firms, as in contract 25.1), month-end total
market value on the rating month (CNRDS monthly returns), and the K1 panels (non-financial, 2016-2025). The
covariance resamples firms with replacement jointly over the union of firms (1,000 draws, seed 28901); normal
scores, notches and size groups are held fixed, as in data_moments.py. Writes data_moments_k3.json (LF).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H_DIR = ROOT / "reports/disclosure_persistence_estimation_20261026"
SOURCE = ROOT / "data/raw/huazheng_panel_netdisk_20261026/huazheng_quarterly_2009_2025.xlsx"
CNRDS = ROOT / "data/processed/cnrds_20261026"
K1_PANEL = ROOT / "data/processed/stage_k_20261029"
K1_PANELS = {"k1_events.csv.gz": "9e18ade4478d9a7036a9bba2e7056b0c600115157a9ef947da7219e8cb7670a4",
             "k1_firm_year.csv.gz": "502873098264df5cea39d996778f1a817976ccf07f724cdbadd2be50a021d3a9"}
R1 = ("2009-01-31", "2021-01-31")
GRADES = ["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
LAGS = (1, 2, 4, 8, 12)
NAMES = [f"m{i}" for i in range(1, 10)]
BOOTSTRAP, SEED = 1000, 28901
LARGE_SHARE, SMALL_SHARE = 0.10, 0.60
# Model: yearly blocks of 252 trading days from day 1260, eight full blocks (days 1260-3275).
BLOCK_START, BLOCK_DAYS, BLOCKS = 1260, 252, 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dm = _load(H_DIR / "data_moments.py", "h_data_moments")


# -- per-firm sufficient statistics ------------------------------------------------------------------------------

def notch_cuts(shares: np.ndarray) -> np.ndarray:
    return np.cumsum(shares)[:-1]


def rank_notches(S: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Firms x dates scores -> rank notches 1..9 (nan where missing); ties take average ranks."""
    Z = dm.normal_scores(S)                    # monotone in the rank; recover u = (rank - 0.5) / n
    from scipy.special import ndtr
    U = ndtr(Z)
    out = np.full(S.shape, np.nan)
    ok = np.isfinite(U)
    out[ok] = np.searchsorted(cuts, U[ok], side="right") + 1
    return out


def notch_sums(N: np.ndarray) -> np.ndarray:
    """Per firm: adjacent-date pairs, pairs with a change, pairs with a change of 2 or more."""
    a, b = N[:, :-1], N[:, 1:]
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(np.where(ok, b - a, 0.0))
    return np.stack([ok.sum(1), (ok & (d > 0)).sum(1), (ok & (d >= 2)).sum(1)], axis=1).astype(float)


def size_sums(Z: np.ndarray, large: np.ndarray, small: np.ndarray) -> np.ndarray:
    """Per firm and date: z*large, large, z*small, small (firms x dates x 4)."""
    zl = np.where(large & np.isfinite(Z), Z, 0.0)
    zs = np.where(small & np.isfinite(Z), Z, 0.0)
    return np.stack([zl, (large & np.isfinite(Z)).astype(float), zs, (small & np.isfinite(Z)).astype(float)], axis=2)


def gradient(sums: np.ndarray, w: np.ndarray | None = None) -> float:
    s = sums.sum(0) if w is None else np.tensordot(w, sums, axes=(0, 0))
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    return float(np.mean(s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]))


def ratio_sums(cases_small, years_small, cases_large, years_large) -> np.ndarray:
    return np.stack([cases_small, years_small, cases_large, years_large], axis=1).astype(float)


def log_ratio(sums: np.ndarray, w: np.ndarray | None = None) -> float:
    cs, ys, cl, yl = sums.sum(0) if w is None else w @ sums
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log((cs / ys) / (cl / yl)))


class Stats:
    """Everything needed for m1-m9, as per-firm arrays aligned on one firm index."""

    def __init__(self, Z, N, large, small, ratio):
        self.pairs = dm.pair_sums(Z, LAGS)
        self.notch = notch_sums(N)
        self.size = size_sums(Z, large, small)
        self.ratio = ratio

    def moments(self, w: np.ndarray | None = None) -> np.ndarray:
        auto = [dm.pooled(self.pairs[k], w) for k in LAGS]
        n = self.notch.sum(0) if w is None else w @ self.notch
        return np.array(auto + [n[1] / n[0], n[2] / n[0], gradient(self.size, w), log_ratio(self.ratio, w)])


# -- model side -------------------------------------------------------------------------------------------------

def model_stats(signal: np.ndarray, size_class: np.ndarray, detections: np.ndarray, cuts: np.ndarray,
                period_days: int = 60) -> Stats:
    """One simulated market. signal: sampled dates x firms (noise already added); detections: periods x firms."""
    S = signal.T
    Z = dm.normal_scores(S)
    N = rank_notches(S, cuts)
    large = np.repeat((size_class == 0)[:, None], S.shape[1], axis=1)
    small = np.repeat((size_class == 2)[:, None], S.shape[1], axis=1)
    days = np.arange(detections.shape[0]) * period_days
    block = (days - BLOCK_START) // BLOCK_DAYS
    hit = np.zeros((BLOCKS, detections.shape[1]), dtype=bool)
    for b in range(BLOCKS):
        rows = (days >= BLOCK_START) & (block == b)
        hit[b] = (detections[rows] > 0).any(axis=0)
    cases = hit.sum(0)
    ratio = ratio_sums(np.where(size_class == 2, cases, 0), np.where(size_class == 2, BLOCKS, 0),
                       np.where(size_class == 0, cases, 0), np.where(size_class == 0, BLOCKS, 0))
    return Stats(Z, N, large, small, ratio)


def yearly_detection_rate(detections: np.ndarray, period_days: int = 60) -> float:
    """Share of firm-years (the eight blocks) with at least one detected case: the model side of K1 (28.6 (e))."""
    days = np.arange(detections.shape[0]) * period_days
    block = (days - BLOCK_START) // BLOCK_DAYS
    hits = [(detections[(days >= BLOCK_START) & (block == b)] > 0).any(axis=0) for b in range(BLOCKS)]
    return float(np.mean(hits))


def pooled_model_moments(stats: list[Stats]) -> np.ndarray:
    """Seed-mean of m1-m8; m9 from the case counts pooled over seeds (contract 28.4)."""
    per_seed = np.array([s.moments()[:8] for s in stats])
    m9 = log_ratio(np.concatenate([s.ratio for s in stats]))
    return np.append(per_seed.mean(0), m9)


def seed_bootstrap_cov(stats: list[Stats], draws: int = 1000, seed: int = 28906) -> np.ndarray:
    """Covariance of the pooled model moments under resampling of seeds (Omega_sim / S)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(stats), len(stats))
        out.append(pooled_model_moments([stats[i] for i in pick]))
    out = np.array(out)
    return np.cov(out[np.isfinite(out).all(1)], rowvar=False)


# -- data side --------------------------------------------------------------------------------------------------

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_stats() -> tuple[Stats, dict]:
    import openpyxl
    import pandas as pd

    rows = list(openpyxl.load_workbook(SOURCE, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    assert rows[0][:5] == ("证券代码", "季度日期", "证券简称", "综合评级", "综合得分"), rows[0]
    h = pd.DataFrame([(f"{r[0]:06d}", r[1].date().isoformat(), r[3], float(r[4])) for r in rows[1:]],
                     columns=["Scode", "date", "grade", "score"])
    h = h[(h.date >= R1[0]) & (h.date <= R1[1])]
    # AAA first appears in 2024-04 (contract 27.1), so its R1 share is 0 and R1 has eight notches in effect.
    shares = h.grade.value_counts(normalize=True).reindex(GRADES).fillna(0.0).to_numpy()
    assert h.grade.isin(GRADES).all()
    cuts = notch_cuts(shares)

    k1 = {name: sha(K1_PANEL / name) for name in K1_PANELS}
    assert k1 == K1_PANELS, "K1 panels differ from the recorded ones"
    ev = pd.read_csv(K1_PANEL / "k1_events.csv.gz", dtype={"Scode": str})
    fy = pd.read_csv(K1_PANEL / "k1_firm_year.csv.gz", dtype={"Scode": str})
    fy = fy[~fy.financial & fy.year.between(2016, 2025) & (fy.mcap_prev_dec > 0)].copy()
    r = fy.groupby("year").mcap_prev_dec.rank(ascending=False, method="first") / fy.groupby("year").Scode.transform("size")
    fy["group"] = np.where(r <= LARGE_SHARE, "large", np.where(r <= 1 - SMALL_SHARE, "mid", "small"))
    hit = ev[ev.tier.isin(["T1", "T2"])].groupby(["Scode", "year"]).size().rename("n").reset_index()
    fy = fy.merge(hit, on=["Scode", "year"], how="left")
    fy["case"] = fy.n.fillna(0) > 0

    firms = sorted(set(h.Scode) | set(fy.Scode))
    fi = {c: i for i, c in enumerate(firms)}
    dates = sorted(h.date.unique())
    di = {d: j for j, d in enumerate(dates)}
    S = np.full((len(firms), len(dates)), np.nan)
    S[h.Scode.map(fi).to_numpy(), h.date.map(di).to_numpy()] = h.score.to_numpy()

    mr = pd.read_csv(CNRDS / "monthly_returns.csv.gz", dtype=str, usecols=["Scode", "Trdmt", "Mmkecap"])
    mr = mr[(mr.Trdmt != "2017-03-10") & mr.Scode.isin(fi)]
    mr["date"] = mr.Trdmt.str[:7]
    month_of = {d: d[:7] for d in dates}
    mr = mr[mr.date.isin(set(month_of.values()))]
    M = np.full(S.shape, np.nan)
    col = {month_of[d]: di[d] for d in dates}
    M[mr.Scode.map(fi).to_numpy(), mr.date.map(col).to_numpy()] = pd.to_numeric(mr.Mmkecap, errors="coerce").to_numpy()
    large = np.zeros(S.shape, dtype=bool)
    small = np.zeros(S.shape, dtype=bool)
    for j in range(S.shape[1]):
        ok = np.isfinite(S[:, j]) & np.isfinite(M[:, j]) & (M[:, j] > 0)
        rank = np.empty(ok.sum())
        rank[np.argsort(-M[ok, j], kind="stable")] = np.arange(1, ok.sum() + 1)
        u = rank / ok.sum()
        idx = np.flatnonzero(ok)
        large[idx[u <= LARGE_SHARE], j] = True
        small[idx[u > 1 - SMALL_SHARE], j] = True

    Z = dm.normal_scores(S)
    N = rank_notches(S, cuts)
    g = fy.assign(i=fy.Scode.map(fi)).groupby(["i", "group"]).case.agg(["sum", "size"]).unstack("group").fillna(0)
    g = g.reindex(range(len(firms)), fill_value=0)
    ratio = ratio_sums(g[("sum", "small")], g[("size", "small")], g[("sum", "large")], g[("size", "large")])
    info = {"huazheng_sha256": sha(SOURCE), "k1_panels": k1, "monthly_returns_sha256": sha(CNRDS / "monthly_returns.csv.gz"),
            "r1": list(R1), "dates": len(dates), "firms_union": len(firms),
            "firms_rated_r1": int(np.isfinite(S).any(1).sum()), "firms_k1": int(fy.Scode.nunique()),
            "notch_shares_r1_C_to_AAA": shares.tolist(), "notch_cuts": cuts.tolist(),
            "rated_firm_dates_without_market_value": int((np.isfinite(S) & ~np.isfinite(M)).sum()),
            "k1_firm_years": {grp: int((fy.group == grp).sum()) for grp in ("large", "mid", "small")}}
    return Stats(Z, N, large, small, ratio), info


def main() -> None:
    stats, info = data_stats()
    value = stats.moments()
    n = len(stats.notch)
    rng = np.random.default_rng(SEED)
    draws = np.array([stats.moments(rng.multinomial(n, np.full(n, 1 / n)).astype(float)) for _ in range(BOOTSTRAP)])
    out = {"contract": "28.4", "names": NAMES, "lags": list(LAGS), "moments": value.tolist(),
           "bootstrap": {"replications": BOOTSTRAP, "seed": SEED, "unit": "firm (union of the rating panel and K1)"},
           "bootstrap_se": draws.std(0, ddof=1).tolist(), "bootstrap_cov": np.cov(draws, rowvar=False).tolist(),
           "inputs": info}
    h = json.loads((H_DIR / "data_moments.json").read_text(encoding="utf-8"))["targets"]["R1"]
    out["check_against_contract_25_1"] = {"moments_m1_m5": h["moments"], "bootstrap_se_m1_m5": h["bootstrap_se"],
                                          "max_abs_moment_difference": float(np.max(np.abs(np.array(h["moments"]) - value[:5])))}
    (HERE / "data_moments_k3.json").write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for name, m, s in zip(NAMES, value, out["bootstrap_se"]):
        print(f"{name} {m:.4f} ({s:.4f})")
    print(json.dumps(out["check_against_contract_25_1"]))


if __name__ == "__main__":
    main()
