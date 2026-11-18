"""Self-test of the stage J estimators (contract 27.9) on synthetic panels; no data are read.

    python -I -X utf8 selftest.py [--reps 500]

1. Exactness: the pooled estimator (one fixed-effect dimension absorbed by demeaning, CR1 errors clustered one or two
   ways) against an explicit dummy-variable regression with the same CR1 formula.
2. Coverage: synthetic event panels shaped like the point-in-time sample (12 dates, 30 industries, 4,700 firms per
   date) with a date-level common shock loading on a characteristic, a firm effect and a known slope; the share of
   95% intervals that cover the slope, for the pooled two-way estimator and Fama-MacBeth.
Writes selftest.json (LF). Seed 27902.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # -I leaves the script directory off sys.path
import estimate as est  # noqa: E402

SEED = 27902


def explicit(df: pd.DataFrame, y: str, xs: list[str], fe: str, clusters: list[str]) -> tuple[float, float]:
    D = pd.get_dummies(df[fe], drop_first=False).to_numpy(float)
    X = np.column_stack([df[xs].to_numpy(float), D])
    yv = df[y].to_numpy(float)
    n, k = X.shape
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ X.T @ yv
    u = yv - X @ b
    rank = np.linalg.matrix_rank(X)

    def v(codes):
        g = pd.factorize(codes)[0]
        s = np.zeros((g.max() + 1, k))
        np.add.at(s, g, X * u[:, None])
        G = g.max() + 1
        return G / (G - 1) * (n - 1) / (n - rank) * inv @ (s.T @ s) @ inv

    if len(clusters) == 1:
        V = v(df[clusters[0]].to_numpy())
    else:
        inter = df[clusters[0]].astype(str) + "|" + df[clusters[1]].astype(str)
        V = v(df[clusters[0]].to_numpy()) + v(df[clusters[1]].to_numpy()) - v(inter.to_numpy())
    return float(b[0]), float(np.sqrt(V[0, 0]))


def synthetic(rng: np.random.Generator, dates: int, firms: int, industries: int, beta: float) -> pd.DataFrame:
    firm_ind = rng.integers(0, industries, firms)
    firm_eff = rng.normal(0, 3, firms)
    rows = []
    for d in range(dates):
        size = rng.normal(0, 1, firms)
        x = np.round(rng.normal(0, 0.6, firms) + 0.2 * size)          # notch-like, correlated with size
        common = rng.normal(0, 4)                                      # date-level shock loading on size
        ind_shock = rng.normal(0, 3, industries)[firm_ind]
        e = rng.standard_t(5, firms) * 10
        y = beta * x + common * size + 1.0 * size + ind_shock + firm_eff + e
        rows.append(pd.DataFrame({"Scode": np.arange(firms).astype(str), "date": f"d{d:02d}",
                                  "industry": firm_ind.astype(str), "y": y, "x": x, "size": size}))
    df = pd.concat(rows, ignore_index=True)
    df["cell"] = df.date + "|" + df.industry
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED}

    small = synthetic(rng, dates=6, firms=300, industries=8, beta=0.5)
    checks = {}
    for name, cl in (("firm", ["Scode"]), ("two_way", ["Scode", "date"])):
        p = est.pooled(small, "y", "x", ["size"], small.cell, cl)
        b, se = explicit(small[small.cell.map(small.cell.value_counts()) > 1], "y", ["x", "size"], "cell", cl)
        checks[name] = {"coef_diff": abs(p["coef"] - b), "se_rel_diff": abs(p["se"] / se - 1)}
    out["exactness_vs_dummies"] = checks

    beta, cover = 0.4, {"pooled_two_way": [], "fama_macbeth": []}
    for _ in range(args.reps):
        df = synthetic(rng, dates=12, firms=4700, industries=30, beta=beta)
        p = est.pooled(df, "y", "x", ["size"], df.cell, ["Scode", "date"])
        h = stats.t.ppf(0.975, p["df"]) * p["se"]
        cover["pooled_two_way"].append(abs(p["coef"] - beta) <= h)
        fm = est.fama_macbeth(df, "y", "x", ["size"], "date", "industry")
        h = stats.t.ppf(0.975, fm["df"]) * fm["se"]
        cover["fama_macbeth"].append(abs(fm["mean"] - beta) <= h)
    out["coverage_95"] = {k: float(np.mean(v)) for k, v in cover.items()}
    out["coverage_reps"] = args.reps
    out["design"] = "12 dates x 4,700 firms x 30 industries; x = round(N(0, 0.6) + 0.2 size); y = 0.4 x + date shock x size + size + industry-date shock + firm effect + 10 t(5)"
    text = json.dumps(out, indent=2)
    (HERE / "selftest.json").write_bytes((text + "\n").encode("utf-8"))
    print(text)


if __name__ == "__main__":
    main()
