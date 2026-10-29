"""Diagnostic (contract 26.25): how the sampled public signal is spread across firms, with and without the rating
channel. Read-only on finished sampled archives; descriptive, no hypothesis test.

  python signal_bunching.py

For each archive and scenario (seeds as listed): quantiles of the public signal across firms on the last sampled
date (day 3465), the mean over dates of the largest group of firms whose sorted signals are all within 0.005 of
their neighbour ("bunch"), and the H2 target moments at measurement error sigma_m = 0, 0.0078 and 0.05 (H2's
seed_moments with H2's noise draws). Writes signal_bunching.json and .md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimate_v3 as e  # noqa: E402

H2 = e.H2_DIR
CASES = (  # label, archive, scenario, seeds, lambda (bp), dynamics
    ("H2 q_hat", H2 / "final_v1", "q_hat", range(48001, 48011), 0.0, "Q(0.505, 0.423)"),
    ("F0 q_k0p505_s0p423", HERE / "feasibility_v3", "q_k0p505_s0p423", range(58901, 58906), 25.75, "Q(0.505, 0.423)"),
    ("F0 u_d0", HERE / "feasibility_v3", "u_d0", range(58901, 58906), 25.75, "U(0)"),
    ("v3 final u_hat", HERE / "final_v3", "u_hat", range(58001, 58011), 25.75, "U(0.151)"),
)
SIGMAS = (0.0, 0.0078, 0.05)


def bunch(dates: np.ndarray) -> float:
    sizes = []
    for row in dates:
        gaps = np.diff(np.sort(row))
        best = current = 1
        for gap in gaps:
            current = current + 1 if gap < 0.005 else 1
            best = max(best, current)
        sizes.append(best)
    return float(np.mean(sizes))


def main() -> None:
    rows = {}
    for label, archive, scenario, seeds, slope, dynamics in CASES:
        signals = {seed: e.h2.finished_signal(archive, scenario, seed) for seed in seeds}
        last = np.concatenate([s[-1] for s in signals.values()])
        moments = {str(sigma): np.mean([e.h2.seed_moments(s, sigma, e.h2.noise(seed, s.shape))
                                        for seed, s in signals.items()], axis=0).round(4).tolist() for sigma in SIGMAS}
        rows[label] = {"archive": archive.relative_to(HERE.parents[1]).as_posix(), "scenario": scenario,
                       "seeds": [min(seeds), max(seeds)], "lambda_bp": slope, "dynamics": dynamics,
                       "last_date_quantiles": dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                                                       np.quantile(last, [0, .1, .25, .5, .75, .9, 1]).round(3).tolist())),
                       "last_date_sd": round(float(np.mean([s[-1].std() for s in signals.values()])), 3),
                       "mean_largest_bunch_of_200": round(float(np.mean([bunch(s) for s in signals.values()])), 1),
                       "moments_by_sigma_m": moments}
    (HERE / "signal_bunching.json").write_bytes((json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    lines = ["# 公开信号在截面上的分布（第 26.25 条的诊断，描述）", "",
             "| 档案 | λ（bp） | 自然动态 | 最后一期 p10 / p50 / p90 / 最大 | 截面 SD | 最大聚集组（200 家中） | 滞后 1 矩：σ_m = 0 / 0.0078 / 0.05 |",
             "|---|---:|---|---|---:|---:|---|"]
    for label, row in rows.items():
        q = row["last_date_quantiles"]
        m = row["moments_by_sigma_m"]
        lines.append(f"| {label} | {row['lambda_bp']:g} | {row['dynamics']} | {q['p10']} / {q['p50']} / {q['p90']} / {q['max']} | "
                     f"{row['last_date_sd']} | {row['mean_largest_bunch_of_200']} | "
                     + " / ".join(f"{m[str(s)][0]:.4f}" for s in SIGMAS) + " |")
    (HERE / "signal_bunching.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
