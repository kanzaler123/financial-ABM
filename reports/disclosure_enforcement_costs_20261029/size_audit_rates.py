"""Contract 28.37, before the freeze: each group's audit rate under the initial_value allocation, from the actual
initial values of seeds 70001-70030.

    .venv/Scripts/python.exe -X utf8 reports/disclosure_enforcement_costs_20261029/size_audit_rates.py

For every seed the final-stage theta_hat market is built exactly as a run builds it (experiments._inputs, then
DisclosureMarket), but never run: only the day-0 draws are used. For each beta of the scan the rates come from
policy.audit_probabilities with the regime of the scan's scenarios. Reports, per beta, the mean rate per period of
the large, mid and small classes, the small/large and mid/large multipliers (mean, min and max over seeds), the
largest single rate, capped firms and the budget. Writes size_audit_rates.json (LF).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _no_wmi import prefill_uname  # noqa: E402

prefill_uname()

from abm.disclosure.experiments import _inputs, freeze_spec  # noqa: E402
from abm.disclosure.game import DisclosureMarket  # noqa: E402
from abm.disclosure.policy import audit_probabilities  # noqa: E402

SEEDS = list(range(70001, 70031))
BETAS = (-0.25, -0.5, -0.75, -1.0, -1.25, -1.5)
PERIODS_PER_YEAR = 252 / 60
GROUPS = {"large": 0, "mid": 1, "small": 2}


def yearly(q: float) -> float:
    return 1 - (1 - q) ** PERIODS_PER_YEAR


def main() -> None:
    frozen = freeze_spec(ROOT / "configs/disclosure_stage_k3_final_v1.json")
    (scenario,) = frozen["scenarios"]
    draws = []
    for seed in SEEDS:
        config, domain, policy = _inputs(scenario, seed)
        market = DisclosureMarket(config, domain, policy)
        draws.append((market.initial_intrinsic_value.copy(), market.firms.size_class.copy(), market.prices.copy(),
                      policy.baseline))
    rows = []
    for beta in BETAS:
        per_seed = []
        for initial, size_class, prices, baseline in draws:
            regime = replace(baseline, audit_allocation="initial_value", audit_value_exponent=beta)
            q = audit_probabilities(regime, prices, initial_values=initial)
            means = {g: float(q[size_class == c].mean()) for g, c in GROUPS.items()}
            per_seed.append({**means, "small_over_large": means["small"] / means["large"],
                             "mid_over_large": means["mid"] / means["large"], "max": float(q.max()),
                             "min": float(q.min()), "capped": int((q >= 1.0).sum()), "budget": float(q.sum())})
        col = {k: np.array([s[k] for s in per_seed]) for k in per_seed[0]}
        rows.append({
            "beta": beta,
            "mean_rate_per_period": {g: float(col[g].mean()) for g in GROUPS},
            "yearly_audit_probability_of_mean_rate": {g: yearly(float(col[g].mean())) for g in GROUPS},
            "small_over_large": {"mean": float(col["small_over_large"].mean()),
                                 "min": float(col["small_over_large"].min()),
                                 "max": float(col["small_over_large"].max())},
            "mid_over_large": {"mean": float(col["mid_over_large"].mean()), "min": float(col["mid_over_large"].min()),
                               "max": float(col["mid_over_large"].max())},
            "largest_rate": float(col["max"].max()), "smallest_rate": float(col["min"].min()),
            "capped_firms": int(col["capped"].sum()),
            "budget_range": [float(col["budget"].min()), float(col["budget"].max())],
            "per_seed": per_seed})
    sizes = {g: sorted({int((d[1] == c).sum()) for d in draws}) for g, c in GROUPS.items()}
    out = {"contract": "28.37", "seeds": SEEDS, "spec_theta_hat": "configs/disclosure_stage_k3_final_v1.json",
           "group_sizes": sizes, "q_bar": draws[0][3].audit_probability, "rows": rows}
    (HERE / "size_audit_rates.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    for r in rows:
        m, s = r["mean_rate_per_period"], r["small_over_large"]
        print(f"beta {r['beta']:5.2f}  small/large {s['mean']:6.2f} [{s['min']:6.2f}, {s['max']:6.2f}]  "
              f"mid/large {r['mid_over_large']['mean']:5.2f}  q small {m['small']:.4f} mid {m['mid']:.4f} "
              f"large {m['large']:.5f}  yearly large {r['yearly_audit_probability_of_mean_rate']['large']:.4f}  "
              f"max {r['largest_rate']:.3f}  capped {r['capped_firms']}  budget {r['budget_range']}")
    print("group sizes", sizes)


if __name__ == "__main__":
    main()
