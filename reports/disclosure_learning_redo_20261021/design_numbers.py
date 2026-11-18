"""Design numbers for the learning-firm redo (research contract section 23, stage D.3).

Read-only. Uses the local D.2 archive (reports/disclosure_rl_comparison_20261013/runs/v2, not in git) and the
archived run times of the targeted-audit and assurance experiments. Writes design_numbers.json next to this file.

1. Run time per contract-scale world: alone, 4 in parallel (D.2), 16 in parallel (E.1, E.2).
2. State-feature distributions of the D.2 learner and of the proposed improvement-stock feature.
3. Coverage of the D.2 tables: the decision-weighted gate proposed for section 23 versus the old gate.
4. Table-level outcomes in D.2: spread across validation seeds within a table, and between two tables.
5. Seed lineage: the new seed blocks against every seed-like field in the repository's JSON files. A first
   choice (pre-policy training 44001-44996) hit 44058 and 44800 from older stages and was replaced by 47001-47996.
6. Budget for the chosen design (K = 10 replicates, both state versions in B, I, F, IF).

Usage: python design_numbers.py
"""

from __future__ import annotations

import glob
import json
import re
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D2 = ROOT / "reports" / "disclosure_rl_comparison_20261013" / "runs" / "v2"
EXPERIMENTS = ("baseline", "B", "I", "F", "IF")
IMPROVE = 0
MAX_CHANGE = 0.25  # domain.max_change_per_period, SD
REPLICATES, VALIDATION, CHECK = 10, 5, 3
PRE_WORLDS, POST_WORLDS, SCENARIOS, VERSIONS = 96, 64, 4, 2


def elapsed(pattern: str) -> dict:
    values = [json.loads(Path(path).read_text(encoding="utf-8"))["elapsed_seconds"] for path in glob.glob(str(pattern))]
    return {"runs": len(values), "median_seconds": round(statistics.median(values), 1),
            "min_seconds": round(min(values), 1), "max_seconds": round(max(values), 1)}


def timing() -> dict:
    out = {
        "d2_alone_baseline_training": elapsed(D2 / "baseline" / "training" / "*" / "manifest.json"),
        "d2_four_parallel_B_training": elapsed(D2 / "B" / "training" / "*" / "manifest.json"),
        "e1_targeted_audits_16_workers": elapsed(ROOT / "reports/disclosure_targeted_audits_20261020/*/*/*/attempt_001/manifest.json"),
        "e2_assurance_16_workers": elapsed(ROOT / "reports/disclosure_assurance_20261021/assurance_v1/*/*/attempt_001/manifest.json"),
    }
    # E.1 confirmatory wall clock (its README): 450 runs, 11:16-13:12 with 16 workers.
    out["e1_wall_clock_runs_per_minute"] = round(450 / 116, 2)
    out["sixteen_worker_runs_per_minute"] = round(16 / out["e2_assurance_16_workers"]["median_seconds"] * 60, 2)
    return out


def decode(states: np.ndarray) -> dict[str, np.ndarray]:
    return {"signal_tercile": states // 27, "inflation": states // 9 % 3, "cost_tier": states // 3 % 3, "stage": states % 3}


def improvement_stock(result) -> np.ndarray:
    """Committed real improvement (SD) before each decision period, including projects still in flight."""
    committed = np.where(result["actions"] == IMPROVE, result["amounts"], 0.0)
    return np.vstack([np.zeros((1, committed.shape[1])), np.cumsum(committed, axis=0)[:-1]])


def features() -> dict:
    out = {}
    patterns = {name: D2 / name / "validation" / "*" / "q_learning" / "result.npz" for name in EXPERIMENTS}
    patterns["baseline_training_first5"] = D2 / "baseline" / "training" / "seed_3120[1-5]" / "result.npz"
    patterns["baseline_training_last7"] = D2 / "baseline" / "training" / "seed_3129[0-6]" / "result.npz"
    for name, pattern in patterns.items():
        bins, stocks = {key: [] for key in ("signal_tercile", "inflation", "cost_tier", "stage")}, []
        for path in sorted(glob.glob(str(pattern))):
            with np.load(path) as result:
                window = slice(21, None)  # decisions after day 1200 (period 20 is day 1200)
                groups, states = result["learning_groups"][window], result["learning_states"][window]
                keep = groups >= 3 if name in ("B", "I", "F", "IF") else np.ones_like(groups, dtype=bool)
                for key, value in decode(states[keep]).items():
                    bins[key].append(value)
                stocks.append(improvement_stock(result)[window][keep])
        stock = np.concatenate(stocks)
        out[name] = {
            "decisions": int(stock.size),
            "feature_bin_shares": {key: np.round(np.bincount(np.concatenate(value), minlength=3) / stock.size, 4).tolist()
                                   for key, value in bins.items()},
            "improvement_stock_bin_shares": np.round(np.bincount(np.digitize(stock, (1e-12, MAX_CHANGE)), minlength=3) / stock.size, 4).tolist(),
            "improvement_stock_median_sd": round(float(np.median(stock)), 4),
            "improvement_stock_p90_sd": round(float(np.quantile(stock, .9)), 4),
        }
    return out


def coverage() -> dict:
    """Share of decisions in states where some legal option has fewer than 20 training updates."""
    comparison = json.loads((ROOT / "reports/disclosure_rl_comparison_20261013/comparison.json").read_text(encoding="utf-8"))
    out = {}
    for name in EXPERIMENTS:
        with np.load(D2 / name / "frozen_model.npz") as model:
            counts = model["counts"]
        pooled, worst = {}, {}
        unseen_pooled, unseen_worst = {}, {}
        for path in sorted(glob.glob(str(D2 / name / "validation" / "*" / "q_learning" / "result.npz"))):
            with np.load(path) as result:
                groups = result["learning_groups"][1:].ravel()
                states = result["learning_states"][1:].ravel()
                legal = result["valid_actions"][1:].reshape(groups.size, -1)
            under = ((counts[groups, states] < 20) & legal).any(axis=1)
            unseen = counts[groups, states].sum(axis=1) == 0
            for group in np.unique(groups):
                member = groups == group
                key = int(group)
                pooled.setdefault(key, [0, 0])
                pooled[key][0] += int(under[member].sum())
                pooled[key][1] += int(member.sum())
                worst[key] = max(worst.get(key, 0.0), float(under[member].mean()))
                unseen_pooled.setdefault(key, 0)
                unseen_pooled[key] += int(unseen[member].sum())
                unseen_worst[key] = max(unseen_worst.get(key, 0.0), float(unseen[member].mean()))
        pooled_share = {key: round(a / b, 4) for key, (a, b) in pooled.items()}
        out[name] = {
            "undercovered_decision_share_pooled": pooled_share,
            "undercovered_decision_share_worst_world": {key: round(value, 4) for key, value in worst.items()},
            "unseen_state_share_pooled": {key: round(unseen_pooled[key] / pooled[key][1], 4) for key in pooled},
            "unseen_state_share_worst_world": {key: round(value, 4) for key, value in unseen_worst.items()},
            "proposed_gate": "PASS" if max(pooled_share.values()) <= .05 and max(worst.values()) <= .10
            and max(unseen_worst.values()) <= .05 else "FAIL",
            "old_gate": comparison["coverage"][name]["status"] if name in comparison.get("coverage", {}) else None,
        }
    return out


def outcomes() -> dict:
    out = {}
    for name in EXPERIMENTS:
        for rule in ("q_learning", "npv_discrete", "npv"):
            rows = [json.loads(Path(path).read_text(encoding="utf-8"))
                    for path in sorted(glob.glob(str(D2 / name / "validation" / "*" / rule / "metrics.json")))]
            if not rows:
                continue
            true = [row["true_change"]["equal_weight_sd"] for row in rows]
            wash = [row["actions"]["wash"] / row["actions"]["decisions"] for row in rows]
            out[f"{name}/{rule}"] = {"seeds": len(rows),
                                     "true_change_mean": round(float(np.mean(true)), 4),
                                     "true_change_seed_sd": round(float(np.std(true, ddof=1)), 4),
                                     "wash_share_mean": round(float(np.mean(wash)), 4),
                                     "wash_share_seed_sd": round(float(np.std(wash, ddof=1)), 4)}
    base, scenario = out["baseline/q_learning"], out["B/q_learning"]
    out["two_tables_same_regime"] = {
        "note": "Pre-policy common table (check seeds, constant baseline) vs scenario B post-policy table (validation seeds).",
        "true_change_difference": round(base["true_change_mean"] - scenario["true_change_mean"], 4),
        "wash_share_difference": round(base["wash_share_mean"] - scenario["wash_share_mean"], 4),
    }
    return out


SEED_KEY = re.compile(r"seed", re.IGNORECASE)


def _collect(node, found: set[int], under_seed_key: bool = False) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _collect(value, found, under_seed_key or bool(SEED_KEY.search(str(key))))
    elif isinstance(node, list):
        for value in node:
            _collect(value, found, under_seed_key)
    elif under_seed_key and isinstance(node, int) and not isinstance(node, bool):
        found.add(node)


def seed_blocks() -> dict[str, list[int]]:
    blocks = {
        "confirmatory_prepolicy_training": [47000 + 100 * r + i for r in range(REPLICATES) for i in range(1, PRE_WORLDS + 1)],
        "confirmatory_postpolicy_training": [45000 + 100 * r + i for r in range(REPLICATES) for i in range(1, POST_WORLDS + 1)],
        "confirmatory_prepolicy_check": list(range(46001, 46001 + CHECK)),
        "confirmatory_validation": list(range(46101, 46101 + VALIDATION)),
        "development_prepolicy_training": [37001, 37002, 37003, 37101, 37102, 37103],
        "development_postpolicy_training": [37201, 37202, 37301, 37302],
        "development_prepolicy_check": [37401],
        "development_validation": [37501, 37502],
    }
    return blocks


def lineage() -> dict:
    files, found = 0, set()
    for folder in ("configs", "reports", "runs", "docs", "data"):
        for path in (ROOT / folder).rglob("*.json"):
            if HERE in path.parents:
                continue
            files += 1
            try:
                _collect(json.loads(path.read_text(encoding="utf-8")), found)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
    blocks = seed_blocks()
    every = [seed for values in blocks.values() for seed in values]
    return {"json_files": files, "distinct_seed_values": len(found),
            "overlap": sorted(found & set(every)), "unique_within_design": len(every) == len(set(every)),
            "blocks": {name: [min(values), max(values), len(values)] for name, values in blocks.items()}}


def budget(time: dict) -> dict:
    per_version = PRE_WORLDS + CHECK + SCENARIOS * (POST_WORLDS + VALIDATION)
    learning = REPLICATES * VERSIONS * per_version
    npv = SCENARIOS * VALIDATION * 2
    replay = REPLICATES * VERSIONS * SCENARIOS * VALIDATION
    development = 2 * VERSIONS * (3 + 1 + SCENARIOS * (2 + 2)) + SCENARIOS * 2 * 2 + 2 * VERSIONS * SCENARIOS * 2
    total = learning + npv + replay + development
    low, high = time["e1_wall_clock_runs_per_minute"], time["sixteen_worker_runs_per_minute"]
    return {"worlds_per_replicate_and_version": per_version, "learning_worlds": learning, "npv_worlds": npv,
            "replay_worlds": replay, "development_worlds": development, "total_worlds": total,
            "hours_at_16_workers": [round(total / high / 60, 1), round(total / low / 60, 1)],
            "longest_chain_worlds": PRE_WORLDS + CHECK + POST_WORLDS + VALIDATION,
            "compact_archive_gb": round(total * 1.0 / 1024, 1), "full_archive_gb": round(total * 28 / 1024, 1)}


def main() -> None:
    time = timing()
    payload = {"timing": time, "features": features(), "coverage": coverage(), "outcomes": outcomes(),
               "lineage": lineage(), "budget": budget(time)}
    (HERE / "design_numbers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    print(json.dumps({key: payload[key] for key in ("timing", "budget")}, indent=2))
    print(json.dumps(payload["lineage"], indent=2))


if __name__ == "__main__":
    main()
