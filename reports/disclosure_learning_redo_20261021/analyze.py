"""Analysis of the learning-firm redo (contract section 23), committed before the confirmatory run.

  python analyze.py --spec configs/disclosure_learning_redo_confirmatory_v1.json --output redo_v1

Reads the archive written by run_redo.py and writes <output>_analysis.json and <output>_tables.md next to this file:
- coverage gate (23.5) per table, stage D coverage acceptance, the old gate as a diagnostic;
- the five primary questions P1-P5 (23.6): one difference per replicate, one-sample t over replicates, Holm;
- the same verdicts without the replicates that contain a table failing the gate;
- auxiliary contrasts (unadjusted), the two research-question measures, M4, training randomness;
- the predictions D1-D4 (23.7) and descriptions (option shares, pre- vs post-policy tables, training counts,
  price truncation).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import _no_wmi
import redo_common as common

_no_wmi.prefill_uname()
from scipy import stats  # noqa: E402

KIND = {0: ("improve", .04), 1: ("wash", .04), 2: ("maintain", 0), 3: ("improve", .2), 4: ("wash", .2),
        5: ("improve", 1.0), 6: ("wash", 1.0)}
D1_SAMPLES = {"median_value_gap": 1.2e8, "p90_value_gap": 2e6}   # D.1 report section "结论" 3


def t_summary(values: list[float]) -> dict:
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(.975, n - 1)) * se
    t = mean / se if se > 0 else (math.inf if mean else 0.0)
    p = float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else (0.0 if mean else 1.0)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "p": p, "ci95": [mean - half, mean + half],
            "values": x.tolist()}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def verdict(summary: dict, holm_p: float, direction: str | None) -> str:
    if direction is None:
        return f"supported ({'positive' if summary['mean'] > 0 else 'negative'})" if holm_p < .05 else "unresolved"
    if holm_p >= .05:
        return "unresolved"
    return "supported" if (summary["mean"] > 0) == (direction == "positive") else "unsupported"


def load(spec: dict, output: Path) -> dict:
    reps = [row["replicate"] for row in spec["replicates"]]
    seeds, checks = spec["validation_seeds"], spec["check_seeds"]
    start = spec["window"]["start_day"]
    interval = spec["config"].get("firm_decision_interval", 60)
    data = {"reps": reps, "tables": {}, "pre": {}, "npv": {}, "gates": {}, "old_gates": {}, "truncation": []}
    for r, v in itertools.product(reps, common.VERSIONS):
        pre = common.table_folder(output, r, v, None)
        if pre is not None:
            rows = [common.outcome(common.ql_run(pre, seed)) for seed in checks]
            data["pre"][r, v] = {"T": np.mean([x["T"] for x in rows]), "W": np.mean([x["W"] for x in rows])}
            data["gates"][r, v, "pre"] = common.coverage_gate(pre, checks)
            data["old_gates"][r, v, "pre"] = common.read(pre / "summary.json")["validation"]["coverage"]["status"]
        for s in common.SCENARIOS:
            table = common.table_folder(output, r, v, s)
            if table is None:
                continue
            per_seed, choices = [], np.zeros(7)
            for seed in seeds:
                run = common.ql_run(table, seed)
                row = common.outcome(run)
                replay = common.replay_record(output, r, v, s, seed)
                row["N"] = None if replay is None else replay["trap_decisions"] / replay["decisions"]
                row["replay"] = replay
                with np.load(run / "result.npz") as result:
                    periods = np.arange(result["learning_choices"].shape[0]) * interval > start
                    choices += np.bincount(result["learning_choices"][periods].ravel(), minlength=7)
                    hits = np.flatnonzero(result["market_price_cap_hits"])
                    if hits.size:
                        data["truncation"].append({"replicate": r, "version": v, "scenario": s, "seed": seed,
                                                   "first_day": int(hits[0])})
                per_seed.append(row)
            with np.load(table / "frozen_model.npz") as model:
                counts = model["counts"]
            data["tables"][r, v, s] = {
                "seeds": per_seed, "choice_shares": (choices / choices.sum()).tolist(), "counts": counts,
                **{key: float(np.mean([row[key] for row in per_seed])) for key in ("T", "W", "improve_share", "maintain_share")},
                "N": (float(np.mean([row["N"] for row in per_seed])) if all(row["N"] is not None for row in per_seed) else None),
                "T_cap": float(np.mean([row["T_cap"] for row in per_seed])) if per_seed[0]["T_cap"] is not None else None,
                "W_by_size": {size: float(np.mean([row["wash_share_by_size"][size] for row in per_seed]))
                              for size in ("large", "mid", "small")},
            }
            data["gates"][r, v, s] = common.coverage_gate(table, seeds)
            data["old_gates"][r, v, s] = common.read(table / "summary.json")["validation"]["coverage"]["status"]
    for s, rule in itertools.product(common.SCENARIOS, spec["npv_rules"]):
        rows = [common.outcome(run) for seed in seeds if (run := common.npv_run(output, s, seed, rule)) is not None]
        if rows:
            data["npv"][s, rule] = {"T": float(np.mean([x["T"] for x in rows])), "W": float(np.mean([x["W"] for x in rows])),
                                    "T_seeds": [x["T"] for x in rows], "W_seeds": [x["W"] for x in rows],
                                    "W_by_size": {size: float(np.mean([x["wash_share_by_size"][size] for x in rows]))
                                                  for size in ("large", "mid", "small")}}
    return data


def value(data: dict, r: int, v: str, s: str, key: str):
    table = data["tables"].get((r, v, s))
    return None if table is None else table[key]


def diff(data: dict, reps: list[int], left: tuple, right: tuple | float, key: str) -> list[float]:
    out = []
    for r in reps:
        a = value(data, r, *left, key)
        b = right if isinstance(right, float) else value(data, r, *right, key)
        if a is not None and b is not None:
            out.append(a - b)
    return out


PRIMARY = {   # name: (left (version, scenario), right, metric, predicted direction)
    "P1_trap_exists_W_A_I_minus_npv_discrete": (("A", "I"), "npv_discrete", "W", "positive"),
    "P2_trap_visibility_N_I_V_minus_A": (("V", "I"), ("A", "I"), "N", "negative"),
    "P3_visibility_T_B_V_minus_A": (("V", "B"), ("A", "B"), "T", "positive"),
    "P4_information_T_A_I_minus_B": (("A", "I"), ("A", "B"), "T", "positive"),
    "P5_penalty_T_A_F_minus_B": (("A", "F"), ("A", "B"), "T", None),
}


def primary(data: dict, reps: list[int]) -> dict:
    rows = {}
    for name, (left, right, key, direction) in PRIMARY.items():
        # P1 compares with the discrete NPV mean on the same scenario and validation seeds.
        rhs = float(data["npv"][left[1], right][key]) if right == "npv_discrete" else right
        rows[name] = {"metric": key, "direction": direction, **t_summary(diff(data, reps, left, rhs, key))}
    adjusted = holm({name: row["p"] for name, row in rows.items() if row.get("p") is not None})
    for name, row in rows.items():
        row["holm_p"] = adjusted.get(name)
        row["verdict"] = verdict(row, row["holm_p"], row["direction"]) if row["holm_p"] is not None else "incomplete"
    return rows


def auxiliary(data: dict, reps: list[int]) -> dict:
    out = {}
    for v in common.VERSIONS:
        for key in ("T", "W"):
            for a, b in (("I", "B"), ("F", "B"), ("IF", "B"), ("I", "F")):
                out[f"{v}_{key}_{a}_minus_{b}"] = t_summary(diff(data, reps, (v, a), (v, b), key))
        out[f"{v}_W_I_minus_npv_discrete"] = t_summary(diff(data, reps, (v, "I"), float(data["npv"]["I", "npv_discrete"]["W"]), "W"))
        out[f"{v}_T_cap_I_minus_B"] = t_summary(diff(data, reps, (v, "I"), (v, "B"), "T_cap"))
    for s in common.SCENARIOS:
        for key in ("T", "W", "N", "improve_share"):
            out[f"V_minus_A_{key}_{s}"] = t_summary(diff(data, reps, ("V", s), ("A", s), key))
    for key in ("T", "W"):
        out[f"V_minus_A_{key}_prepolicy_check"] = t_summary(
            [data["pre"][r, "V"][key] - data["pre"][r, "A"][key] for r in reps if (r, "V") in data["pre"] and (r, "A") in data["pre"]])
    return {name: {k: v for k, v in row.items() if k != "values"} for name, row in out.items()}


def research_question(data: dict, reps: list[int]) -> dict:
    out = {}
    for v, s in itertools.product(common.VERSIONS, common.SCENARIOS):
        records = [row["replay"] for r in reps if (r, v, s) in data["tables"] for row in data["tables"][r, v, s]["seeds"]
                   if row["replay"] is not None]
        if not records:
            continue
        wash = sum(x["wash_decisions"] for x in records)
        losses = [loss for x in records for loss in x["wash_loss_bp"]]
        trap_losses = [loss for x in records for loss in x["trap_loss_vs_npv_choice_bp"]]
        out[f"{v}_{s}"] = {
            "runs": len(records), "decisions": sum(x["decisions"] for x in records), "wash_decisions": wash,
            "contract_19_4_share_of_wash_below_maintain": sum(x["wash_below_maintain"] for x in records) / wash if wash else None,
            "contract_19_4_median_loss_bp": float(np.median(losses)) if losses else None,
            "trap_share_of_decisions_N": sum(x["trap_decisions"] for x in records) / sum(x["decisions"] for x in records),
            "trap_share_of_wash": sum(x["trap_decisions"] for x in records) / wash if wash else None,
            "trap_median_loss_vs_npv_choice_bp": float(np.median(trap_losses)) if trap_losses else None,
            "npv_choice_of_trap_improve_wash_maintain": np.sum([x["npv_choice_of_trap_improve_wash_maintain"] for x in records], axis=0).tolist(),
            "by_size": {size: {key: sum(x["by_size"][size][key] for x in records)
                               for key in ("decisions", "wash_decisions", "wash_below_maintain", "trap_decisions")}
                        for size in ("large", "mid", "small")},
        }
    return out


def m4(data: dict, reps: list[int]) -> dict:
    out = {}
    npv = {rule: {s: data["npv"][s, rule]["T"] for s in common.SCENARIOS} for rule in ("npv_discrete", "npv")}
    npv_if = {rule: t_summary([a - b for a, b in zip(data["npv"]["I", rule]["T_seeds"], data["npv"]["F", rule]["T_seeds"])])
              for rule in npv}
    rank = {rule: sorted(common.SCENARIOS, key=lambda s: -npv[rule][s]) for rule in npv}
    for v in common.VERSIONS:
        complete = [r for r in reps if all((r, v, s) in data["tables"] for s in common.SCENARIOS)]
        means = {s: float(np.mean([value(data, r, v, s, "T") for r in complete])) for s in common.SCENARIOS}
        ql_rank = sorted(common.SCENARIOS, key=lambda s: -means[s])
        per_rep_sign = [np.sign(value(data, r, v, "I", "T") - value(data, r, v, "F", "T")) for r in complete]
        per_rep_rank = [sorted(common.SCENARIOS, key=lambda s: -value(data, r, v, s, "T")) for r in complete]
        out[v] = {"q_learning_mean_T": means, "q_learning_ranking": ql_rank, "replicates": len(complete),
                  "I_minus_F_mean": means["I"] - means["F"]}
        for rule in npv:
            sign = np.sign(npv_if[rule]["mean"])
            out[v][rule] = {"mean_T": npv[rule], "ranking": rank[rule], "I_minus_F": npv_if[rule]["mean"],
                            "I_minus_F_sign_agrees": bool(np.sign(means["I"] - means["F"]) == sign),
                            "replicates_with_agreeing_I_minus_F_sign": int(sum(x == sign for x in per_rep_sign)),
                            "ranking_agrees": ql_rank == rank[rule],
                            "replicates_with_agreeing_ranking": int(sum(x == rank[rule] for x in per_rep_rank))}
    return out


def randomness(data: dict, reps: list[int]) -> dict:
    out = {}
    for v, s, key in itertools.product(common.VERSIONS, common.SCENARIOS, ("T", "W")):
        groups = [[row[key] for row in data["tables"][r, v, s]["seeds"]] for r in reps if (r, v, s) in data["tables"]]
        if len(groups) < 2:
            continue
        array = np.asarray(groups)
        k = array.shape[1]
        between = float(array.mean(axis=1).std(ddof=1))
        within = float(np.median(array.std(axis=1, ddof=1)))
        msb = k * array.mean(axis=1).var(ddof=1)
        msw = float(array.var(axis=1, ddof=1).mean())
        out[f"{v}_{s}_{key}"] = {"between_table_sd": between, "median_within_table_sd": within,
                                 "ratio": between / within if within > 0 else None,
                                 "icc": (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw > 0 else None,
                                 "table_means": array.mean(axis=1).tolist()}
    return out


def predictions(data: dict, reps: list[int], rand: dict, rq: dict) -> dict:
    d1 = {s: rand.get(f"A_{s}_W", {}).get("ratio") for s in common.SCENARIOS}
    d2 = rq.get("A_I", {}).get("trap_median_loss_vs_npv_choice_bp")
    f_b = t_summary(diff(data, reps, ("A", "F"), ("A", "B"), "T"))
    i_b = t_summary(diff(data, reps, ("A", "I"), ("A", "B"), "T"))
    d4 = t_summary(diff(data, reps, ("V", "B"), ("A", "B"), "improve_share"))
    return {
        "D1_training_randomness": {"ratios": d1, "verdict": "成立" if all(x is not None and x > 3 for x in d1.values()) else "不成立"},
        "D2_trap_loss_unlearnable": {"median_bp": d2, "verdict": "成立" if d2 is not None and d2 < .5 else "不成立"},
        "D3_penalty_harder_than_information": {"abs_F_minus_B": abs(f_b["mean"]), "I_minus_B": i_b["mean"],
                                               "verdict": "成立" if abs(f_b["mean"]) < i_b["mean"] else "不成立"},
        "D4_visible_improvement_raises_improvement": {**{k: v for k, v in d4.items() if k != "values"},
                                                      "verdict": "成立" if d4["ci95"][0] > 0 else ("相反" if d4["ci95"][1] < 0 else "不确定")},
    }


def descriptions(data: dict, reps: list[int], spec: dict) -> dict:
    shares = {}
    for v, s in itertools.product(common.VERSIONS, common.SCENARIOS):
        rows = [data["tables"][r, v, s]["choice_shares"] for r in reps if (r, v, s) in data["tables"]]
        if rows:
            mean = np.mean(rows, axis=0)
            shares[f"{v}_{s}"] = {f"{KIND[i][0]}_{KIND[i][1]}": float(mean[i]) for i in range(7)}
    pre_post = {v: {"pre_check_W": [data["pre"][r, v]["W"] for r in reps if (r, v) in data["pre"]],
                    "post_B_W": [value(data, r, v, "B", "W") for r in reps]} for v in common.VERSIONS}
    counts = {}
    for v, s in itertools.product(common.VERSIONS, common.SCENARIOS):
        values = []
        for r in reps:
            table = data["tables"].get((r, v, s))
            if table is None:
                continue
            counts_post = table["counts"][3:]
            visited = counts_post.sum(axis=2) > 0
            values.extend(counts_post[visited].ravel().tolist())
        if values:
            x = np.asarray(values)
            counts[f"{v}_{s}"] = {"post_group_cells": int(x.size), "min": int(x.min()), "p10": float(np.quantile(x, .1)),
                                  "median": float(np.median(x)), "max": int(x.max()),
                                  "share_cells_reaching_D1_p90_requirement": float((x >= D1_SAMPLES["p90_value_gap"]).mean())}
    npv_size = {f"{s}_{rule}": data["npv"][s, rule]["W_by_size"] for s, rule in data["npv"]}
    ql_size = {f"{v}_{s}": {size: float(np.mean([data["tables"][r, v, s]["W_by_size"][size] for r in reps if (r, v, s) in data["tables"]]))
                            for size in ("large", "mid", "small")}
               for v, s in itertools.product(common.VERSIONS, common.SCENARIOS)
               if any((r, v, s) in data["tables"] for r in reps)}
    return {"option_shares_in_window": shares, "prepolicy_vs_post_B_wash_share": pre_post,
            "post_group_training_counts_visited_cells": counts, "d1_required_samples": D1_SAMPLES,
            "wash_share_by_size": {"q_learning": ql_size, "npv": npv_size},
            "npv_means": {f"{s}_{rule}": {"T": row["T"], "W": row["W"]} for (s, rule), row in data["npv"].items()},
            "price_truncation": data["truncation"]}


def tables_md(result: dict) -> str:
    lines = ["# 学习型企业重做：主要结果（自动生成）", "", "| 问题 | n | 均值 | 95% 区间 | Holm p | 判定 |", "|---|---:|---:|---|---:|---|"]
    for name, row in result["primary"].items():
        if row.get("p") is None:
            lines.append(f"| {name} | {row['n']} | — | — | — | {row['verdict']} |")
            continue
        lines.append(f"| {name} | {row['n']} | {row['mean']:+.4f} | [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | "
                     f"{row['holm_p']:.2g} | {row['verdict']} |")
    lines += ["", f"覆盖验收：{result['coverage']['acceptance']}（{result['coverage']['passing_tables']}/{result['coverage']['tables']} 张表通过）", ""]
    lines += ["| 预测 | 判定 |", "|---|---|"] + [f"| {name} | {row['verdict']} |" for name, row in result["predictions"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = common.load_spec(args.spec)
    output = args.output if args.output.is_absolute() else common.HERE / args.output
    data = load(spec, output)
    reps = data["reps"]
    gates = {f"r{r}_{v}_{s}": gate["status"] for (r, v, s), gate in data["gates"].items()}
    failing_reps = sorted({r for (r, v, s), gate in data["gates"].items() if gate["status"] != "PASS"})
    coverage = {"tables": len(gates), "passing_tables": sum(x == "PASS" for x in gates.values()),
                "expected_tables": len(reps) * 2 * 5,
                "acceptance": "PASS" if len(gates) == len(reps) * 10 and all(x == "PASS" for x in gates.values()) else "FAIL",
                "status": gates, "failing_replicates": failing_reps,
                "max_pooled_undercovered_share": max(g["max_pooled_undercovered_share"] for g in data["gates"].values()),
                "max_world_undercovered_share": max(g["max_world_undercovered_share"] for g in data["gates"].values()),
                "max_unseen_share": max(g["max_unseen_share"] for g in data["gates"].values()),
                "old_gate_diagnostic": {f"r{r}_{v}_{s}": x for (r, v, s), x in data["old_gates"].items()}}
    rand = randomness(data, reps)
    rq = research_question(data, reps)
    result = {"spec": args.spec.as_posix(), "output": output.name, "coverage": coverage,
              "primary": primary(data, reps),
              "primary_without_failing_replicates": primary(data, [r for r in reps if r not in failing_reps])
              if failing_reps else "no table failed the gate",
              "auxiliary_unadjusted": auxiliary(data, reps), "research_question": rq, "m4": m4(data, reps),
              "training_randomness": rand, "predictions": predictions(data, reps, rand, rq),
              "descriptions": descriptions(data, reps, spec),
              "table_values": {f"r{r}_{v}_{s}": {k: t[k] for k in ("T", "W", "N", "improve_share", "maintain_share", "T_cap")}
                               for (r, v, s), t in data["tables"].items()},
              "prepolicy_check_values": {f"r{r}_{v}": x for (r, v), x in data["pre"].items()}}
    text = json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n"
    (common.HERE / f"{output.name}_analysis.json").write_text(text, encoding="utf-8", newline="\n")
    (common.HERE / f"{output.name}_tables.md").write_text(tables_md(result), encoding="utf-8", newline="\n")
    print(tables_md(result))


if __name__ == "__main__":
    main()
