"""Pipeline checks for the learning-firm redo (contract 23.9). Prints no effect estimates.

  python check_pipeline.py --spec configs/disclosure_learning_redo_development_v1.json --output dev_v1 [--confirmatory]

Development (all items) and confirmatory (items 1, 2, 4, 5):
1. completion: every job of the spec has a complete attempt;
2. frozen tables: every evaluation run starts and ends with the table's frozen fingerprint, and the frozen model file
   still has the hash recorded in the table summary;
3. windows: every Q-learning and NPV run measures days 1200-3500;
4. pre-switch identity: within each replicate and version, the four scenarios' Q-learning runs on one validation seed
   have identical actions, amounts, inflation and financing rates before the switch (and likewise the NPV runs);
5. replays: every replay reproduced the archived fingerprint and found the state inputs of its version
   (version V: the stock equals the committed real improvement and sets the third feature);
6. the coverage gate (contract 23.5) can be computed for every table; only pass/fail counts are printed.
Writes <output>_pipeline.json next to this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import redo_common as common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    spec = common.load_spec(args.spec)
    output = args.output if args.output.is_absolute() else common.HERE / args.output
    reps = [row["replicate"] for row in spec["replicates"]]
    versions, scenarios, seeds = list(spec["versions"]), list(spec["scenarios"]), spec["validation_seeds"]
    switch_period = (spec["switch_day"] - 1) // spec["config"].get("firm_decision_interval", 60) + 1
    problems: list[str] = []
    checks: dict = {}

    # 1. completion
    missing = []
    tables = {}
    for r in reps:
        for v in versions:
            for s in [None] + scenarios:
                folder = common.table_folder(output, r, v, s)
                tables[r, v, s] = folder
                if folder is None:
                    missing.append(f"table r{r} {v} {s or 'pre'}")
                elif s is not None:
                    for seed in seeds:
                        if common.replay_record(output, r, v, s, seed) is None:
                            missing.append(f"replay r{r} {v} {s} {seed}")
    for s in scenarios:
        for seed in seeds:
            for rule in spec["npv_rules"]:
                if common.npv_run(output, s, seed, rule) is None:
                    missing.append(f"npv {s} {seed} {rule}")
    checks["completion"] = {"pass": not missing, "missing": missing,
                            "tables": sum(folder is not None for folder in tables.values())}

    # 2. frozen tables
    frozen_bad = []
    for key, folder in tables.items():
        if folder is None:
            continue
        summary = common.read(folder / "summary.json")
        if common.file_sha256(folder / "frozen_model.npz") != summary["frozen_model_sha256"]:
            frozen_bad.append(f"{key}: frozen model hash")
        evaluation = spec["check_seeds"] if key[2] is None else seeds
        for seed in evaluation:
            report = common.read(common.ql_run(folder, seed) / "learning.json")
            if not (report["initial_table_fingerprint"] == report["final_table_fingerprint"]
                    == summary["frozen_table_fingerprint"]):
                frozen_bad.append(f"{key} seed {seed}: table changed")
    checks["frozen_tables"] = {"pass": not frozen_bad, "problems": frozen_bad}

    # 3. windows
    window_bad = []
    expected = (spec["window"]["start_day"], spec["window"]["end_day"])
    runs = [common.ql_run(folder, seed) for key, folder in tables.items() if folder is not None and key[2] is not None
            for seed in seeds]
    runs += [common.npv_run(output, s, seed, rule) for s in scenarios for seed in seeds for rule in spec["npv_rules"]]
    for run in runs:
        if run is None:
            continue
        window = common.metrics(run)["window"]
        if (window["start_day"], window["end_day"]) != expected:
            window_bad.append(run.relative_to(output).as_posix())
    checks["windows"] = {"pass": not window_bad, "runs": len(runs), "problems": window_bad}

    # 4. pre-switch identity
    def prefix(run: Path) -> list[np.ndarray]:
        with np.load(run / "result.npz") as result:
            return [result["actions"][:switch_period], result["amounts"][:switch_period],
                    result["inflation"][:spec["switch_day"]], result["financing_rate"][:spec["switch_day"]]]

    pairs, differing = 0, []
    groups = [[common.ql_run(tables[r, v, s], seed) for s in scenarios if tables[r, v, s] is not None]
              for r in reps for v in versions for seed in seeds]
    groups += [[common.npv_run(output, s, seed, rule) for s in scenarios if common.npv_run(output, s, seed, rule)]
               for seed in seeds for rule in spec["npv_rules"]]
    for group in groups:
        if len(group) < 2:
            continue
        first = prefix(group[0])
        for other in group[1:]:
            pairs += 1
            if not all(np.array_equal(a, b) for a, b in zip(first, prefix(other))):
                differing.append(other.relative_to(output).as_posix())
    checks["prefix_identity"] = {"pass": not differing, "pairs": pairs, "differing": differing,
                                 "switch_period": switch_period}

    # 5. replays
    replay_bad, replays = [], 0
    for r in reps:
        for v in versions:
            for s in scenarios:
                for seed in seeds:
                    record = common.replay_record(output, r, v, s, seed)
                    if record is None:
                        continue
                    replays += 1
                    context = spec["versions"][v]
                    if not (record["fingerprint_matches"] and record["state_inputs_ok"] and record["context"] == context):
                        replay_bad.append(f"r{r} {v} {s} {seed}")
    checks["replays"] = {"pass": not replay_bad, "replays": replays, "problems": replay_bad}

    # 6. coverage gate computable (statuses only)
    if not args.confirmatory:
        statuses = {}
        for key, folder in tables.items():
            if folder is None:
                continue
            gate = common.coverage_gate(folder, spec["check_seeds"] if key[2] is None else seeds)
            statuses[gate["status"]] = statuses.get(gate["status"], 0) + 1
        checks["coverage_gate_computed"] = {"pass": bool(statuses), "status_counts": statuses}

    result = {"spec": args.spec.as_posix(), "output": output.name, "confirmatory": args.confirmatory, "checks": checks,
              "all_pass": all(item["pass"] for item in checks.values())}
    path = common.HERE / f"{output.name}_pipeline.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({name: item["pass"] for name, item in checks.items()} | {"all_pass": result["all_pass"]}, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
