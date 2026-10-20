"""Pipeline checks for contract section 21 on the development run (seeds 34001-34003); no effect estimates.

  python check_pipeline.py [--archive <development archive>]

From the compact development archive (default check_v1/):
  - every trial complete, metric windows 1200-3500;
  - within each market setting (natural decay 0.2: 9 scenarios; no decay: 6), inflation, financing rates and
    actions up to day 1200 are bit-identical across scenarios;
  - budget: in every targeted scenario, before the switch every audit probability is 1%, and on every audit day
    after it the probabilities sum to n * 1% = 2 (error at most 1e-12);
  - realized audits per audit day and, for the jump rule, flagged firms per audit day (descriptive).
From full-archive reruns of seed 34001 (V and J, in work_check/pipeline/, local only):
  - V: audit probabilities proportional to market value on the audit day;
  - J: flags equal report rises above 0.125 SD recomputed from true values and inflation; flagged firms get
    the higher probability.
Writes check_v1_pipeline.json (LF).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import _no_wmi  # noqa: E402

_no_wmi.prefill_uname()

import numpy as np  # noqa: E402

PERIOD, SWITCH_PERIOD, LAST_PRE_SWITCH_DAY = 60, 21, 1200
SETTINGS = {"decay 0.2": ("B", "I", "F", "A2", "V", "V_hidden", "J", "J_hidden", "VF"),
            "no decay": ("decay0_B", "decay0_I", "decay0_V", "decay0_J", "decay0_J_hidden", "decay0_VF")}
TARGETED = ("V", "V_hidden", "J", "J_hidden", "VF", "decay0_V", "decay0_J", "decay0_J_hidden", "decay0_VF")
RERUN_SEED, RERUNS = 34001, ("V", "J")


def complete_attempt(archive: Path, scenario: str, seed: int) -> Path | None:
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*")):
        manifest = attempt / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete":
            return attempt
    return None


def compact_arrays(path: Path) -> SimpleNamespace:
    """The arrays used here from a compact result archive (load_result needs a full one)."""
    names = ("inflation", "financing_rate", "actions", "audit_selected", "audit_probability", "audit_flags")
    with np.load(path / "result.npz") as data:
        return SimpleNamespace(**{name: data[name] if name in data.files else None for name in names})


def rerun(scenario: dict, destination: str) -> None:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import run_and_archive
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, RERUN_SEED)
    run_and_archive(config, domain, destination, policy, archive="full")


def main() -> None:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import load_result

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=HERE / "check_v1")
    args = parser.parse_args()
    frozen = json.loads((args.archive / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    names = [row["name"] for row in frozen["scenarios"]]
    out: dict = {"experiment_id": frozen["experiment_id"], "planned": len(names) * len(seeds)}

    attempts = {(name, seed): complete_attempt(args.archive, name, seed) for name in names for seed in seeds}
    out["complete"] = sum(path is not None for path in attempts.values())
    out["missing_or_failed"] = sorted(f"{name}/{seed}" for (name, seed), path in attempts.items() if path is None)
    summary = json.loads((args.archive / "summary.json").read_text(encoding="utf-8"))
    out["metric_windows"] = sorted({(result["window"]["start_day"], result["window"]["end_day"])
                                    for comparison in summary["comparisons"]
                                    for metric, result in comparison["metrics"].items()
                                    if not metric.startswith("full_period_change.")})
    out["failing_core_validation"] = summary["trials_failing_core_validation"]

    results = {key: compact_arrays(path) for key, path in attempts.items() if path is not None}
    pairing = {}
    for setting, group in SETTINGS.items():
        compared = differing = 0
        for seed in seeds:
            reference = results[(group[0], seed)]
            for name in group[1:]:
                other = results[(name, seed)]
                compared += 1
                same = (np.array_equal(reference.inflation[:LAST_PRE_SWITCH_DAY + 1], other.inflation[:LAST_PRE_SWITCH_DAY + 1])
                        and np.array_equal(reference.financing_rate[:LAST_PRE_SWITCH_DAY + 1],
                                           other.financing_rate[:LAST_PRE_SWITCH_DAY + 1])
                        and np.array_equal(reference.actions[:SWITCH_PERIOD], other.actions[:SWITCH_PERIOD]))
                differing += not same
        pairing[setting] = {"pairs_compared": compared, "pairs_differing_before_switch": differing}
    out["pre_switch_pairing"] = pairing

    budget, realized = {}, {}
    for name in names:
        selected = [results[(name, seed)].audit_selected[SWITCH_PERIOD:].sum(axis=1).mean() for seed in seeds]
        realized[name] = round(float(np.mean(selected)), 3)
        if name not in TARGETED:
            assert all(results[(name, seed)].audit_probability is None for seed in seeds), name
            continue
        worst = 0.
        for seed in seeds:
            probability = results[(name, seed)].audit_probability
            assert (probability[1:SWITCH_PERIOD] == .01).all(), (name, seed)
            worst = max(worst, float(np.abs(probability[SWITCH_PERIOD:].sum(axis=1) - 2.).max()))
        budget[name] = worst
    out["budget_max_abs_error_after_switch"] = budget
    out["budget_ok"] = all(error <= 1e-12 for error in budget.values())
    out["realized_audits_per_audit_day_after_switch"] = realized
    out["flagged_per_audit_day_after_switch"] = {
        name: round(float(np.mean([results[(name, seed)].audit_flags[SWITCH_PERIOD:].sum(axis=1).mean() for seed in seeds])), 2)
        for name in TARGETED if "J" in name}

    rows = {row["name"]: row for row in frozen["scenarios"]}
    work = HERE / "work_check" / "pipeline"
    work.mkdir(parents=True, exist_ok=True)
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=len(RERUNS)) as pool:
        for future in [pool.submit(rerun, rows[name], str(work / name)) for name in RERUNS
                       if not (work / name / "result.npz").exists()]:
            future.result()
    full = {name: load_result(work / name / "result.npz") for name in RERUNS}
    for name in RERUNS:
        assert np.array_equal(full[name].actions, results[(name, RERUN_SEED)].actions), f"{name} rerun differs"
    value = full["V"]
    ratios = [value.audit_probability[k] / value.market.prices[k * PERIOD] for k in range(SWITCH_PERIOD, value.actions.shape[0])]
    out["value_rule_max_relative_spread"] = float(max(np.ptp(r) / r.mean() for r in ratios))
    jump = full["J"]
    top = _inputs(rows["J"], RERUN_SEED)[1].report_bound
    report = np.minimum(jump.true_goodness + jump.inflation, top)
    mismatches = ordered = 0
    for k in range(SWITCH_PERIOD, jump.actions.shape[0]):
        day = k * PERIOD
        # The report just before this audit against the report just after the previous one.
        before = np.minimum(jump.true_goodness[day] + jump.inflation[day - 1], top)
        flags = jump.audit_flags[k]
        mismatches += int((flags != (before - report[day - PERIOD] > .125)).sum())
        if flags.any() and not flags.all():
            ordered += int(jump.audit_probability[k][flags].min() > jump.audit_probability[k][~flags].max())
        else:
            ordered += 1
    out["jump_flag_mismatches"] = mismatches
    out["jump_flagged_get_higher_probability"] = ordered == jump.actions.shape[0] - SWITCH_PERIOD
    out["all_ok"] = (out["complete"] == out["planned"] and out["metric_windows"] == [(1200, 3500)]
                     and all(row["pairs_differing_before_switch"] == 0 for row in pairing.values())
                     and out["budget_ok"] and out["value_rule_max_relative_spread"] < 1e-10
                     and mismatches == 0 and out["jump_flagged_get_higher_probability"])
    (HERE / "check_v1_pipeline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
