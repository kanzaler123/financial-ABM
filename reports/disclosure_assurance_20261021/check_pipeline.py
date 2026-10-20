"""Pipeline checks for contract section 22 on the development run (seeds 35001-35003); no effect estimates.

  python check_pipeline.py [--archive <development archive>]

From the compact development archive (default check_v1/):
  - every trial complete, metric windows 1200-3500;
  - within each market setting (natural decay 0.2: 13 scenarios; no decay: 5), inflation, financing rates and
    actions up to day 1200 are bit-identical across scenarios;
  - quality rule: in every assurance scenario each provider's quality equals min(1, lambda s), with s recomputed
    from the archive (first assurance year: share of its clients with inflation after the day-1260 audit; later
    years: share of its checks in the past year whose report carried inflation), error at most 1e-12;
  - checks: none before the first assurance year; corrections only of checked firms whose report carried
    inflation, and corrected reports are clean on the check day; only large firms checked under LRG;
  - ALL_R0: quality 0 throughout and no corrections; VOL16: no firm buys assurance, and actions and inflation equal
    those of B;
  - descriptive: checks, corrections and liability events per period, mean quality (no effect estimates).
Writes check_v1_pipeline.json (LF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import _no_wmi  # noqa: E402

_no_wmi.prefill_uname()

import numpy as np  # noqa: E402

PERIOD, DELAY, SWITCH_PERIOD, LAST_PRE_SWITCH_DAY, YEAR = 60, 20, 21, 1200, 4
FIRST_DAY = SWITCH_PERIOD * PERIOD
ARRAYS = ("inflation", "financing_rate", "actions", "market_size_class", "assurer_assignment",
          "assurance_checked_inflated", "assurance_corrections", "assurer_quality", "assurer_inflation_share",
          "assurer_liability_events")


def complete_attempt(archive: Path, scenario: str, seed: int) -> Path | None:
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*")):
        manifest = attempt / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete":
            return attempt
    return None


def compact_arrays(path: Path) -> SimpleNamespace:
    with np.load(path / "result.npz") as data:
        return SimpleNamespace(**{name: data[name] if name in data.files else None for name in ARRAYS})


def quality_error(result: SimpleNamespace, slope: float) -> float:
    """Largest gap between the archived quality and min(1, lambda s) with s recomputed from the archive."""
    assignment, inflated = result.assurer_assignment, result.assurance_checked_inflated
    count = result.assurer_quality.shape[1]
    worst = 0.
    for start in range(SWITCH_PERIOD, assignment.shape[0], YEAR):
        if start == SWITCH_PERIOD:
            clients = assignment[start]
            present = result.inflation[FIRST_DAY] > 0
            share = np.array([present[clients == k].mean() if (clients == k).any() else 0. for k in range(count)])
        else:
            year = slice(start - YEAR, start)
            share = np.array([inflated[year][assignment[year] == k].mean() if (assignment[year] == k).any() else 0.
                              for k in range(count)])
        quality = np.minimum(1., slope * share)
        for period in range(start, min(start + YEAR, assignment.shape[0])):
            worst = max(worst, float(np.abs(result.assurer_quality[period] - quality).max()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=HERE / "check_v1")
    args = parser.parse_args()
    frozen = json.loads((args.archive / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    rows = {row["name"]: row for row in frozen["scenarios"]}
    names = list(rows)
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

    settings = {"decay 0.2": [n for n in names if not n.startswith("decay0_")],
                "no decay": [n for n in names if n.startswith("decay0_")]}
    pairing = {}
    for setting, group in settings.items():
        compared = differing = 0
        for seed in seeds:
            reference = results[(group[0], seed)]
            for name in group[1:]:
                other = results[(name, seed)]
                compared += 1
                differing += not (
                    np.array_equal(reference.inflation[:LAST_PRE_SWITCH_DAY + 1], other.inflation[:LAST_PRE_SWITCH_DAY + 1])
                    and np.array_equal(reference.financing_rate[:LAST_PRE_SWITCH_DAY + 1],
                                       other.financing_rate[:LAST_PRE_SWITCH_DAY + 1])
                    and np.array_equal(reference.actions[:SWITCH_PERIOD], other.actions[:SWITCH_PERIOD]))
        pairing[setting] = {"pairs_compared": compared, "pairs_differing_before_switch": differing}
    out["pre_switch_pairing"] = pairing

    quality, checks, description = {}, {}, {}
    for name in names:
        intervention = rows[name]["policy"]["intervention"] or {}
        scope = intervention.get("assurance_scope", "none")
        if scope == "none":
            assert all(results[(name, seed)].assurer_assignment is None for seed in seeds), name
            continue
        slope = intervention.get("assurer_liability", 0.) * intervention["audit_probability"] / (
            2. * intervention["assurer_effort_cost"])
        quality[name] = max(quality_error(results[(name, seed)], slope) for seed in seeds)
        ok = True
        rows_described = []
        for seed in seeds:
            r = results[(name, seed)]
            assigned = r.assurer_assignment >= 0
            ok &= not assigned[:SWITCH_PERIOD].any() and not r.assurance_checked_inflated[:SWITCH_PERIOD].any()
            ok &= not (r.assurance_checked_inflated & ~assigned).any()
            ok &= not (r.assurance_corrections & ~r.assurance_checked_inflated).any()
            for period in np.flatnonzero(r.assurance_corrections.any(axis=1)):
                ok &= bool((r.inflation[period * PERIOD + DELAY][r.assurance_corrections[period]] == 0).all())
            if scope == "large":
                ok &= not assigned[:, r.market_size_class != 0].any()
                ok &= bool(assigned[SWITCH_PERIOD:, r.market_size_class == 0].all())
            if scope == "all":
                ok &= bool(assigned[SWITCH_PERIOD:].all())
            window = slice(SWITCH_PERIOD, None)
            rows_described.append((assigned[window].sum(axis=1).mean(), r.assurance_checked_inflated[window].sum(axis=1).mean(),
                                   r.assurance_corrections[window].sum(axis=1).mean(),
                                   r.assurer_liability_events[window].sum(axis=1).mean(), r.assurer_quality[window].mean()))
        checks[name] = bool(ok)
        description[name] = dict(zip(("checked_firms", "checks_with_inflation", "corrections", "liability_events",
                                      "mean_quality"),
                                     (round(float(np.mean(column)), 4) for column in zip(*rows_described))))
    out["quality_rule_max_abs_error"] = quality
    out["checks_follow_the_rule"] = checks
    out["per_period_after_switch_descriptive"] = description

    zero = [results[("ALL_R0", seed)] for seed in seeds]
    out["ALL_R0_quality_zero_no_corrections"] = all(not r.assurer_quality.any() and not r.assurance_corrections.any()
                                                    for r in zero)
    out["VOL16_no_buyers_and_equal_to_B"] = all(
        (results[("VOL16", seed)].assurer_assignment == -1).all()
        and np.array_equal(results[("VOL16", seed)].actions, results[("B", seed)].actions)
        and np.array_equal(results[("VOL16", seed)].inflation, results[("B", seed)].inflation)
        for seed in seeds)
    out["all_ok"] = (out["complete"] == out["planned"] and out["metric_windows"] == [(1200, 3500)]
                     and all(row["pairs_differing_before_switch"] == 0 for row in pairing.values())
                     and all(error <= 1e-12 for error in quality.values()) and all(checks.values())
                     and out["ALL_R0_quality_zero_no_corrections"] and out["VOL16_no_buyers_and_equal_to_B"])
    (HERE / "check_v1_pipeline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
