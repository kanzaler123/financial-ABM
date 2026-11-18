"""Pipeline check of an H3 archive (contract 25.8, 25.10). Read-only on the archive; reports counts, no estimate.

  python check_pipeline.py check_v1   # development archive
  python check_pipeline.py h3_v1      # confirmatory archive, before analysis

Checks: every planned trial complete and its failed attempts; core validation failures with their first price cap
day; metric windows; 38 x 200 decisions in the policy window; within each block (main Q block, u0p2_, perm_) every
scenario's inflation, financing rates and actions before the switch on day 1201 bit-identical across policies for
each seed; targeted audits and assurance active after the switch (firm-level audit probabilities under V and VF,
flags under J and J_hidden, assigned providers under the assurance scenarios) and absent elsewhere. Writes
<run>_pipeline.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PERIOD, SWITCH, FIRST, LAST = 60, 1201, 21, 58
WINDOW = slice(FIRST, LAST + 1)
TARGETED = ("A2", "V", "V_hidden", "J", "J_hidden", "VF")
ASSURED = ("ALL_R1", "ALL_R16", "ALL_F1", "ALL_F16", "LRG_R16", "LRG_F16")
BLOCKS = {"main": ("B", "I", "F", "IF", "F_2x", "F_8x") + TARGETED + ASSURED,
          "u0p2": ("u0p2_B", "u0p2_I", "u0p2_F"), "perm": ("perm_B", "perm_I", "perm_F")}
NAMES = ("actions", "inflation", "financing_rate", "market_price_cap_hits", "audit_probability", "audit_flags",
         "assurer_assignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    args = parser.parse_args()
    run = HERE / args.run
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == frozen["experiment_id"]
    seeds = frozen["spec"]["seeds"]
    names = [row["name"] for row in frozen["scenarios"]]
    assert sorted(names) == sorted(sum(BLOCKS.values(), ()))
    rows = {(r["scenario"], r["seed"]): r for r in summary["records"]}
    complete = {key: r for key, r in rows.items() if r["status"] == "complete"}

    def arrays(key):
        with np.load(run / complete[key]["path"] / "result.npz") as data:
            return {name: data[name] for name in NAMES if name in data.files}

    attempts: Counter = Counter()
    for record in summary["records"]:
        for attempt in record["attempts"]:
            attempts[attempt["status"]] += 1
    loaded = {key: arrays(key) for key in complete}
    windows = sorted({(r["metrics"]["window"]["start_day"], r["metrics"]["window"]["end_day"]) for r in complete.values()})
    decisions = sorted({r["metrics"]["actions"]["decisions"] for r in complete.values()})
    invalid = []
    for key, r in sorted(complete.items()):
        if any(r["validation"][gate]["status"] != "PASS" for gate in ("mechanical", "stylized")):
            hits = np.flatnonzero(loaded[key]["market_price_cap_hits"])
            invalid.append({"scenario": key[0], "seed": key[1], "first_cap_day": int(hits[0]) if hits.size else None})
    pairing = {}
    for block, members in BLOCKS.items():
        compared = differing = 0
        for seed in seeds:
            series = [loaded[(name, seed)] for name in members if (name, seed) in loaded]
            series = [(d["inflation"][:SWITCH], d["financing_rate"][:SWITCH], d["actions"][:SWITCH // PERIOD + 1])
                      for d in series]
            for other in series[1:]:
                compared += 1
                differing += not all(np.array_equal(x, y) for x, y in zip(series[0], other))
        pairing[block] = {"pairs_compared": compared, "pairs_differing_before_switch": differing}
    activity = {}
    for name in names:
        cells = [loaded[(name, seed)] for seed in seeds if (name, seed) in loaded]
        row = {"has_audit_probability": sum("audit_probability" in d for d in cells),
               "has_audit_flags": sum("audit_flags" in d for d in cells),
               "has_assurer_assignment": sum("assurer_assignment" in d for d in cells)}
        if row["has_audit_probability"]:
            # Firm-level rates differ across firms after the switch only under the value rule or after flags.
            row["periods_with_unequal_rates"] = int(sum(
                (d["audit_probability"][WINDOW].std(axis=1) > 0).sum() for d in cells if "audit_probability" in d))
        if row["has_audit_flags"]:
            row["flags_in_window"] = int(sum(d["audit_flags"][WINDOW].sum() for d in cells if "audit_flags" in d))
        if row["has_assurer_assignment"]:
            row["assigned_firm_periods_in_window"] = int(sum(
                (d["assurer_assignment"][WINDOW] >= 0).sum() for d in cells if "assurer_assignment" in d))
        activity[name] = row
    out = {"experiment_id": frozen["experiment_id"], "purpose": frozen["spec"]["purpose"],
           "planned": summary["planned_trials"], "complete": summary["complete_trials"],
           "attempts_by_status": dict(sorted(attempts.items())),
           "failing_core_validation": summary["trials_failing_core_validation"], "core_validation_failures": invalid,
           "metric_windows": windows, "policy_window_decisions_per_trial": decisions,
           "pre_switch_pairing": pairing, "policy_activity": activity}
    expected_active = {"has_audit_probability": {"V", "V_hidden", "J", "J_hidden", "VF"},
                       "has_assurer_assignment": set(ASSURED)}
    out["activity_where_expected"] = all(
        (activity[name][field] == len(seeds)) == (name in members) for field, members in expected_active.items()
        for name in names)
    (HERE / f"{args.run}_pipeline.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
