"""Pipeline check of a stage I archive (contract 26.9, 26.11, 26.20 (f)). Read-only on the archive; reports counts,
no estimate.

  python check_pipeline.py check_v2        # development archive
  python check_pipeline.py stage_i_v2      # confirmatory archive, before analysis

v2 (contract 26.20): the blocks differ in financing_rating_slope lambda instead of financing_spread. The frozen
configs must carry lambda where the spec puts it (none at lambda0 = 0) and financing_spread 0.05 everywhere; the
day-0 rate's cross-sectional mean is r_f + s f / (1 + f) (the rating term averages zero), its firm-level deviation
is zero at lambda0 and nonzero otherwise. Also reported (26.20 (f)): completion over the full 3500 days, failure
messages, the range of the contract rate over all days and the price cap hits. The v1 description follows.

Checks: every planned trial complete and its failed attempts; core validation failures with their first price cap
day; metric windows; 38 x 200 decisions in the policy window; financing_spread only where the spec puts it (the
frozen scenario configs, and in every run the day-0 rate r_f + s E / (E + P0) = r_f + s f / (1 + f)); within each
block of one spread and one domain (the s* main block, the s* perm_ block, each sensitivity block) every scenario's
inflation, financing rates and actions before the switch on day 1201 bit-identical across policies for each seed;
targeted audits and assurance active only in their scenarios. Writes <run>_pipeline.json.
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
AUDITED = ("V", "V_hidden", "J", "J_hidden", "VF")
PREFIXES = {"s0": "s0_", "s_lo": "slo_", "s_hi": "shi_", "s_28": "s28_"}
STAR, POINTS_KEY = "s_star", "financing_spreads"
V2 = False
NAMES = ("actions", "inflation", "financing_rate", "market_price_cap_hits", "audit_probability", "audit_flags",
         "assurer_assignment")


def base_name(name: str) -> str:
    return next((name[len(p):] for p in PREFIXES.values() if name.startswith(p)), name)


def configure(spec: dict) -> None:
    global PREFIXES, STAR, POINTS_KEY, V2
    if "financing_rating_slopes" in spec:
        PREFIXES = {"lambda0": "l0_", "lambda_lo": "llo_", "lambda_hi": "lhi_", "lambda_fe": "lfe_"}
        STAR, POINTS_KEY, V2 = "lambda_star", "financing_rating_slopes", True


def blocks(names: list[str]) -> dict[str, list[str]]:
    """Scenarios that share a spread and a domain: their histories before the switch must coincide."""
    out: dict[str, list[str]] = {}
    for name in names:
        prefix = next((p for p in PREFIXES.values() if name.startswith(p)), "")
        key = prefix + ("perm" if base_name(name).startswith("perm_") else "main") if prefix else (
            STAR + ("_perm" if name.startswith("perm_") else "_main"))
        out.setdefault(key, []).append(name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    args = parser.parse_args()
    run = HERE / args.run
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == frozen["experiment_id"]
    spec = frozen["spec"]
    configure(spec)
    seeds, spreads = spec["seeds"], spec[POINTS_KEY]
    names = [row["name"] for row in frozen["scenarios"]]
    scenario = {row["name"]: row for row in frozen["scenarios"]}
    expected = {}
    for name in names:
        prefix = next((key for key, p in PREFIXES.items() if name.startswith(p)), None)
        expected[name] = spreads[prefix] if prefix else spreads[STAR]
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
    # financing_spread (v1) or financing_rating_slope (v2): in the frozen configs, and in the day-0 contract rate.
    if V2:
        spread_config = {name: scenario[name]["config"].get("financing_rating_slope", 0.0) == expected[name]
                         and scenario[name]["config"]["financing_spread"] == 0.05 for name in names}
    else:
        spread_config = {name: scenario[name]["config"]["financing_spread"] == expected[name] for name in names}
    rate_ok = Counter()
    for (name, seed), data in loaded.items():
        cfg = scenario[name]["config"]
        f = cfg["financing_exposure_fraction"]
        day0 = data["financing_rate"][0] - cfg["risk_free_rate"]
        if V2:
            level = np.isclose(day0.mean(), cfg["financing_spread"] * f / (1 + f), rtol=1e-12, atol=0)
            spread = np.ptp(day0) > 1e-12
            rate_ok[bool(level and spread == (expected[name] > 0))] += 1
        else:
            rate_ok[bool(np.allclose(day0, expected[name] * f / (1 + f), rtol=1e-12, atol=0))] += 1
    by_point: dict = {}
    for name in names:
        label = next((key for key, p in PREFIXES.items() if name.startswith(p)), STAR)
        row = by_point.setdefault(label, {"value": expected[name], "planned": 0, "complete": 0, "failed": 0,
                                          "errors": Counter(), "rate_min": None, "rate_max": None, "price_cap_hits": 0})
        for seed in seeds:
            row["planned"] += 1
            record = rows.get((name, seed))
            if (name, seed) in complete:
                row["complete"] += 1
                data = loaded[(name, seed)]
                low, high = float(data["financing_rate"].min()), float(data["financing_rate"].max())
                row["rate_min"] = low if row["rate_min"] is None else min(row["rate_min"], low)
                row["rate_max"] = high if row["rate_max"] is None else max(row["rate_max"], high)
                row["price_cap_hits"] += int(np.sum(data["market_price_cap_hits"]))
            else:
                row["failed"] += 1
                message = (record or {}).get("error") or next(
                    (a.get("error") for a in reversed((record or {}).get("attempts", [])) if a.get("error")), "missing")
                row["errors"][str(message).split(" on day ")[0]] += 1
    for row in by_point.values():
        row["errors"] = dict(row["errors"])
        row["completion_rate"] = row["complete"] / row["planned"]
    pairing = {}
    for block, members in blocks(names).items():
        compared = differing = 0
        for seed in seeds:
            series = [loaded[(name, seed)] for name in members if (name, seed) in loaded]
            series = [(d["inflation"][:SWITCH], d["financing_rate"][:SWITCH], d["actions"][:SWITCH // PERIOD + 1])
                      for d in series]
            for other in series[1:]:
                compared += 1
                differing += not all(np.array_equal(x, y) for x, y in zip(series[0], other))
        pairing[block] = {"scenarios": len(members), "spread": expected[members[0]], "pairs_compared": compared,
                          "pairs_differing_before_switch": differing}
    activity = {}
    for name in names:
        cells = [loaded[(name, seed)] for seed in seeds if (name, seed) in loaded]
        row = {"has_audit_probability": sum("audit_probability" in d for d in cells),
               "has_audit_flags": sum("audit_flags" in d for d in cells),
               "has_assurer_assignment": sum("assurer_assignment" in d for d in cells)}
        if row["has_audit_flags"]:
            row["flags_in_window"] = int(sum(d["audit_flags"][WINDOW].sum() for d in cells if "audit_flags" in d))
        if row["has_assurer_assignment"]:
            row["assigned_firm_periods_in_window"] = int(sum(
                (d["assurer_assignment"][WINDOW] >= 0).sum() for d in cells if "assurer_assignment" in d))
        activity[name] = row
    expected_active = {"has_audit_probability": set(AUDITED), "has_assurer_assignment": set(ASSURED)}
    out = {"experiment_id": frozen["experiment_id"], "purpose": spec["purpose"], POINTS_KEY: spreads,
           "completion_by_point": by_point,
           "planned": summary["planned_trials"], "complete": summary["complete_trials"],
           "attempts_by_status": dict(sorted(attempts.items())),
           "failing_core_validation": summary["trials_failing_core_validation"], "core_validation_failures": invalid,
           "metric_windows": windows, "policy_window_decisions_per_trial": decisions,
           "spread_in_frozen_configs_as_planned": all(spread_config.values()),
           "day0_rate_matches_spread": dict(rate_ok), "pre_switch_pairing": pairing, "policy_activity": activity,
           "activity_where_expected": all(
               (activity[name][field] == len(seeds)) == (base_name(name) in members)
               for field, members in expected_active.items() for name in names)}
    (HERE / f"{args.run}_pipeline.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "policy_activity"}, indent=2))


if __name__ == "__main__":
    main()
