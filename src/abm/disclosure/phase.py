"""Frozen parameter-response classification and plots from archived trials.

G and W use complete terminal 252-state-day windows. Threshold classes are
descriptive; development proportions are never confirmatory dominant states.
No market simulation or market-parameter estimation is performed here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from math import ceil, isfinite, sqrt
from pathlib import Path
from statistics import NormalDist

from ..manifest import canonical_json_bytes, sha256_file


RULES = {
    "version": 1,
    "taste_medians": [0, .01, .02, .04],
    "initial_impact_probabilities": [0, .2, .5, .8],
    "g_thresholds": [.125, .25, .5],
    "w_thresholds": [.125, .25, .5],
    "terminal_observations": 252,
    "dominant_fraction": .8,
    "confirmatory_min_seeds": 30,
    "G_metric": "terminal_year.change_from_initial_sd",
    "W_metric": "terminal_year.inflation_mean_sd",
    "G_reference": "Last 252 state-day mean true level minus day-0 mean; initial SD.",
    "W_reference": "Last 252 state-day mean inflation; initial SD, without transparency weighting.",
    "classes": {
        "real_improvement": "G >= g_threshold and W < w_threshold",
        "inflation_trap": "G < g_threshold and W >= w_threshold",
        "low_change_low_inflation": "G < g_threshold and W < w_threshold",
        "mixed": "G >= g_threshold and W >= w_threshold",
    },
    "proportion_intervals": "Pointwise Wilson score intervals across valid seeds; not multiplicity-adjusted.",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _classify(g, w, g_threshold, w_threshold):
    if g >= g_threshold:
        return "mixed" if w >= w_threshold else "real_improvement"
    return "inflation_trap" if w >= w_threshold else "low_change_low_inflation"


def _wilson(count, observations, confidence):
    if not observations:
        return None
    p = count / observations
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    divisor = 1 + z * z / observations
    centre = (p + z * z / (2 * observations)) / divisor
    radius = z * sqrt(p * (1 - p) / observations + z * z / (4 * observations**2)) / divisor
    return [max(0., centre - radius), min(1., centre + radius)]


def analyze_phase(frozen: dict, summary: dict) -> dict:
    """Classify each archived seed, retaining missing trials and all nine cutoffs."""
    spec = frozen["spec"]
    if spec.get("phase_analysis") != RULES:
        raise ValueError("phase rules must match the fixed version-1 classification design")
    if summary["experiment_id"] != frozen["experiment_id"] or summary["purpose"] != spec["purpose"]:
        raise ValueError("summary does not belong to the frozen experiment")
    seeds = spec["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("phase seeds must be distinct integers")
    confidence = spec["analysis"]["confidence_level"]
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be between zero and one")
    scenarios = frozen["scenarios"]
    expected = {(taste, share) for taste in RULES["taste_medians"]
                for share in RULES["initial_impact_probabilities"]}
    points = {(row["config"]["taste_median"], row["config"]["initial_impact_share"]) for row in scenarios}
    if points != expected or len(scenarios) != len(expected) or len({row["name"] for row in scenarios}) != len(expected):
        raise ValueError("phase design must contain each fixed 4-by-4 grid point exactly once")
    reference = scenarios[0]
    common = {key: value for key, value in reference["config"].items()
              if key not in ("taste_median", "initial_impact_share")}
    for scenario in scenarios:
        config = {key: value for key, value in scenario["config"].items()
                  if key not in ("taste_median", "initial_impact_share")}
        if config != common or scenario["domain"] != reference["domain"] or scenario["policy"] != reference["policy"]:
            raise ValueError("phase scenarios may differ only in taste and initial investor probability")
        if scenario["policy"]["intervention"] is not None:
            raise ValueError("phase design requires a constant baseline policy")
    expected_trials = {(scenario["name"], seed) for scenario in scenarios for seed in seeds}
    by_trial = {}
    for row in summary["records"]:
        key = (row["scenario"], row["seed"])
        if key not in expected_trials or key in by_trial:
            raise ValueError("unexpected or duplicate phase trial")
        by_trial[key] = row
    estimates = {row["name"]: row["metrics"] for row in summary["scenarios"]}
    if len(estimates) != len(summary["scenarios"]):
        raise ValueError("duplicate archived scenario estimates")

    cells = []
    for scenario in scenarios:
        config = scenario["config"]
        observations = []
        core_failures = 0
        for seed in seeds:
            row = by_trial.get((scenario["name"], seed), {})
            terminal = row.get("metrics", {}).get("terminal_year", {})
            validation = row.get("validation", {})
            gates = [validation.get(gate, {}).get("status") for gate in ("mechanical", "stylized")]
            core_failures += int(row.get("status") == "complete" and gates != ["PASS", "PASS"])
            reason = None
            if row.get("status") != "complete":
                reason = "trial_" + row.get("status", "missing")
            elif gates[0] != "PASS":
                reason = "mechanical_validation_failed"
            elif terminal.get("complete_year") is not True or terminal.get("observations") != 252:
                reason = "incomplete_terminal_window"
            elif terminal.get("start_day") != config["trading_days"] - 251 or row["metrics"].get("window", {}).get("end_day") != config["trading_days"]:
                reason = "terminal_window_mismatch"
            g, w = terminal.get("change_from_initial_sd"), terminal.get("inflation_mean_sd")
            if reason is None and (not _finite(g) or not _finite(w) or w < 0):
                reason = "invalid_terminal_metrics"
            observations.append({"seed": seed, "status": "valid" if reason is None else "unclassified",
                                 "reason": reason, "G": g if reason is None else None,
                                 "W": w if reason is None else None})
        valid = [row for row in observations if row["status"] == "valid"]
        complete = len(valid) == len(seeds)
        status = ("incomplete" if not complete else "development_only" if spec["purpose"] != "confirmatory"
                  else "insufficient_seeds" if len(seeds) < RULES["confirmatory_min_seeds"]
                  else "core_validation_failed" if core_failures else "classified")
        classifications = []
        for g_threshold in RULES["g_thresholds"]:
            for w_threshold in RULES["w_thresholds"]:
                assignments = {str(row["seed"]): _classify(row["G"], row["W"], g_threshold, w_threshold)
                               for row in valid}
                counts = {category: sum(value == category for value in assignments.values())
                          for category in RULES["classes"]}
                largest = max(counts, key=counts.get)
                dominant = (largest if counts[largest] >= ceil(RULES["dominant_fraction"] * len(seeds))
                            else "cross_seed_uncertain") if status == "classified" else None
                classifications.append({"g_threshold": g_threshold, "w_threshold": w_threshold,
                                        "seed_classes": assignments, "counts": counts,
                                        "proportions": {key: count / len(valid) if valid else None for key, count in counts.items()},
                                        "intervals": {key: _wilson(count, len(valid), confidence) for key, count in counts.items()},
                                        "status": status, "dominant_state": dominant})
        continuous = {}
        for name, metric in (("G", RULES["G_metric"]), ("W", RULES["W_metric"]),
                             ("impact_share_at_burn_in", "impact.initial_share"),
                             ("impact_wealth_share_at_burn_in", "impact.initial_wealth_share")):
            saved = estimates.get(scenario["name"], {}).get(metric, {})
            available = complete and saved.get("status") == "complete" and saved.get("n") == len(seeds) and _finite(saved.get("mean"))
            continuous[name] = {"status": "complete" if available else "incomplete",
                                "mean": saved["mean"] if available else None,
                                "ci_low": saved.get("ci_low") if available and _finite(saved.get("ci_low")) else None,
                                "ci_high": saved.get("ci_high") if available and _finite(saved.get("ci_high")) else None,
                                "n": saved["n"] if available else len(valid)}
        cells.append({"scenario": scenario["name"], "taste_median": config["taste_median"],
                      "initial_impact_probability": config["initial_impact_share"],
                      "burn_in_day": config["burn_in_days"], "n_planned": len(seeds), "n_valid": len(valid),
                      "core_validation_failures": core_failures, "observations": observations,
                      "continuous": continuous, "classifications": classifications})
    result = {"schema_version": 1, "experiment_id": frozen["experiment_id"], "purpose": spec["purpose"],
              "claim_support": "pending", "rules": deepcopy(RULES), "confidence_level": confidence,
              "continuous_source": "Archived summary.scenarios estimates; no aggregation during rendering.",
              "classification_source": "Archived per-seed terminal_year metrics; proportions condition on n_valid.",
              "core_validation_failures": sum(cell["core_validation_failures"] for cell in cells), "cells": cells}
    canonical_json_bytes(result)
    return result


def _write_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_phase(destination: str | Path) -> Path:
    """Freeze a phase analysis once, then render only its archived numbers."""
    destination = Path(destination)
    if (destination / "phase_manifest.json").exists() or (destination / "phase_analysis.json").exists():
        return render_phase(destination)
    frozen = json.loads((destination / "experiment.json").read_text(encoding="utf-8"))
    identity = {key: value for key, value in frozen.items() if key != "experiment_id"}
    if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != frozen["experiment_id"]:
        raise ValueError("experiment manifest hash mismatch")
    summary_manifest = json.loads((destination / "summary_manifest.json").read_text(encoding="utf-8"))
    if summary_manifest["experiment_id"] != frozen["experiment_id"] or summary_manifest["summary_sha256"] != sha256_file(destination / "summary.json"):
        raise ValueError("experiment summary hash mismatch")
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    inputs = {name: sha256_file(destination / name) for name in ("experiment.json", "summary.json", "summary_manifest.json")}
    for row in summary["records"]:
        if row["status"] != "complete" or "metrics" in row:
            continue
        trial = (destination / row["path"]).resolve()
        trial.relative_to(destination.resolve())
        manifest_path = trial / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario = next(item for item in frozen["scenarios"] if item["name"] == row["scenario"])
        identity = {"schema_version": 1, "config": {**scenario["config"], "seed": row["seed"]},
                    "domain": scenario["domain"], "policy": scenario["policy"],
                    "source_hashes": frozen["source_hashes"], "python_version": frozen["python_version"],
                    "numpy_version": frozen["numpy_version"]}
        if manifest["run_id"] != hashlib.sha256(canonical_json_bytes(identity)).hexdigest():
            raise ValueError("archived trial identity does not match frozen experiment")
        if manifest["status"] != "complete" or manifest["artifacts"]["metrics.json"] != sha256_file(trial / "metrics.json"):
            raise ValueError("archived trial metric hash mismatch")
        row["metrics"] = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
        for path in (manifest_path, trial / "metrics.json"):
            inputs[path.relative_to(destination.resolve()).as_posix()] = sha256_file(path)
    analysis = analyze_phase(frozen, summary)
    if any(sha256_file(destination / name) != digest for name, digest in inputs.items()):
        raise ValueError("phase input changed during analysis")
    _write_json(destination / "phase_analysis.json", analysis)
    _write_json(destination / "phase_manifest.json", {
        "experiment_id": frozen["experiment_id"], "inputs_sha256": inputs,
        "rules_sha256": hashlib.sha256(canonical_json_bytes(analysis["rules"])).hexdigest(),
        "analysis_sha256": sha256_file(destination / "phase_analysis.json"),
        "analysis_source_sha256": sha256_file(Path(__file__)),
    })
    return render_phase(destination)


def render_phase(destination: str | Path) -> Path:
    """Re-render a saved phase analysis without reclassification or simulation."""
    import numpy as np
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(destination)
    manifest = json.loads((destination / "phase_manifest.json").read_text(encoding="utf-8"))
    for filename, digest in manifest["inputs_sha256"].items():
        if sha256_file(destination / filename) != digest:
            raise ValueError("phase input hash mismatch")
    if sha256_file(destination / "phase_analysis.json") != manifest["analysis_sha256"]:
        raise ValueError("phase analysis hash mismatch")
    analysis = json.loads((destination / "phase_analysis.json").read_text(encoding="utf-8"))
    rules = analysis["rules"]
    if manifest["experiment_id"] != analysis["experiment_id"] or hashlib.sha256(canonical_json_bytes(rules)).hexdigest() != manifest["rules_sha256"]:
        raise ValueError("phase classification identity mismatch")
    files = []
    with rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        for field, title, palette in (("G", "G: terminal real-level change from day 0", "RdBu"),
                                      ("W", "W: terminal report inflation", "cividis")):
            values = np.ma.masked_all((len(rules["initial_impact_probabilities"]), len(rules["taste_medians"])))
            for cell in analysis["cells"]:
                x = rules["taste_medians"].index(cell["taste_median"])
                y = rules["initial_impact_probabilities"].index(cell["initial_impact_probability"])
                if cell["continuous"][field]["mean"] is not None:
                    values[y, x] = cell["continuous"][field]["mean"]
            extent = max(float(np.max(np.abs(values.compressed()))) if values.count() else 0., .01)
            figure = Figure(figsize=(6.5, 5.1))
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            cmap = colormaps[palette].with_extremes(bad="#eeeeee")
            image = axis.imshow(values, origin="lower", interpolation="none", aspect="auto", cmap=cmap,
                                vmin=-extent if field == "G" else 0, vmax=extent)
            for cell in analysis["cells"]:
                x = rules["taste_medians"].index(cell["taste_median"])
                y = rules["initial_impact_probabilities"].index(cell["initial_impact_probability"])
                mean = cell["continuous"][field]["mean"]
                label = "missing" if mean is None else f"{mean:.3f}"
                color = ("white" if abs(mean) > .55 * extent else "black") if field == "G" and mean is not None else (
                    "white" if mean is not None and mean < .5 * extent else "black")
                axis.text(x, y, f"{label}\nn={cell['n_valid']}/{cell['n_planned']}", ha="center", va="center", color=color)
            axis.set_xticks(range(len(rules["taste_medians"])), rules["taste_medians"])
            axis.set_yticks(range(len(rules["initial_impact_probabilities"])), rules["initial_impact_probabilities"])
            axis.set_xlabel("Taste median")
            axis.set_ylabel("Initial impact-investor probability")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, label="Initial SD; archived seed mean")
            figure.text(.02, .02,
                        f"{analysis['purpose']} | claims: pending | core-validation failures: {analysis['core_validation_failures']}\n"
                        "Baseline-policy parameter response; no interpolation or measured phase boundary.\n"
                        f"Source: phase_analysis.json, SHA-256 {manifest['analysis_sha256'][:12]}; full 252-state-day windows.",
                        fontsize=8, va="bottom")
            figure.subplots_adjust(left=.12, right=.88, bottom=.25, top=.90)
            for extension in ("png", "pdf"):
                filename = f"phase_{field}.{extension}"
                figure.savefig(destination / filename, dpi=300, bbox_inches="tight")
                files.append(filename)
            figure.clear()
    lines = ["# Disclosure parameter-response analysis", "", f"Purpose: {analysis['purpose']}; claim support: pending.",
             f"Experiment ID: `{analysis['experiment_id']}`", f"Analysis SHA-256: `{manifest['analysis_sha256']}`", "",
             "G is the last 252 state-day mean true level minus day-0 mean. W is mean inflation over the same days, without transparency weighting. Both use initial SD.",
             "The axes are taste median and initialization probability, not realized investor or wealth shares. This finite baseline-policy grid does not identify a sharp phase boundary or policy-induced boundary movement.",
             "Development proportions are descriptive only. Confirmatory dominant labels require at least 30 planned seeds, all complete terminal windows and passing core validation; 80% in one class, otherwise cross_seed_uncertain.",
             "Missing or mechanically invalid trials are unclassified. Proportions use n_valid and do not impute missing seeds; incomplete cells have no dominant label or plotted continuous mean.",
             f"Class intervals are pointwise {100 * analysis['confidence_level']:g}% Wilson score intervals, unadjusted across classes, cells and thresholds. They exclude parameter and model uncertainty. [Method](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).", "",
             f"Core-validation failures: {analysis['core_validation_failures']}. Stylized-validation failures remain visible in descriptive results and block confirmatory dominance.", "",
             "| Scenario | Taste median | Initial probability | G | W | Valid/planned | II share at burn-in | II wealth share at burn-in |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    def _display(row: dict, name: str) -> str:
        mean = row["continuous"][name]["mean"]
        return "missing" if mean is None else f"{mean:.6g}"

    for cell in analysis["cells"]:
        lines.append(
            f"| {cell['scenario']} | {cell['taste_median']:g} | {cell['initial_impact_probability']:g} "
            f"| {_display(cell, 'G')} | {_display(cell, 'W')} | {cell['n_valid']}/{cell['n_planned']} "
            f"| {_display(cell, 'impact_share_at_burn_in')} (day {cell['burn_in_day']}) "
            f"| {_display(cell, 'impact_wealth_share_at_burn_in')} |"
        )
    for filename in files:
        lines.extend(["", f"![{filename}]({filename})" if filename.endswith(".png") else f"[PDF: {filename}]({filename})"])
    lines.extend(["", "## All threshold sensitivities", "", "Thresholds are display assumptions, not empirically validated success criteria.", "",
                  "| Scenario | G cutoff | W cutoff | Real improvement | Inflation trap | Low change/low inflation | Mixed | Status | Dominant |",
                  "|---|---:|---:|---|---|---|---|---|---|"])
    for cell in analysis["cells"]:
        for row in cell["classifications"]:
            entries = []
            for category in rules["classes"]:
                p, interval = row["proportions"][category], row["intervals"][category]
                entries.append("missing" if p is None else f"{row['counts'][category]}/{cell['n_valid']} = {p:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]")
            lines.append(f"| {cell['scenario']} | {row['g_threshold']:g} | {row['w_threshold']:g} | " + " | ".join(entries) + f" | {row['status']} | {row['dominant_state'] or 'pending'} |")
    lines.extend(["", "## Unclassified observations", ""])
    for cell in analysis["cells"]:
        for row in cell["observations"]:
            if row["status"] != "valid":
                lines.append(f"- {cell['scenario']}, seed {row['seed']}: {row['reason']}")
    report = destination / "phase_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(destination / "phase_charts_manifest.json", {
        "experiment_id": analysis["experiment_id"], "analysis_sha256": manifest["analysis_sha256"],
        "renderer_source_sha256": sha256_file(Path(__file__)),
        "artifacts": {name: sha256_file(destination / name) for name in [*files, report.name]},
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args(argv)
    report = render_phase(arguments.experiment) if arguments.render_only else archive_phase(arguments.experiment)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
