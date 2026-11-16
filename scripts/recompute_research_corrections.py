"""Recompute the 2026-11-16 audit corrections without changing archived evidence.

Public H3 point differences:
    python scripts/recompute_research_corrections.py h3

Optional licensed/local archives (read only; no simulation is launched):
    python scripts/recompute_research_corrections.py h3 --archive /path/to/h3_v1
    python scripts/recompute_research_corrections.py profiles --stage h2 --grid /path/to/grid_v1
    python scripts/recompute_research_corrections.py profiles --stage i3 --grid /path/to/grid_v3

By default JSON goes to stdout. --output must name a new file outside reports/.
The profile command replaces only auxiliary intervals, using the original point
estimates and covariance convention; it does not rerun or silently amend fit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abm.research_inference import distance_profile_intervals, paired_mean_difference  # noqa: E402

H2 = ROOT / "reports/disclosure_persistence_estimation_20261026"
I3 = ROOT / "reports/disclosure_stage_i_20261027"
H3 = ROOT / "reports/disclosure_stage_h3_20261027"
METRIC = "true_change.equal_weight_sd"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recompute_profiles(stage: str, archive: Path, forms: list[str]) -> dict:
    """Read existing sampled signals; reconstruct the original interpolation grid."""
    frozen_path = archive / "experiment.json"
    frozen = read_json(frozen_path)
    recorded_path = H2 / "estimation_v1.json" if stage == "h2" else I3 / "estimation_v3.json"
    recorded = read_json(recorded_path)
    if frozen["experiment_id"] != recorded["grid_experiment_id"]:
        raise ValueError("grid archive does not match the recorded estimation experiment")
    seeds = frozen["spec"]["seeds"]
    if seeds != recorded["grid_seeds"]:
        raise ValueError("grid archive seeds differ from the recorded estimation")

    legacy = load_module("audit_h2_estimator", H2 / "estimate.py")
    sources = [H2 / "estimate.py", H2 / "data_moments.py", H2 / "build_spec.py"]
    if stage == "i3":
        grid_spec = load_module("audit_i3_grid", I3 / "build_grid_spec_v3.py")
        legacy.FORMS = {
            "U": (("delta",), (grid_spec.DECAYS,), lambda d: f"u_d{grid_spec.label(d)}"),
            "R": (("kappa",), (grid_spec.KAPPAS,), lambda k: f"r_k{grid_spec.label(k)}"),
            "Q": (("kappa_T", "s_P"), (grid_spec.KAPPAS_T, grid_spec.SHARES),
                  lambda k, s: f"q_k{grid_spec.label(k)}_s{grid_spec.label(s)}"),
        }
        sources.append(I3 / "build_grid_spec_v3.py")
    legacy.FORMS = {name: legacy.FORMS[name] for name in forms}
    target_path = H2 / "data_moments.json"
    data = read_json(target_path)["targets"][recorded["target"]]
    if data["lags"] != list(legacy.LAGS):
        raise ValueError("target moment lags differ from the historical estimator")
    target, data_cov = np.array(data["moments"]), np.array(data["bootstrap_cov"])
    grid = legacy.grid_moments(archive, seeds)
    corrected = {}
    for name in forms:
        form = legacy.Form(name, grid[name])
        row = recorded["forms"][name]
        theta = np.array([row["estimate"][key] for key in (*form.labels, "sigma_m")])
        model_moments = form(theta)
        if not np.allclose(model_moments, row["moments_model"], rtol=1e-8, atol=1e-10):
            raise ValueError(f"{name}: reconstructed moments differ from the archived estimation")
        covariance = data_cov + form.seed_covariance(theta)
        nodes = np.array(np.meshgrid(*form.axes, indexing="ij")).reshape(len(form.axes), -1).T
        corrected[name] = {
            "historical_intervals_do_not_use": row["profile_intervals"],
            "corrected": distance_profile_intervals(form, theta, target, covariance, bounds=form.bounds,
                                                     names=form.labels, start_points=nodes),
        }
    return {"correction": "2026-11-16 distance-profile reference minimum", "stage": stage,
            "grid_experiment_id": frozen["experiment_id"], "status": "recomputed_from_existing_archive",
            "scope": "auxiliary profiles only; historical point estimates and fit tests are unchanged",
            "covariance": "original data covariance plus grid-seed covariance at the original point estimate",
            "sources": [source(p) for p in [recorded_path, target_path, frozen_path, *sources,
                                             ROOT / "src/abm/research_inference.py"]],
            "forms": corrected}


def h3_correction(analysis_path: Path, archive: Path | None = None) -> dict:
    """Use B for equal-budget contrasts; means alone never produce paired CIs."""
    analysis = read_json(analysis_path)
    if analysis["complete_trials"] != analysis["planned_trials"]:
        raise ValueError("the public analysis contains incomplete trials")
    names = ("B", "V", "J", "A2")
    describe = analysis["describe"]
    counts = {int(describe[name]["complete"]) for name in names}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError("scenario means must cover the same nontrivial seed count")
    means = {name: float(describe[name]["true_change"]) for name in names}
    if not all(np.isfinite(value) for value in means.values()):
        raise ValueError("scenario means must be finite")
    contrasts = {f"{name}_minus_B": {"mean": means[name] - means["B"], "n": min(counts),
                                    "ci_low": None, "ci_high": None,
                                    "interval_status": "not_computable_from_scenario_means"}
                 for name in ("V", "J")}
    sources = [source(analysis_path)]
    status = "public_point_differences_only"
    if archive is not None:
        frozen_path, summary_path = archive / "experiment.json", archive / "summary.json"
        frozen, summary = read_json(frozen_path), read_json(summary_path)
        if not (frozen["experiment_id"] == summary["experiment_id"] == analysis["experiment_id"]):
            raise ValueError("H3 archive, summary and public analysis experiment IDs must match")
        seeds = frozen["spec"]["seeds"]
        if len(seeds) != min(counts) or len(set(seeds)) != len(seeds):
            raise ValueError("H3 archive seed set does not match the public analysis count")
        values = {name: {} for name in ("B", "V", "J")}
        for record in summary["records"]:
            name, seed = record["scenario"], record["seed"]
            if name not in values:
                continue
            if seed not in seeds or seed in values[name]:
                raise ValueError(f"unexpected or duplicate H3 record: {name}, seed {seed}")
            if record["status"] != "complete":
                raise ValueError(f"incomplete H3 record: {name}, seed {seed}")
            values[name][seed] = float(record["metrics"]["true_change"]["equal_weight_sd"])
        for name, by_seed in values.items():
            if set(by_seed) != set(seeds) or not np.isfinite(list(by_seed.values())).all():
                raise ValueError(f"incomplete or nonfinite H3 seed values for {name}")
            if not np.isclose(np.mean(list(by_seed.values())), means[name], rtol=1e-10, atol=1e-12):
                raise ValueError(f"H3 seed values do not reproduce the public {name} mean")
        contrasts = {f"{name}_minus_B": {
            **paired_mean_difference(values[name], values["B"], seeds), "interval_status": "recomputed_from_paired_seeds"}
            for name in ("V", "J")}
        sources.extend([source(frozen_path), source(summary_path)])
        status = "paired_intervals_recomputed_from_archive"
    return {"correction": "2026-11-16 H3 equal-budget comparator", "status": status,
            "experiment_id": analysis["experiment_id"], "metric": METRIC, "sources": sources,
            "scenario_means": means, "equal_budget_contrasts": contrasts,
            "audit_design": {"B_V_J_probability": 0.01, "A2_probability": 0.02,
                             "source": "paper/main.tex, targeted-audit design; configs/disclosure_stage_h3_v1.json"},
            "different_budget_comparisons_not_equal_budget": {
                f"{name}_minus_A2": means[name] - means["A2"] for name in ("V", "J")},
            "scope": "descriptive correction; no A2 confidence intervals are reused and no primary verdict is replaced"}


def write_new_output(payload: dict, output: Path | None, protected_roots: tuple[Path, ...] = ()) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
        return
    resolved = output.resolve()
    if resolved.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("correction output must be outside the historical reports directory")
    if any(resolved.is_relative_to(path.resolve()) for path in protected_roots):
        raise ValueError("correction output must be outside input archives")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects files outside reports/ from accidental replacement.
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    h3 = commands.add_parser("h3", help="correct equal-budget contrasts; public point differences by default")
    h3.add_argument("--archive", type=Path, help="optional original h3_v1 archive, read only")
    profiles = commands.add_parser("profiles", help="recompute auxiliary profiles from an existing sampled grid")
    profiles.add_argument("--stage", choices=("h2", "i3"), required=True)
    profiles.add_argument("--grid", type=Path, required=True)
    profiles.add_argument("--forms", choices=("U", "R", "Q"), nargs="+", default=["U", "R", "Q"])
    for command in (h3, profiles):
        command.add_argument("--output", type=Path, help="new JSON file outside reports; defaults to stdout")
    args = parser.parse_args()
    try:
        payload = h3_correction(H3 / "h3_v1_analysis.json", args.archive) if args.command == "h3" else recompute_profiles(
            args.stage, args.grid, list(dict.fromkeys(args.forms)))
        archive = args.archive if args.command == "h3" else args.grid
        write_new_output(payload, args.output, (archive,) if archive is not None else ())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
