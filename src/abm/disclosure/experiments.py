"""Frozen, resumable disclosure experiments with paired seed comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..manifest import canonical_json_bytes, sha256_file
from ..platform_tuning import disable_power_throttling
from .config import MarketConfig
from .domain import DomainSpec
from .policy import PolicySchedule
from .report import render_charts
from .runner import render_run, run_and_archive, run_identity, source_hashes, write_json


def freeze_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    # "estimation" runs simulated-moment grids (research contract section 25), not hypothesis tests.
    if spec["purpose"] not in ("development", "estimation", "confirmatory"):
        raise ValueError("purpose must be development, estimation or confirmatory")
    seeds = spec["seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("at least two distinct nonnegative seeds are required")
    if spec["purpose"] == "confirmatory" and len(seeds) < 30:
        raise ValueError("confirmatory experiments require at least 30 seeds")
    if spec.get("archive", "full") not in ("full", "compact", "sampled"):
        raise ValueError("archive must be 'full', 'compact' or 'sampled'")
    raw_domain = json.loads((path.parent / spec["domain"]).read_text(encoding="utf-8"))
    domain = DomainSpec.from_dict({key: value for key, value in raw_domain.items() if not key.startswith("_")})
    base_config = MarketConfig.from_dict(spec["market"])
    scenarios = []
    for scenario in spec["scenarios"]:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", scenario["name"]):
            raise ValueError("scenario names must be simple directory names")
        if "seed" in scenario.get("market", {}):
            raise ValueError("scenario overrides cannot change paired seeds")
        config = MarketConfig.from_dict({**base_config.to_dict(), **scenario.get("market", {})})
        scenario_domain = DomainSpec.from_dict({**domain.to_dict(), **scenario.get("domain", {})})
        policy = PolicySchedule.constant(scenario_domain)
        if scenario.get("policy"):
            policy_spec = scenario["policy"]
            policy = PolicySchedule(replace(policy.baseline, **policy_spec.get("baseline", {})))
            if set(policy_spec) - {"baseline"}:
                policy = policy.with_intervention(policy_spec["start_day"], **policy_spec["changes"])
        scenarios.append({"name": scenario["name"], "config": config.to_dict(),
                          "domain": scenario_domain.to_dict(), "policy": policy.to_dict()})
    names = [scenario["name"] for scenario in scenarios]
    if not names or len(names) != len(set(names)):
        raise ValueError("scenario names must be nonempty and unique")
    comparisons = spec["comparisons"]
    comparison_names = [comparison["name"] for comparison in comparisons]
    if len(comparison_names) != len(set(comparison_names)):
        raise ValueError("comparison names must be unique")
    for comparison in comparisons:
        weights = comparison["weights"]
        if not weights or set(weights) - set(names) or not all(np.isfinite(weight) for weight in weights.values()):
            raise ValueError("comparison weights must refer to defined scenarios and be finite")
        windows = {(scenario["config"]["burn_in_days"], scenario["config"]["trading_days"])
                   for scenario in scenarios if scenario["name"] in weights}
        if len(windows) != 1:
            raise ValueError("a comparison requires identical metric windows across its scenarios")
    analysis = spec["analysis"]
    if not 0 < analysis["confidence_level"] < 1:
        raise ValueError("confidence_level must be between zero and one")
    if analysis["primary_metric"] not in analysis["metrics"]:
        raise ValueError("primary_metric must appear in metrics")
    for comparison in comparisons:
        if comparison.get("primary_metric", analysis["primary_metric"]) not in analysis["metrics"]:
            raise ValueError("comparison primary_metric must appear in metrics")
    identity = run_identity(base_config, domain, PolicySchedule.constant(domain))
    payload = {"schema_version": 1, "spec": spec, "raw_domain": raw_domain, "scenarios": scenarios,
               "source_hashes": identity["source_hashes"], "python_version": identity["python_version"],
               "numpy_version": identity["numpy_version"], "scipy_version": version("scipy")}
    return {**payload, "experiment_id": hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}


def _inputs(scenario: dict, seed: int):
    from .policy import PolicyRegime

    config = MarketConfig.from_dict({**scenario["config"], "seed": seed})
    domain = DomainSpec.from_dict(scenario["domain"])
    policy = scenario["policy"]
    schedule = PolicySchedule(PolicyRegime(**policy["baseline"]),
                              PolicyRegime(**policy["intervention"]) if policy["intervention"] else None,
                              policy["start_day"])
    return config, domain, schedule


def _run_trial(scenario: dict, seed: int, destination: str, purpose: str, expected_id: str,
               archive: str = "full") -> dict:
    config, domain, policy = _inputs(scenario, seed)
    if run_identity(config, domain, policy)["run_id"] != expected_id:
        raise RuntimeError("source or environment changed before a trial started")
    try:
        run_and_archive(config, domain, destination, policy, purpose=purpose, archive=archive)
        return {"status": "complete", "path": destination}
    except Exception as error:
        failed_path = Path(destination)
        failed_path.mkdir(parents=True, exist_ok=True)
        if not (failed_path / "manifest.json").exists():
            write_json(failed_path / "manifest.json", {
                "run_id": expected_id, "status": "failed", "error_type": type(error).__name__, "error": str(error),
            })
        return {"status": "failed", "path": destination,
                "error_type": type(error).__name__, "error": str(error)}


def _expected_id(frozen: dict, scenario: dict, seed: int) -> str:
    identity = {"schema_version": 1, "config": {**scenario["config"], "seed": seed}, "domain": scenario["domain"],
                "policy": scenario["policy"], "source_hashes": frozen["source_hashes"],
                "python_version": frozen["python_version"], "numpy_version": frozen["numpy_version"]}
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _metric(metrics: dict, key: str):
    value = metrics
    for part in key.split("."):
        value = value[part]
    return value


def paired_statistics(values: list[float], confidence_level: float) -> dict:
    """Two-sided paired-t inference; seeds, not firms, are observations."""
    from scipy.stats import t as student_t

    differences = np.asarray(values, dtype=float)
    if differences.size < 2 or not np.isfinite(differences).all():
        raise ValueError("paired inference requires at least two finite seed contrasts")
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(differences.size))
    half_width = float(student_t.ppf((1 + confidence_level) / 2, differences.size - 1)) * standard_error
    p_value = float(2 * student_t.sf(abs(mean / standard_error), differences.size - 1)) if standard_error else (
        1.0 if mean == 0 else 0.0
    )
    return {"mean": mean, "standard_error": standard_error, "ci_low": mean - half_width,
            "ci_high": mean + half_width, "p_value": p_value, "n": len(values), "paired_values": values}


def analyze(frozen: dict, records: list[dict]) -> dict:
    spec = frozen["spec"]
    seeds, analysis = spec["seeds"], spec["analysis"]
    by_trial = {(row["scenario"], row["seed"]): row for row in records}
    configs = {scenario["name"]: scenario["config"] for scenario in frozen["scenarios"]}
    comparisons = []
    for comparison in spec["comparisons"]:
        metrics = {}
        for metric in analysis["metrics"]:
            values = []
            missing = []
            for seed in seeds:
                value = 0.0
                for scenario, weight in comparison["weights"].items():
                    row = by_trial.get((scenario, seed))
                    observed = _metric(row["metrics"], metric) if row and row["status"] == "complete" else None
                    if observed is None or not np.isfinite(observed):
                        missing.append(seed)
                        break
                    value += weight * observed
                else:
                    values.append(float(value))
            metrics[metric] = ({"status": "incomplete", "missing_seeds": missing,
                                "n_available": len(values), "n_planned": len(seeds)} if missing else
                               {"status": "complete", **paired_statistics(values, analysis["confidence_level"])})
            config = configs[next(iter(comparison["weights"]))]
            metrics[metric]["window"] = {
                "start_day": 0 if metric.startswith("full_period_change.") else config["burn_in_days"],
                "end_day": config["trading_days"],
            }
        validation_pass = all(
            by_trial.get((scenario, seed), {}).get("status") == "complete"
            and all(by_trial[(scenario, seed)]["validation"][gate]["status"] == "PASS"
                    for gate in ("mechanical", "stylized"))
            for scenario in comparison["weights"] for seed in seeds
        )
        comparisons.append({**comparison, "metrics": metrics, "core_validation_pass": validation_pass})
    primary = [row["metrics"][row.get("primary_metric", analysis["primary_metric"])]
               for row in comparisons if row.get("primary", False)]
    if primary and all(row["status"] == "complete" for row in primary):
        adjusted = 0.0
        for rank, row in enumerate(sorted(primary, key=lambda row: row["p_value"])):
            adjusted = max(adjusted, min(1.0, (len(primary) - rank) * row["p_value"]))
            row["holm_p_value"] = adjusted
    else:
        for row in primary:
            row["holm_p_value"] = None
    expected = len(spec["scenarios"]) * len(seeds)
    complete = sum(row["status"] == "complete" for row in records)
    invalid = sum(row["status"] == "complete" and (
        row["validation"]["mechanical"]["status"] != "PASS"
        or row["validation"]["stylized"]["status"] != "PASS"
    ) for row in records)
    scenarios = []
    for scenario in frozen["scenarios"]:
        scenario_metrics = {}
        for metric in analysis["metrics"]:
            values = []
            for seed in seeds:
                row = by_trial.get((scenario["name"], seed))
                value = _metric(row["metrics"], metric) if row and row["status"] == "complete" else None
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            if len(values) == len(seeds):
                statistics = paired_statistics(values, analysis["confidence_level"])
                scenario_metrics[metric] = {"status": "complete", **{
                    key: statistics[key] for key in ("mean", "ci_low", "ci_high", "n")}}
            else:
                scenario_metrics[metric] = {"status": "incomplete", "n_available": len(values), "n_planned": len(seeds)}
        scenarios.append({"name": scenario["name"], "metrics": scenario_metrics})
    return {"experiment_id": frozen["experiment_id"], "purpose": spec["purpose"],
            "status": "incomplete" if complete != expected else "complete",
            "claim_support": "pending", "planned_trials": expected, "complete_trials": complete,
            "trials_failing_core_validation": invalid, "comparisons": comparisons, "scenarios": scenarios, "records": records,
            "inference": "Paired t intervals across independent seeds; primary contrasts use Holm adjustment. "
                         "Intervals exclude parameter and model uncertainty. Failed seeds are not dropped. "
                         "Pairing shares initial seeds and exogenous streams, not necessarily learning draws."}


def collect_records(destination: Path, frozen: dict) -> list[dict]:
    records = []
    for scenario in frozen["scenarios"]:
        for seed in frozen["spec"]["seeds"]:
            attempts = sorted((destination / scenario["name"] / f"seed_{seed}").glob("attempt_*"))
            history = []
            row = {"scenario": scenario["name"], "seed": seed, "status": "missing"}
            for attempt in attempts:
                path = attempt / "manifest.json"
                if not path.exists():
                    history.append({"path": attempt.relative_to(destination).as_posix(), "status": "interrupted"})
                    continue
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest["run_id"] != _expected_id(frozen, scenario, seed):
                    raise ValueError("trial identity does not match frozen experiment")
                history.append({"path": attempt.relative_to(destination).as_posix(), "status": manifest["status"],
                                "error": manifest.get("error")})
                if row["status"] != "complete":
                    row.update({"status": manifest["status"], "path": attempt.relative_to(destination).as_posix()})
                if manifest["status"] == "complete":
                    render_run(attempt)
                    if row.get("result_fingerprint") not in (None, manifest["result_fingerprint"]):
                        raise ValueError("completed attempts disagree for the same frozen trial")
                    row.update({"status": "complete", "path": attempt.relative_to(destination).as_posix(),
                                "result_fingerprint": manifest["result_fingerprint"],
                                "metrics": json.loads((attempt / "metrics.json").read_text(encoding="utf-8")),
                                "validation": json.loads((attempt / "validation.json").read_text(encoding="utf-8"))})
            records.append({**row, "attempts": history})
    return records


def render_experiment(destination: str | Path) -> Path:
    destination = Path(destination)
    frozen = json.loads((destination / "experiment.json").read_text(encoding="utf-8"))
    identity = {key: value for key, value in frozen.items() if key != "experiment_id"}
    if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != frozen["experiment_id"]:
        raise ValueError("experiment manifest hash mismatch")
    summary_manifest = json.loads((destination / "summary_manifest.json").read_text(encoding="utf-8"))
    if summary_manifest["experiment_id"] != frozen["experiment_id"] or (
        sha256_file(destination / "summary.json") != summary_manifest["summary_sha256"]
    ):
        raise ValueError("experiment summary hash mismatch")
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    if summary["experiment_id"] != frozen["experiment_id"] or summary["records"] != collect_records(destination, frozen):
        raise ValueError("saved inference does not match the current trial archive")
    lines = ["# Disclosure experiment", "", f"Purpose: {summary['purpose']}",
             f"Experiment ID: `{summary['experiment_id']}`", f"Completed trials: {summary['complete_trials']}/{summary['planned_trials']}",
             f"Trials failing core validation: {summary['trials_failing_core_validation']}", "",
             "Claim support: pending. These results do not establish empirical calibration, real financing, or social policy efficiency.",
             "", summary["inference"], "", "| Comparison | Metric | Window | Primary | Mean | Unadjusted interval | Holm p | n |",
             "|---|---|---|---|---:|---|---:|---:|"]
    for comparison in summary["comparisons"]:
        for metric, result in comparison["metrics"].items():
            primary = comparison.get("primary", False) and metric == comparison.get(
                "primary_metric", frozen["spec"]["analysis"]["primary_metric"]
            )
            holm = result.get("holm_p_value")
            holm_text = f"{holm:.6g}" if holm is not None else "—"
            window = f"{result['window']['start_day']}–{result['window']['end_day']}"
            if result["status"] == "complete":
                lines.append(f"| {comparison['name']} | {metric} | {window} | {primary} | {result['mean']:.6g} | "
                             f"[{result['ci_low']:.6g}, {result['ci_high']:.6g}] | {holm_text} | {result['n']} |")
            else:
                lines.append(f"| {comparison['name']} | {metric} | {window} | {primary} | incomplete | "
                             f"missing {result['missing_seeds']} | — | {result['n_available']} |")
    figure_files = render_charts(destination, frozen, summary)
    lines.extend(["", "## Figures", ""])
    for filename in figure_files:
        if filename.endswith(".png"):
            lines.extend([f"![{Path(filename).stem.replace('_', ' ')}]({filename})", ""])
        else:
            lines.extend([f"[PDF: {Path(filename).stem.replace('_', ' ')}]({filename})", ""])
    report = destination / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _run_frozen_experiment(frozen: dict, destination: Path, workers: int) -> Path:
    jobs = []
    for scenario in frozen["scenarios"]:
        for seed in frozen["spec"]["seeds"]:
            trial_path = destination / scenario["name"] / f"seed_{seed}"
            expected_id = _expected_id(frozen, scenario, seed)
            complete = False
            attempts = sorted(trial_path.glob("attempt_*"))
            for attempt in attempts:
                if not (attempt / "manifest.json").exists():
                    continue
                manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
                if manifest["run_id"] != expected_id:
                    raise ValueError("trial identity does not match frozen experiment")
                if manifest["status"] == "complete":
                    render_run(attempt)
                    complete = True
            if not complete:
                output = trial_path / f"attempt_{len(attempts) + 1:03d}"
                jobs.append((scenario, seed, str(output), frozen["spec"]["purpose"], expected_id,
                             frozen["spec"].get("archive", "full")))
    if source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("source changed while preparing experiment")
    if workers == 1:
        disable_power_throttling()
        for job in jobs:
            print(json.dumps(_run_trial(*job)), flush=True)
    elif jobs:
        names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = "1"
            with ProcessPoolExecutor(max_workers=workers, initializer=disable_power_throttling) as pool:
                futures = [pool.submit(_run_trial, *job) for job in jobs]
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    if source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("source changed during experiment")
    summary = analyze(frozen, collect_records(destination, frozen))
    write_json(destination / "summary.json", summary)
    write_json(destination / "summary_manifest.json", {
        "experiment_id": frozen["experiment_id"], "summary_sha256": sha256_file(destination / "summary.json"),
    })
    return render_experiment(destination)


@contextmanager
def _experiment_lock(destination: Path):
    with (destination / "experiment.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_experiment(spec_path: str | Path, destination: str | Path, *, workers: int = 1, resume: bool = False) -> Path:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    frozen = freeze_spec(spec_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=resume)
    with _experiment_lock(destination):
        manifest_path = destination / "experiment.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != frozen:
                raise ValueError("cannot resume with different inputs, source, or environment")
        else:
            write_json(manifest_path, frozen)
        return _run_frozen_experiment(frozen, destination, workers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    if args.render_only:
        print(render_experiment(args.output))
        return 0
    if args.spec is None:
        parser.error("--spec is required for a new experiment or resume")
    print(run_experiment(args.spec, args.output, workers=args.workers, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
