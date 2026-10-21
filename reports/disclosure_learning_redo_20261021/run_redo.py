"""Learning-firm redo (contract section 23): train, validate and replay every table of a spec, unattended.

  python run_redo.py --spec configs/disclosure_learning_redo_confirmatory_v1.json --output redo_v1 [--workers 16] [--resume]

Jobs, each run in its own Python process (a crash or MemoryError affects only that job):
- pre:    one pre-policy table per replicate and state version (constant baseline; frozen Q-learning on the check seeds);
- post:   one post-policy table per replicate, version and scenario, on the frozen pre-policy table of its replicate
          and version (protocol frozen_baseline_greedy_prefix_post_zero_v1; frozen Q-learning on the validation seeds);
- npv:    discrete and continuous NPV on every scenario and validation seed (shared by both versions);
- replay: an instrumented replay of every frozen Q-learning validation run (result fingerprint must match), giving
          the shared-grid NPV values of every candidate and the inputs of every learning state.
Pre-policy jobs run first; free slots take the next ready job. Each job writes into its own attempt_NNN folder.
A failed job is not retried and its dependants are skipped; --resume starts the next attempt of every job that is
not complete (contract 23.10: infrastructure failures only). Progress goes to <output>/orchestrator.log, job logs
to <output>/logs/. Start detached (PowerShell Start-Process); the scheduler itself does no simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import _no_wmi  # noqa: E402

_no_wmi.prefill_uname()

from abm.manifest import canonical_json_bytes  # noqa: E402

KINDS = ("pre", "post", "npv", "replay")


# ---------------------------------------------------------------- spec, paths, status

def load_spec(path: Path) -> tuple[dict, str]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec, hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


def job_name(job: dict) -> str:
    if job["kind"] == "pre":
        return f"pre_r{job['replicate']:02d}_{job['version']}"
    if job["kind"] == "post":
        return f"post_r{job['replicate']:02d}_{job['version']}_{job['scenario']}"
    if job["kind"] == "npv":
        return f"npv_{job['scenario']}_{job['seed']}_{job['rule']}"
    return f"replay_r{job['replicate']:02d}_{job['version']}_{job['scenario']}_{job['seed']}"


def job_root(output: Path, job: dict) -> Path:
    if job["kind"] == "pre":
        return output / f"rep_{job['replicate']:02d}" / job["version"] / "pre"
    if job["kind"] == "post":
        return output / f"rep_{job['replicate']:02d}" / job["version"] / job["scenario"]
    if job["kind"] == "npv":
        return output / "npv" / job["scenario"] / f"seed_{job['seed']}" / job["rule"]
    return output / "replay" / f"rep_{job['replicate']:02d}" / job["version"] / job["scenario"] / f"seed_{job['seed']}"


def attempts(output: Path, job: dict) -> list[Path]:
    return sorted(job_root(output, job).glob("attempt_*"))


def attempt_complete(folder: Path, kind: str) -> bool:
    name = {"pre": "summary.json", "post": "summary.json", "npv": "manifest.json", "replay": "replay.json"}[kind]
    path = folder / name
    if not path.exists():
        return False
    if kind == "replay":
        return True
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def complete_attempt(output: Path, job: dict) -> Path | None:
    done = [folder for folder in attempts(output, job) if attempt_complete(folder, job["kind"])]
    return done[-1] if done else None


def all_jobs(spec: dict) -> list[dict]:
    jobs = []
    for row in spec["replicates"]:
        for version in spec["versions"]:
            jobs.append({"kind": "pre", "replicate": row["replicate"], "version": version})
            for scenario in spec["scenarios"]:
                jobs.append({"kind": "post", "replicate": row["replicate"], "version": version, "scenario": scenario})
                for seed in spec["validation_seeds"]:
                    jobs.append({"kind": "replay", "replicate": row["replicate"], "version": version,
                                 "scenario": scenario, "seed": seed})
    for scenario in spec["scenarios"]:
        for seed in spec["validation_seeds"]:
            for rule in spec["npv_rules"]:
                jobs.append({"kind": "npv", "scenario": scenario, "seed": seed, "rule": rule})
    return jobs


def dependency(job: dict) -> dict | None:
    if job["kind"] == "post":
        return {"kind": "pre", "replicate": job["replicate"], "version": job["version"]}
    if job["kind"] == "replay":
        return {"kind": "post", "replicate": job["replicate"], "version": job["version"], "scenario": job["scenario"]}
    return None


# ---------------------------------------------------------------- job bodies (run in a child process)

def _inputs(spec: dict, version: str | None, scenario: str | None):
    from abm.disclosure.config import MarketConfig
    from abm.disclosure.domain import load_domain
    from abm.disclosure.policy import PolicySchedule

    path = ROOT / spec["domain"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["domain_sha256"]:
        raise RuntimeError("domain file changed")
    domain = load_domain(path)
    config = MarketConfig.from_dict(spec["config"])
    if version is not None:
        config = replace(config, firm_learning_context=spec["versions"][version])
    policy = PolicySchedule.constant(domain)
    if scenario is not None:
        policy = policy.with_intervention(spec["switch_day"], **spec["scenarios"][scenario])
    return config, domain, policy


def _replicate(spec: dict, number: int) -> dict:
    return next(row for row in spec["replicates"] if row["replicate"] == number)


def run_pre(spec: dict, job: dict, folder: Path) -> None:
    from abm.disclosure.training import train_and_evaluate

    config, domain, policy = _inputs(spec, job["version"], None)
    summary = train_and_evaluate(
        config, domain, policy, folder, training_seeds=_replicate(spec, job["replicate"])["prepolicy_training_seeds"],
        validation_seeds=spec["check_seeds"], reserved_seeds=spec["reserved_seeds"],
        coverage_target=spec["coverage_target"], validation_rules=spec["prepolicy_validation_rules"],
        archive=spec["archive"])
    if summary["status"] != "complete":
        raise RuntimeError(f"pre-policy table ended with status {summary['status']}")


def run_post(spec: dict, job: dict, folder: Path, output: Path) -> None:
    from abm.disclosure.training import train_and_evaluate

    source = complete_attempt(output, dependency(job))
    if source is None:
        raise RuntimeError("the pre-policy table is not complete")
    summary_in = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config, domain, policy = _inputs(spec, job["version"], job["scenario"])
    summary = train_and_evaluate(
        config, domain, policy, folder, training_seeds=_replicate(spec, job["replicate"])["postpolicy_training_seeds"],
        validation_seeds=spec["validation_seeds"], reserved_seeds=spec["reserved_seeds"],
        coverage_target=spec["coverage_target"], common_prepolicy_model=source / "frozen_model.npz",
        common_prepolicy_sha256=summary_in["frozen_model_sha256"],
        validation_rules=spec["postpolicy_validation_rules"], archive=spec["archive"])
    if summary["status"] != "complete":
        raise RuntimeError(f"post-policy table ended with status {summary['status']}")


def run_npv(spec: dict, job: dict, folder: Path) -> None:
    from abm.disclosure.runner import run_and_archive

    config, domain, policy = _inputs(spec, None, job["scenario"])
    run_and_archive(replace(config, seed=job["seed"], firm_policy=job["rule"]), domain, folder, policy,
                    purpose="development", archive=spec["archive"])


def replay(folder: Path, window_start: int) -> dict:
    """Rerun a frozen Q-learning validation run with read-only wrappers; the fingerprint must match the archive."""
    import numpy as np

    from abm.disclosure import game
    from abm.disclosure.config import MarketConfig
    from abm.disclosure.domain import DomainSpec
    from abm.disclosure.game import MINIMUM_NPV, DisclosureMarket
    from abm.disclosure.learning import IMPROVE, INFLATE, MAINTAIN, CandidateSet, action_kinds, load_frozen, npv_discrete
    from abm.disclosure.policy import PolicyRegime, PolicySchedule

    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    learning = manifest["learning"]
    config = MarketConfig.from_dict(manifest["config"])
    domain = DomainSpec.from_dict(manifest["domain"])
    payload = manifest["policy"]
    policy = PolicySchedule(PolicyRegime(**payload["baseline"]),
                            None if payload["intervention"] is None else PolicyRegime(**payload["intervention"]),
                            payload["start_day"])
    learner, _ = load_frozen(folder / "input_model.npz", expected_sha256=manifest["artifacts"]["input_model.npz"])
    market = DisclosureMarket(config, domain, policy, learner=learner, training=False,
                              coverage_target=learning.get("coverage_target", 0),
                              common_prepolicy_sha256=learning.get("common_prepolicy_sha256"))
    values, stocks, signals = [], [], []
    original_candidates, original_states = game.discrete_candidates, game.context_states

    def candidates(**kwargs):
        result = original_candidates(**kwargs)
        values.append(result.values.copy())
        return result

    def states(**kwargs):
        stock = kwargs.get("improvement_stock")
        stocks.append(None if stock is None else np.array(stock, dtype=float))
        signals.append(kwargs.get("own_signal") is not None)
        return original_states(**kwargs)

    game.discrete_candidates, game.context_states = candidates, states
    try:
        result = market.run()
    finally:
        game.discrete_candidates, game.context_states = original_candidates, original_states
    if result.fingerprint() != manifest["result_fingerprint"]:
        raise RuntimeError(f"replay of {folder} does not reproduce the archived result")
    values = np.stack(values)  # decision periods 1.. x firms x choices
    periods = values.shape[0]
    choices, kinds = result.learning_choices[1:periods + 1], np.asarray(result.actions[1:periods + 1])
    decision_days = np.arange(1, periods + 1) * config.firm_decision_interval
    window = decision_days > window_start
    rows = np.arange(values.shape[1])
    chosen = np.take_along_axis(values, choices[:, :, None], axis=2)[:, :, 0]
    loss = values[:, :, MAINTAIN] - chosen
    wash = (kinds == INFLATE) & window[:, None]
    legal = result.valid_actions[1:periods + 1]
    grid = action_kinds(values.shape[2])
    empty = np.zeros(values.shape[1:])
    npv_choice = np.stack([npv_discrete(CandidateSet(empty, empty, empty, values[p], legal[p], grid), MINIMUM_NPV)
                           for p in range(periods)])
    npv_value = np.take_along_axis(values, npv_choice[:, :, None], axis=2)[:, :, 0]
    trap = wash & (grid[npv_choice] != INFLATE)
    # State inputs: version A never passes a stock; version V passes the committed real improvement before each decision.
    context = manifest["config"].get("firm_learning_context", "exposure")
    committed = np.where(np.asarray(result.actions) == IMPROVE, np.asarray(result.amounts), 0.0)
    expected_stock = np.vstack([np.zeros((1, committed.shape[1])), np.cumsum(committed, axis=0)[:-1]])[1:periods + 1]
    if context == "improvement_stock":
        # The market sums the history in another order, so the stock agrees to rounding; the bins must agree exactly.
        state_check = (all(stock is not None for stock in stocks) and all(signals)
                       and np.allclose(np.stack(stocks), expected_stock, rtol=0, atol=1e-12)
                       and np.array_equal(result.learning_states[1:periods + 1] // 3 % 3,
                                          np.digitize(np.stack(stocks), (1e-12, domain.max_change_per_period))))
    else:
        state_check = all(stock is None for stock in stocks) and all(signals) == (context == "signal_tercile")
    size = market.firms.size_class
    out = {"fingerprint_matches": True, "context": context, "state_inputs_ok": bool(state_check),
           "decisions": int(window.sum() * rows.size),
           "wash_decisions": int(wash.sum()), "improve_decisions": int(((kinds == IMPROVE) & window[:, None]).sum()),
           "wash_below_maintain": int((wash & (loss > 0)).sum()),
           "wash_loss_bp": (loss[wash & (loss > 0)] * 1e4).tolist(),
           "trap_decisions": int(trap.sum()),
           "npv_choice_of_trap_improve_wash_maintain": np.bincount(grid[npv_choice][trap], minlength=3).tolist(),
           "trap_loss_vs_npv_choice_bp": ((npv_value - chosen)[trap] * 1e4).tolist(),
           "npv_shadow_wash_decisions": int(((grid[npv_choice] == INFLATE) & window[:, None]).sum()),
           "by_size": {}}
    for cls, label in enumerate(("large", "mid", "small")):
        member = window[:, None] & (size == cls)[None, :]
        out["by_size"][label] = {"decisions": int(member.sum()), "wash_decisions": int((wash & member).sum()),
                                 "wash_below_maintain": int((wash & member & (loss > 0)).sum()),
                                 "trap_decisions": int((trap & member).sum())}
    return out


def run_replay(spec: dict, job: dict, folder: Path, output: Path) -> None:
    source = complete_attempt(output, dependency(job))
    if source is None:
        raise RuntimeError("the post-policy table is not complete")
    run = source / "validation" / f"seed_{job['seed']}" / "q_learning"
    record = replay(run, spec["window"]["start_day"])
    record["path"] = run.relative_to(output).as_posix()
    if not record["state_inputs_ok"]:
        raise RuntimeError(f"state inputs of {run} differ from the definition")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "replay.json").write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")


def run_job(spec_path: Path, output: Path, job: dict, folder: Path) -> None:
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    spec, _ = load_spec(spec_path)
    if job["kind"] == "pre":
        run_pre(spec, job, folder)
    elif job["kind"] == "post":
        run_post(spec, job, folder, output)
    elif job["kind"] == "npv":
        run_npv(spec, job, folder)
    else:
        run_replay(spec, job, folder, output)


# ---------------------------------------------------------------- scheduler

def log(output: Path, message: str) -> None:
    with (output / "orchestrator.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def schedule(spec_path: Path, output: Path, workers: int, resume: bool) -> int:
    spec, experiment_id = load_spec(spec_path)
    from abm.disclosure.runner import source_hashes

    sources = source_hashes()
    identity = {"experiment_id": experiment_id, "spec": spec_path.resolve().relative_to(ROOT).as_posix(),
                "spec_file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "source_digest": hashlib.sha256(canonical_json_bytes(sources)).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    marker = output / "experiment.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if {k: previous[k] for k in identity} != identity:
            raise SystemExit("output belongs to another spec or source; refusing to mix")
        if not resume:
            raise SystemExit("output exists; pass --resume to continue it")
    else:
        marker.write_text(json.dumps({**identity, "source_hashes": sources}, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    jobs = all_jobs(spec)
    order = {kind: index for index, kind in enumerate(KINDS)}
    jobs.sort(key=lambda job: order[job["kind"]])
    running: dict[str, tuple[subprocess.Popen, dict, Path, object]] = {}
    failed: set[str] = set()
    log(output, f"start {experiment_id} workers={workers} resume={resume} jobs={len(jobs)}")
    while True:
        for name, (process, job, folder, handle) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            del running[name]
            ok = attempt_complete(folder, job["kind"])
            if not ok:
                failed.add(name)
            log(output, f"{'done' if ok else 'FAILED'} {name} exit={process.returncode} {folder.relative_to(output).as_posix()}")
        pending = []
        for job in jobs:
            name = job_name(job)
            if name in running or name in failed or complete_attempt(output, job) is not None:
                continue
            parent = dependency(job)
            if parent is not None and (job_name(parent) in failed or complete_attempt(output, parent) is None):
                if job_name(parent) in failed:
                    failed.add(name)
                    log(output, f"SKIPPED {name}: dependency {job_name(parent)} failed")
                continue
            if attempts(output, job) and not resume:
                failed.add(name)
                log(output, f"SKIPPED {name}: an incomplete attempt exists (use --resume)")
                continue
            pending.append(job)
        if not running and not pending:
            break
        for job in pending[:max(0, workers - len(running))]:
            name = job_name(job)
            folder = job_root(output, job) / f"attempt_{len(attempts(output, job)) + 1:03d}"
            folder.parent.mkdir(parents=True, exist_ok=True)
            handle = (output / "logs" / f"{name}_{folder.name}.log").open("w", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--spec", str(spec_path.resolve()),
                       "--output", str(output.resolve()), "--job", json.dumps(job), "--folder", str(folder.resolve())]
            running[name] = (subprocess.Popen(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT, env=env),
                             job, folder, handle)
            log(output, f"started {name} {folder.relative_to(output).as_posix()}")
        time.sleep(5)
    status = {job_name(job): ("complete" if complete_attempt(output, job) is not None else "failed") for job in jobs}
    counts = {state: sum(value == state for value in status.values()) for state in ("complete", "failed")}
    (output / "status.json").write_text(json.dumps({"counts": counts, "jobs": status}, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")
    log(output, f"finished {counts}")
    return 0 if counts["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.job is not None:
        run_job(args.spec, args.output, json.loads(args.job), args.folder)
        return 0
    if not 1 <= args.workers <= 24:
        raise SystemExit("workers must be between 1 and 24 (machine-wide limit)")
    return schedule(args.spec, args.output, args.workers, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
