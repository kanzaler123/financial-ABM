"""Train one scenario and evaluate a frozen table against both NPV rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from ..manifest import canonical_json_bytes, sha256_file
from ..tabular import CompanyQLearner
from . import runner
from .config import MarketConfig, load_market_config
from .domain import DomainSpec, load_domain
from .learning import action_kinds, load_frozen, save_frozen, table_fingerprint
from .policy import PolicySchedule

VALIDATION_RULES = ("q_learning", "npv_discrete", "npv")


def _seeds(values: Iterable[int], name: str) -> list[int]:
    seeds = list(values)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"{name} must contain nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must contain unique seeds")
    return seeds


def _run_record(folder: Path, destination: Path, seed: int, mode: str, phase: str) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("run archive is not complete")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(folder / name) != expected:
            raise ValueError(f"archive hash mismatch: {name}")
    record = {"phase": phase, "seed": seed, "firm_policy": mode,
              "path": folder.relative_to(destination).as_posix(), "status": "complete",
              "run_id": manifest["run_id"], "artifacts": manifest["artifacts"],
              "result_fingerprint": manifest["result_fingerprint"]}
    for name in ("metrics", "validation", "learning"):
        path = folder / (name + ".json")
        if path.exists():
            record[name] = json.loads(path.read_text(encoding="utf-8"))
    if mode == "q_learning" and "learning" not in record:
        raise ValueError("Q-learning archive is missing its learning report")
    return record


def _failed_record(folder: Path, destination: Path, seed: int, mode: str, phase: str, error: BaseException) -> dict:
    record = {"phase": phase, "seed": seed, "firm_policy": mode,
              "path": folder.relative_to(destination).as_posix(), "status": "failed",
              "error_type": type(error).__name__, "error": str(error)}
    path = folder / "manifest.json"
    if path.exists():
        try:
            record["archive_manifest"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as secondary:
            record["archive_manifest_error"] = str(secondary)
    return record


def _coverage(records: list[dict], counts: np.ndarray, expected_worlds: int) -> dict:
    evaluations = [row for row in records if row["phase"] == "validation" and row["firm_policy"] == "q_learning"]
    if len(evaluations) != expected_worlds or any(row["status"] != "complete" for row in evaluations):
        return {"status": "incomplete", "reason": "Not every planned frozen evaluation completed."}
    kinds = action_kinds(counts.shape[2])
    visits, legal = np.zeros((6, 81), dtype=np.int64), np.zeros_like(counts)
    worlds, undercovered, group_undercovered = [], [], []
    unseen = 0
    for row in evaluations:
        report = row["learning"]["coverage"]
        seen = np.asarray(report["visits"])
        allowed = np.asarray(report["legal_action_visits"])
        if (seen.shape != visits.shape or allowed.shape != legal.shape
                or not np.issubdtype(seen.dtype, np.integer) or not np.issubdtype(allowed.dtype, np.integer)
                or np.any(seen < 0) or np.any(allowed < 0) or np.any(allowed > seen[:, :, None])
                or np.any(allowed.sum(axis=2) < seen) or int(seen.sum()) != report["decisions"]):
            raise ValueError("invalid archived coverage counts")
        total = int(seen.sum())
        missing = int(seen[counts.sum(axis=2) == 0].sum())
        if missing != report["unseen_state_decisions"]:
            raise ValueError("archived unseen-state count disagrees with the frozen table")
        fraction = missing / total if total else None
        common = (seen > 0) & (seen >= .01 * total)
        insufficient = common[:, :, None] & (allowed > 0) & (counts < 20)
        for group, state, action in np.argwhere(insufficient):
            undercovered.append({"seed": row["seed"], "group": int(group), "state": int(state),
                                 "choice_index": int(action), "action": int(kinds[action]),
                                 "count": int(counts[group, state, action])})
        group_totals = seen.sum(axis=1)
        group_common = (seen > 0) & (seen >= .01 * group_totals[:, None])
        group_insufficient = group_common[:, :, None] & (allowed > 0) & (counts < 20)
        for group, state, action in np.argwhere(group_insufficient):
            group_undercovered.append({"seed": row["seed"], "group": int(group), "state": int(state),
                                       "choice_index": int(action), "action": int(kinds[action]),
                                       "count": int(counts[group, state, action])})
        group_reports = []
        for group, group_total in enumerate(group_totals):
            group_missing = int(seen[group, counts[group].sum(axis=1) == 0].sum())
            share = group_missing / int(group_total) if group_total else None
            status = "not_observed" if not group_total else (
                "PASS" if share <= .05 and not group_insufficient[group].any() else "FAIL"
            )
            group_reports.append({"group": group, "size_class": group % 3, "policy_epoch": group // 3,
                                  "decisions": int(group_total), "unseen_state_decisions": group_missing,
                                  "unseen_state_fraction": share, "status": status})
        global_status = "PASS" if fraction is not None and fraction <= .05 and not insufficient.any() else "FAIL"
        worlds.append({"seed": row["seed"], "decisions": total, "unseen_state_decisions": missing,
                       "unseen_state_fraction": fraction, "global_status": global_status, "groups": group_reports,
                       "status": "PASS" if global_status == "PASS"
                       and all(group["status"] != "FAIL" for group in group_reports) else "FAIL"})
        visits += seen
        legal += allowed
        unseen += missing
    total = int(visits.sum())
    common = (visits > 0) & (visits >= .01 * total)
    for group, state, action in np.argwhere(common[:, :, None] & (legal > 0) & (counts < 20)):
        undercovered.append({"seed": None, "group": int(group), "state": int(state),
                             "choice_index": int(action), "action": int(kinds[action]),
                             "count": int(counts[group, state, action])})
    group_totals = visits.sum(axis=1)
    group_common = (visits > 0) & (visits >= .01 * group_totals[:, None])
    group_insufficient = group_common[:, :, None] & (legal > 0) & (counts < 20)
    for group, state, action in np.argwhere(group_insufficient):
        group_undercovered.append({"seed": None, "group": int(group), "state": int(state),
                                   "choice_index": int(action), "action": int(kinds[action]),
                                   "count": int(counts[group, state, action])})
    group_reports = []
    for group, group_total in enumerate(group_totals):
        group_missing = int(visits[group, counts[group].sum(axis=1) == 0].sum())
        share = group_missing / int(group_total) if group_total else None
        status = "not_observed" if not group_total else (
            "PASS" if share <= .05 and not group_insufficient[group].any() else "FAIL"
        )
        group_reports.append({"group": group, "size_class": group % 3, "policy_epoch": group // 3,
                              "decisions": int(group_total), "unseen_state_decisions": group_missing,
                              "unseen_state_fraction": share, "status": status})
    fraction = unseen / total if total else None
    global_status = (
        "PASS" if fraction is not None and fraction <= .05 and not undercovered
        and all(world["global_status"] == "PASS" for world in worlds) else "FAIL"
    )
    return {"status": "PASS" if global_status == "PASS" and not group_undercovered
            and all(group["status"] != "FAIL" for group in group_reports)
            and all(world["status"] == "PASS" for world in worlds) else "FAIL",
            "scope": "Global and each observed size/policy-epoch group must pass in every validation world and pooled.",
            "decisions": total, "unseen_state_decisions": unseen, "unseen_state_fraction": fraction,
            "maximum_unseen_state_fraction": .05, "common_state_share_threshold": .01,
            "common_state_minimum_action_count": 20, "common_state_undercovered_actions": undercovered,
            "group_common_state_share_threshold": .01,
            "group_common_state_undercovered_actions": group_undercovered,
            "groups": group_reports, "global_status": global_status,
            "worlds": worlds, "visits": visits.tolist(), "legal_action_visits": legal.tolist()}


def _validation(records: list[dict], planned_runs: int) -> dict:
    result = {}
    for name in ("mechanical", "stylized", "return_acf", "cross_asset_correlation", "empirical_calibration"):
        statuses = [{"phase": row["phase"], "seed": row["seed"], "firm_policy": row["firm_policy"],
                     "status": row.get("validation", {}).get(name, {}).get("status", "incomplete")}
                    for row in records]
        values = [row["status"] for row in statuses]
        status = "FAIL" if "FAIL" in values else (
            "incomplete" if len(values) != planned_runs or "incomplete" in values else
            "PASS" if values and all(value == "PASS" for value in values) else "pending"
        )
        result[name] = {"status": status, "runs": statuses}
    return result


def train_and_evaluate(
    config: MarketConfig, domain: DomainSpec, policy: PolicySchedule | None, output: str | Path, *,
    training_seeds: Iterable[int], validation_seeds: Iterable[int], reserved_seeds: Iterable[int] = (),
    coverage_target: int = 0,
    common_prepolicy_model: str | Path | None = None, common_prepolicy_sha256: str | None = None,
    validation_rules: Iterable[str] = VALIDATION_RULES, archive: str = "full",
) -> dict:
    """Run a fixed development budget; refuse overwrite and automatic resume.

    ``validation_rules`` keeps the frozen Q-learning run and any of the NPV rules (in the default order);
    ``archive="compact"`` drops the day-by-firm matrices from every run's result.npz.
    """
    if type(coverage_target) is not int or coverage_target < 0:
        raise ValueError("coverage_target must be a nonnegative integer")
    rules = list(validation_rules)
    if "q_learning" not in rules or rules != [rule for rule in VALIDATION_RULES if rule in rules]:
        raise ValueError("validation_rules must include q_learning and follow q_learning, npv_discrete, npv")
    if archive not in ("full", "compact"):
        raise ValueError("archive must be 'full' or 'compact'")
    train_seeds, valid_seeds = _seeds(training_seeds, "training_seeds"), _seeds(validation_seeds, "validation_seeds")
    if set(train_seeds) & set(valid_seeds):
        raise ValueError("training and validation seeds must not overlap")
    reserved = list(reserved_seeds)
    if reserved:
        reserved = _seeds(reserved, "reserved_seeds")
    if set(train_seeds + valid_seeds) & set(reserved):
        raise ValueError("training or validation seeds overlap reserved_seeds")
    policy = policy or PolicySchedule.constant(domain)
    learning_options = {"coverage_target": coverage_target} if coverage_target else {}
    if (common_prepolicy_model is None) != (common_prepolicy_sha256 is None):
        raise ValueError("common pre-policy model and SHA256 must be supplied together")
    common_source, common_origin = None, None
    if common_prepolicy_sha256 is not None:
        learning_options["common_prepolicy_sha256"] = common_prepolicy_sha256
    contract = runner.learning_contract(config, domain, policy, **learning_options)
    if common_prepolicy_model is not None:
        common_source, common_origin = runner.load_common_prepolicy(common_prepolicy_model, common_prepolicy_sha256,
                                                                   config, domain, policy, coverage_target=coverage_target)
        used = set(common_origin["training_seeds"] + common_origin["validation_seeds"])
        if (used & set(train_seeds + valid_seeds + reserved)
                or set(common_origin["reserved_seeds"]) & set(train_seeds + valid_seeds)):
            raise ValueError("common pre-policy seed lineage overlaps the current budget")
    epsilon = np.linspace(1.0, .1, len(train_seeds)).tolist()
    choice_count = 2 * len(config.firm_action_fractions) + 1
    initial_table = {"shape": [6, 81, choice_count], "values": 0.0, "counts": 0,
                     "values_dtype": "float64", "counts_dtype": "int64"}
    if common_origin is not None:
        initial_table = {"shape": [6, 81, choice_count], "prefix_groups": [0, 1, 2],
                         "prefix_source_sha256": common_origin["sha256"],
                         "prefix_table_fingerprint": common_origin["table_fingerprint"],
                         "post_groups": [3, 4, 5], "post_values": 0.0, "post_counts": 0,
                         "values_dtype": "float64", "counts_dtype": "int64"}
    budget = {"training_seeds": train_seeds, "validation_seeds": valid_seeds,
              "reserved_seeds": reserved,
              "training_worlds": len(train_seeds), "validation_worlds_per_rule": len(valid_seeds),
              "validation_rules": rules, "epsilon_by_world": epsilon,
              "learning_rate": .1, "initial_table": initial_table,
              "coverage": {"scope": "each_world_and_pooled", "maximum_unseen_state_fraction": .05,
                           "common_state_share_threshold": .01, "minimum_action_count": 20,
                           "group_scope": "each_observed_size_policy_epoch_group",
                           "group_common_state_share_threshold": .01, "unobserved_group_status": "not_observed"}}
    if coverage_target:
        budget["coverage_target"] = coverage_target
    if archive != "full":
        budget["archive"] = archive
    archive_option = {"archive": archive} if archive != "full" else {}
    if config.firm_learning_rate_power:
        budget["learning_rate_power"] = config.firm_learning_rate_power
        budget["step_size_rule"] = "min(1, k * (n + k) ** -learning_rate_power) per batch of k same-key samples; learning_rate unused"
    spec = {"schema_version": 1, "purpose": "development", "contract": contract,
            "config": config.to_dict(), "domain": domain.to_dict(), "policy": policy.to_dict(), "budget": budget,
            "comparison_scope": "within_scenario_algorithm_comparison",
            "discount_rule": "exp(-firm_discount_rate * elapsed_days / 252); learner.discount is unused",
            "resumable": False}
    if common_origin is not None:
        spec.update({"common_prepolicy": common_origin, "comparison_scope": "conditional_strategy_deployment"})
    experiment_id = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    runner.write_json(destination / "experiment.json", {**spec, "experiment_id": experiment_id})
    learner = CompanyQLearner(groups=6, actions=choice_count, epsilon=1.0, learning_rate=.1)
    learner.learning_rate_power = config.firm_learning_rate_power
    if common_source is not None:
        learner.values[:3] = common_source.values[:3]
        learner.counts[:3] = common_source.counts[:3]
    completed, records = [], []
    metadata = {"contract": contract, "budget": budget, "initial_table": initial_table,
                "completed_training_seeds": completed.copy(), "experiment_id": experiment_id,
                "discount_rule": spec["discount_rule"]}
    if common_origin is not None:
        metadata["common_prepolicy"] = common_origin
    summary = {"experiment_id": experiment_id, "purpose": "development", "status": "running",
               "phase": "training", "claim_support": "pending", "m4_status": "pending",
               "comparison_scope": spec["comparison_scope"], "contract": contract, "budget": budget,
               "initial_model_sha256": None, "completed_training_seeds": completed,
               "records": records, "limitations": [
                   "Coverage gates and successful runs do not establish empirical validity or a policy ranking.",
                   "Scenario-specific tables may learn future policy through bootstrap before its switch; epoch labels do not ensure paired pre-policy paths.",
                   "Across-scenario policy causality requires a common pre-policy table and switch protocol.",
                   "Observed private ledger rewards and unfinished liabilities are not lifetime profit or social welfare.",
               ]}
    if common_origin is not None:
        summary["common_prepolicy"] = common_origin
        summary["limitations"][1:3] = [
            "This benchmark deploys a separately trained conditional strategy after a frozen greedy baseline prefix; it is not from-zero online policy learning.",
            "Cross-scenario pre-policy pairing requires the same independently trained baseline model SHA256, environment and switch calendar.",
        ]
    summary_path = destination / "summary.json"
    runner.write_json(summary_path, summary)
    planned_runs = len(train_seeds) + len(rules) * len(valid_seeds)
    frozen_counts = None
    try:
        summary["initial_model_sha256"] = save_frozen(destination / "initial_model.npz", learner, metadata)
        runner.write_json(summary_path, summary)
        for index, seed in enumerate(train_seeds):
            if runner.learning_contract(config, domain, policy, **learning_options) != contract:
                raise RuntimeError("learning contract changed during the training budget")
            learner.epsilon = epsilon[index]
            folder = destination / "training" / f"seed_{seed}"
            run_metadata = {**metadata, "completed_training_seeds": completed.copy(),
                            "phase": "training", "epsilon": learner.epsilon}
            try:
                runner.run_and_archive(replace(config, seed=seed, firm_policy="q_learning"), domain, folder,
                                       policy, purpose="development", learner=learner, training=True,
                                       learning_metadata=run_metadata, **archive_option, **learning_options)
                record = _run_record(folder, destination, seed, "q_learning", "training")
            except BaseException as error:
                record = _failed_record(folder, destination, seed, "q_learning", "training", error)
                records.append(record)
                if record.get("archive_manifest", {}).get("status") == "complete":
                    completed.append(seed)
                raise
            records.append(record)
            completed.append(seed)
            runner.write_json(summary_path, summary)
        metadata = {**metadata, "completed_training_seeds": completed.copy(), "phase": "frozen"}
        frozen_path = destination / "frozen_model.npz"
        frozen_hash = save_frozen(frozen_path, learner, metadata)
        frozen_counts = learner.counts.copy()
        summary.update({"phase": "validation", "frozen_model_sha256": frozen_hash,
                        "frozen_table_fingerprint": table_fingerprint(learner)})
        runner.write_json(summary_path, summary)
        for seed in valid_seeds:
            frozen, restored_metadata = load_frozen(frozen_path, metadata, expected_sha256=frozen_hash)
            if frozen.values.flags.writeable or frozen.counts.flags.writeable or frozen.epsilon != 0:
                raise RuntimeError("evaluation model is not frozen")
            before = table_fingerprint(frozen)
            for mode in budget["validation_rules"]:
                if runner.learning_contract(config, domain, policy, **learning_options) != contract:
                    raise RuntimeError("learning contract changed during validation")
                folder = destination / "validation" / f"seed_{seed}" / mode
                kwargs = {"learner": frozen, "learning_metadata": restored_metadata, **learning_options} if mode == "q_learning" else {}
                try:
                    runner.run_and_archive(replace(config, seed=seed, firm_policy=mode), domain, folder,
                                           policy, purpose="development", **archive_option, **kwargs)
                    record = _run_record(folder, destination, seed, mode, "validation")
                except BaseException as error:
                    record = _failed_record(folder, destination, seed, mode, "validation", error)
                    if not isinstance(error, Exception):
                        records.append(record)
                        raise
                records.append(record)
                if (table_fingerprint(frozen) != before or frozen.values.flags.writeable
                        or frozen.counts.flags.writeable or frozen.epsilon != 0
                        or sha256_file(frozen_path) != frozen_hash):
                    raise RuntimeError("evaluation changed the frozen learning table")
                runner.write_json(summary_path, summary)
        if runner.learning_contract(config, domain, policy, **learning_options) != contract:
            raise RuntimeError("learning contract changed during validation")
        summary.update({"status": "complete" if all(row["status"] == "complete" for row in records) else "incomplete",
                        "phase": "finished", "validation": _validation(records, planned_runs)})
        summary["validation"]["coverage"] = _coverage(records, frozen_counts, len(valid_seeds))
        runner.write_json(summary_path, summary)
    except BaseException as error:
        summary.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "error_type": type(error).__name__, "error": str(error), "resumable": False,
                        "validation": _validation(records, planned_runs)})
        summary["validation"]["coverage"] = {"status": "incomplete", "reason": "The fixed budget did not finish."}
        if summary["phase"] == "training":
            path = destination / "failed_training_state.npz"
            try:
                with path.open("xb") as handle:
                    np.savez_compressed(handle, values=learner.values, counts=learner.counts,
                                        epsilon=np.array(learner.epsilon))
                summary["failed_training_state"] = {"path": path.name, "sha256": sha256_file(path), "resumable": False}
            except Exception as secondary:
                summary["failed_checkpoint_error"] = {"error_type": type(secondary).__name__, "error": str(secondary)}
        runner.write_json(summary_path, summary)
        raise
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--reserved-seeds", type=int, nargs="*", default=())
    parser.add_argument("--coverage-target", type=int, default=0)
    parser.add_argument("--common-prepolicy-model", type=Path)
    parser.add_argument("--common-prepolicy-sha256")
    args = parser.parse_args(argv)
    config = load_market_config(args.config) if args.config else MarketConfig()
    domain = load_domain(args.domain)
    policy = runner.load_policy(args.policy, domain) if args.policy else None
    summary = train_and_evaluate(config, domain, policy, args.output,
                                 training_seeds=args.training_seeds, validation_seeds=args.validation_seeds,
                                 reserved_seeds=args.reserved_seeds,
                                 **({"coverage_target": args.coverage_target} if args.coverage_target else {}),
                                 **({"common_prepolicy_model": args.common_prepolicy_model,
                                     "common_prepolicy_sha256": args.common_prepolicy_sha256}
                                    if args.common_prepolicy_model is not None or args.common_prepolicy_sha256 is not None else {}))
    print(json.dumps({"status": summary["status"], "summary": str(args.output / "summary.json"),
                      "coverage": summary["validation"]["coverage"]["status"]}))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
