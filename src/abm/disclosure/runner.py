"""Archive and inspect reproducible single disclosure-market runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import canonical_json_bytes, sha256_file
from ..platform_tuning import disable_power_throttling
from ..tabular import CompanyQLearner
from .config import TRADING_DAYS_PER_YEAR, MarketConfig, load_market_config
from .domain import DomainSpec, load_domain
from .game import DisclosureMarket, DisclosureResult
from .harness import CrossSectionResult
from .learning import action_kinds, load_frozen, save_frozen, table_fingerprint
from .metrics import compute_metrics
from .policy import PolicyRegime, PolicySchedule
from .validate import validate_market


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package.parent).as_posix(): sha256_file(path)
            for path in sorted(package.rglob("*.py"))}


def learning_contract(
    config: MarketConfig, domain: DomainSpec, policy: PolicySchedule, *, coverage_target: int = 0,
    common_prepolicy_sha256: str | None = None,
) -> dict[str, Any]:
    if type(coverage_target) is not int or coverage_target < 0:
        raise ValueError("coverage_target must be a nonnegative integer")
    environment = config.to_dict()
    environment.pop("seed")
    environment.pop("firm_policy")
    contract = {
        "version": 2, "candidate_fractions": list(config.firm_action_fractions),
        "choice_kinds": action_kinds(2 * len(config.firm_action_fractions) + 1).tolist(),
        "exploration": "uniform_legal_kind_then_uniform_legal_amplitude",
        "feasibility": "positive_value_anchor_after_known_commitments_v1",
        "groups": 6, "config": environment, "domain": domain.to_dict(),
        "policy": policy.to_dict(), "source_hashes": source_hashes(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    if coverage_target:
        contract.update({"coverage_target": coverage_target,
                         "coverage_exploration": "minimum_completed_td_plus_input_row_reservations_then_kind_amplitude_v1"})
    if common_prepolicy_sha256 is not None:
        if (not isinstance(common_prepolicy_sha256, str) or len(common_prepolicy_sha256) != 64
                or any(char not in "0123456789abcdef" for char in common_prepolicy_sha256)):
            raise ValueError("common_prepolicy_sha256 must be a SHA256 digest")
        if policy.intervention is None or not 0 < policy.start_day <= config.trading_days:
            raise ValueError("common pre-policy deployment requires an explicit policy switch")
        contract.update({"common_prepolicy_sha256": common_prepolicy_sha256,
                         "prepolicy_protocol": "frozen_baseline_greedy_prefix_post_zero_v1"})
    return contract


def load_common_prepolicy(
    path: str | Path, expected_sha256: str, config: MarketConfig, domain: DomainSpec,
    policy: PolicySchedule, *, coverage_target: int = 0,
) -> tuple[CompanyQLearner, dict[str, Any]]:
    source, metadata = load_frozen(path, expected_sha256=expected_sha256)
    expected = learning_contract(config, domain, PolicySchedule(policy.baseline), coverage_target=coverage_target)
    budget = metadata.get("budget", {})
    seeds = {}
    for name in ("training_seeds", "validation_seeds", "reserved_seeds"):
        values = budget.get(name, [])
        if (not isinstance(values, list) or (name != "reserved_seeds" and not values)
                or any(type(seed) is not int or seed < 0 for seed in values)
                or len(set(values)) != len(values)):
            raise ValueError("common pre-policy source has invalid seed lineage")
        seeds[name] = values
    if (set(seeds["training_seeds"]) & set(seeds["validation_seeds"])
            or set(seeds["training_seeds"] + seeds["validation_seeds"]) & set(seeds["reserved_seeds"])):
        raise ValueError("common pre-policy source has overlapping seed lineage")
    shape = [6, 81, 2 * len(config.firm_action_fractions) + 1]
    initial = {"shape": shape, "values": 0.0, "counts": 0,
               "values_dtype": "float64", "counts_dtype": "int64"}
    expected_samples = config.n_firms * max(config.trading_days // config.firm_decision_interval - 1, 0) * len(seeds["training_seeds"])
    if (metadata.get("phase") != "frozen" or metadata.get("contract") != expected
            or "common_prepolicy" in metadata or metadata.get("completed_training_seeds") != seeds["training_seeds"]
            or metadata.get("initial_table") != initial or budget.get("initial_table") != initial
            or budget.get("training_worlds") != len(seeds["training_seeds"])
            or budget.get("learning_rate") != source.learning_rate or source.learning_rate != .1 or source.discount != .9
            or budget.get("learning_rate_power", 0.0) != source.learning_rate_power
            or source.learning_rate_power != config.firm_learning_rate_power
            or list(source.values.shape) != shape or source.values.dtype != np.float64 or source.counts.dtype != np.int64
            or np.any(source.values[3:] != 0) or np.any(source.counts[3:] != 0)
            or int(source.counts.sum()) != expected_samples):
        raise ValueError("common pre-policy source must be a complete independent frozen baseline training")
    origin = {"model_path": str(Path(path).resolve()), "sha256": expected_sha256,
              "table_fingerprint": table_fingerprint(source, prefix_groups=3),
              **seeds, "source_experiment_id": metadata.get("experiment_id"), "source_contract": expected}
    return source, origin


def run_identity(
    config: MarketConfig, domain: DomainSpec, policy: PolicySchedule, *, learning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "config": config.to_dict(),
        "domain": domain.to_dict(),
        "policy": policy.to_dict(),
        "source_hashes": source_hashes(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if learning is not None:
        identity["learning"] = learning
    return {**identity, "run_id": hashlib.sha256(canonical_json_bytes(identity)).hexdigest()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


# Day-by-firm float matrices dropped by a compact archive (about 27 of 28 MB at contract scale).
# The manifest fingerprint still covers them; a deterministic replay regenerates them.
COMPACT_OMITTED = ("true_goodness", "public_signal", "planning_signal", "market_prices", "market_intrinsic_values")
# archive="sampled" adds these day-by-firm arrays on every SAMPLE_INTERVAL_DAYS-th day from
# SAMPLE_START_DAY: one quarter (63 trading days) apart, like the rating panel's observation
# dates (research contract section 25.3).
SAMPLE_START_DAY = 1260
SAMPLE_INTERVAL_DAYS = 63
SAMPLED_ARRAYS = ("public_signal", "true_goodness", "reversion_anchor")


def compact_omitted(result: DisclosureResult) -> tuple[str, ...]:
    """COMPACT_OMITTED, plus the day-by-firm anchor history when reversion has permanent levels."""
    return COMPACT_OMITTED + (("reversion_anchor",) if result.reversion_anchor is not None else ())


def sampled_arrays(result: DisclosureResult) -> dict[str, np.ndarray]:
    days = np.arange(SAMPLE_START_DAY, np.asarray(result.true_goodness).shape[0], SAMPLE_INTERVAL_DAYS)
    arrays = {"sampled_days": days}
    for name in SAMPLED_ARRAYS:
        value = getattr(result, name)
        if value is not None:
            arrays["sampled_" + name] = np.asarray(value)[days]
    return arrays


def save_result(path: Path, result: DisclosureResult, *, omit: tuple[str, ...] = (),
                extra: dict[str, np.ndarray] | None = None) -> None:
    arrays = {item.name: getattr(result, item.name) for item in fields(result)
              if item.name != "market" and getattr(result, item.name) is not None}
    arrays.update({"market_" + item.name: getattr(result.market, item.name) for item in fields(result.market)})
    if set(omit) - set(arrays):
        raise ValueError("cannot omit unknown result arrays")
    arrays = {name: value for name, value in arrays.items() if name not in omit}
    if extra:
        if set(extra) & set(arrays):
            raise ValueError("extra arrays may not replace result arrays")
        arrays.update(extra)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_result(path: str | Path) -> DisclosureResult:
    with np.load(path, allow_pickle=False) as arrays:
        market = {item.name: arrays["market_" + item.name].copy() for item in fields(CrossSectionResult)}
        market["burn_in_days"] = int(market["burn_in_days"])
        values = {item.name: arrays[item.name].copy() for item in fields(DisclosureResult)
                  if item.name != "market" and (item.name in arrays or item.default is MISSING)}
    return DisclosureResult(market=CrossSectionResult(**market), **values)


def load_policy(path: str | Path, domain: DomainSpec) -> PolicySchedule:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    baseline = PolicyRegime(**payload["baseline"]) if "baseline" in payload else PolicyRegime.baseline(domain)
    intervention = PolicyRegime(**payload["intervention"]) if payload.get("intervention") is not None else None
    return PolicySchedule(baseline, intervention, payload.get("start_day", 0))


def learning_report(
    result: DisclosureResult, config: MarketConfig, learner: CompanyQLearner | None, *, training: bool,
    coverage_target: int = 0,
) -> dict[str, Any]:
    visits = np.zeros((6, 81), dtype=np.int64)
    kinds = action_kinds(2 * len(config.firm_action_fractions) + 1)
    legal_visits = np.zeros((6, 81, len(kinds)), dtype=np.int64)
    groups, states = result.learning_groups[1:], result.learning_states[1:]
    np.add.at(visits, (groups.ravel(), states.ravel()), 1)
    for action in range(len(kinds)):
        np.add.at(legal_visits[:, :, action], (groups.ravel(), states.ravel()),
                  result.valid_actions[1:, :, action].ravel().astype(np.int64))
    decisions = int(visits.sum())
    unseen = None if result.unvisited_state is None else int(result.unvisited_state[1:].sum())
    fraction = unseen / decisions if unseen is not None and decisions else None
    undercovered, group_undercovered = [], []
    group_totals = visits.sum(axis=1)
    if learner is not None:
        common = (visits > 0) & (visits >= .01 * decisions)
        for group, state, action in np.argwhere(common[:, :, None] & (legal_visits > 0) & (learner.counts < 20)):
            undercovered.append({"group": int(group), "state": int(state),
                                 "choice_index": int(action), "action": int(kinds[action]),
                                 "count": int(learner.counts[group, state, action])})
        group_common = (visits > 0) & (visits >= .01 * group_totals[:, None])
        for group, state, action in np.argwhere(group_common[:, :, None] & (legal_visits > 0) & (learner.counts < 20)):
            group_undercovered.append({"group": int(group), "state": int(state),
                                       "choice_index": int(action), "action": int(kinds[action]),
                                       "count": int(learner.counts[group, state, action])})
    group_unseen = None
    if result.unvisited_state is not None:
        group_unseen = np.zeros(6, dtype=np.int64)
        np.add.at(group_unseen, groups.ravel(), result.unvisited_state[1:].ravel().astype(np.int64))
    group_reports = []
    for group, total in enumerate(group_totals):
        missing = None if group_unseen is None else int(group_unseen[group])
        share = missing / int(total) if missing is not None and total else None
        status = "not_observed" if not total else (
            "not_applicable" if learner is None or training else (
                "PASS" if share is not None and share <= .05
                and not any(item["group"] == group for item in group_undercovered) else "FAIL"
            )
        )
        group_reports.append({"group": group, "size_class": group % 3, "policy_epoch": group // 3,
                              "decisions": int(total), "unseen_state_decisions": missing,
                              "unseen_state_fraction": share, "status": status})
    global_status = "not_applicable" if learner is None or training else (
        "PASS" if fraction is not None and fraction <= .05 and not undercovered else "FAIL"
    )
    coverage_status = "FAIL" if any(group["status"] == "FAIL" for group in group_reports) else global_status
    period = config.firm_decision_interval
    last = config.trading_days
    last_decision = last // period * period
    ledger = {
        "financing_adjustment": result.financing_value_adjustment,
        "resource_cost": result.applied_resource_cost,
        "fine": result.applied_fine,
        "subsidy": result.applied_subsidy,
    }
    net = ledger["financing_adjustment"] - ledger["resource_cost"] - ledger["fine"] + ledger["subsidy"]
    discount = np.exp(-config.firm_discount_rate * np.arange(last + 1) / TRADING_DAYS_PER_YEAR)
    pending = []
    for decision, firm in np.argwhere(result.action_due_day > last):
        pending.append({"decision_day": int(decision * period), "firm": int(firm),
                        "action": int(result.actions[decision, firm]),
                        "due_day": int(result.action_due_day[decision, firm]),
                        "amount": float(result.amounts[decision, firm])})
    unapplied = {}
    for name, fractions in (("resource_cost", result.cost_fraction), ("fine", result.penalty_fraction),
                            ("subsidy", result.subsidy_fraction)):
        amount = fractions[-1] * result.market.prices[-1] if last_decision == last else np.zeros(config.n_firms)
        unapplied[name] = amount.tolist()
    report = {
        "firm_policy": config.firm_policy, "training": training,
        "coverage": {
            "visits": visits.tolist(), "legal_action_visits": legal_visits.tolist(),
            "unseen_state_decisions": unseen, "decisions": decisions, "unseen_state_fraction": fraction,
            "common_state_minimum_action_count": 20, "common_state_share_threshold": .01,
            "common_state_undercovered_actions": undercovered,
            "group_common_state_undercovered_actions": group_undercovered,
            "group_common_state_share_threshold": .01, "maximum_unseen_state_fraction": .05,
            "groups": group_reports, "global_status": global_status, "status": coverage_status,
        },
        "ledger": {
            "scope": "Full observed window; private financing-baseline differences, not causal effects or lifetime profit.",
            "totals_by_firm": {key: value.sum(axis=0).tolist() for key, value in ledger.items()},
            "discounted_net_by_initial_cap": (discount @ net / result.market.prices[0]).tolist(),
            "completed_transitions": int(result.learning_transition.sum()),
        },
        "tail": {
            "last_decision_day": last_decision if last_decision else None,
            "unclosed_decisions": config.n_firms if last_decision else 0,
            "observed_days_of_unclosed_window": last - last_decision if last_decision else 0,
            "unapplied_next_day_commitments": unapplied,
            "pending_projects": pending,
            "outstanding_inflation": result.inflation[-1].tolist(),
        },
    }
    updates = result.learning_update
    if updates is None:
        updates = result.learning_transition if training else np.zeros_like(result.learning_transition)
    else:
        report["conditional_deployment"] = {
            "scope": "Frozen baseline greedy prefix followed by a separately trained conditional strategy; not from-zero online policy learning.",
            "updated_transition_samples": int(updates.sum()),
            "excluded_completed_transition_samples": int((result.learning_transition & ~updates).sum()),
        }
    if coverage_target:
        forced = np.zeros_like(result.learning_transition) if result.coverage_forced is None else result.coverage_forced
        completed_forced = int((forced[:-1] & updates[1:]).sum())
        report["quota_exploration"] = {
            "target": coverage_target, "forced_decisions": int(forced.sum()),
            "completed_forced_transitions": completed_forced,
            "unsettled_forced_decisions": int(forced.sum()) - completed_forced,
            "updated_transition_samples": int(updates.sum()),
            "flag_definition": "Quota branch executed, including choices equal to the unforced sample.",
        }
    return report


def run_and_archive(
    config: MarketConfig,
    domain: DomainSpec,
    output: str | Path,
    policy: PolicySchedule | None = None,
    *,
    purpose: str = "development",
    learner: CompanyQLearner | None = None,
    training: bool = False,
    learning_metadata: dict[str, Any] | None = None,
    coverage_target: int = 0,
    common_prepolicy_sha256: str | None = None,
    archive: str = "full",
) -> dict[str, Any]:
    """Save inputs and source before running; completion requires a result hash.

    ``archive="compact"`` leaves COMPACT_OMITTED out of result.npz; such archives cannot be loaded
    with load_result, but metrics, validation and the result fingerprint are complete.
    ``archive="sampled"`` is compact plus the arrays of ``sampled_arrays``.
    """
    if archive not in ("full", "compact", "sampled"):
        raise ValueError("archive must be 'full', 'compact' or 'sampled'")
    if type(coverage_target) is not int or coverage_target < 0:
        raise ValueError("coverage_target must be a nonnegative integer")
    policy = policy or PolicySchedule.constant(domain)
    common_options = {"common_prepolicy_sha256": common_prepolicy_sha256} if common_prepolicy_sha256 is not None else {}
    learning = None
    if config.firm_policy == "q_learning":
        if learner is None or learning_metadata is None:
            raise ValueError("Q-learning requires an explicit learner and frozen learning metadata")
        if learning_metadata.get("contract") != learning_contract(config, domain, policy, coverage_target=coverage_target, **common_options):
            raise ValueError("learning contract mismatch")
        if (learner.values.shape != (6, 81, 2 * len(config.firm_action_fractions) + 1)
                or learner.counts.shape != learner.values.shape):
            raise ValueError("archived Q-learning requires six size/regime groups")
        if training != (learner.values.flags.writeable and learner.counts.flags.writeable):
            raise ValueError("training requires writable tables; evaluation requires frozen tables")
        if not training and (learner.values.flags.writeable or learner.counts.flags.writeable or learner.epsilon != 0):
            raise ValueError("evaluation requires frozen tables and zero epsilon")
        if common_prepolicy_sha256 is not None:
            origin = learning_metadata.get("common_prepolicy", {})
            if not origin.get("model_path"):
                raise ValueError("common pre-policy deployment requires its source model")
            source, expected_origin = load_common_prepolicy(origin["model_path"], common_prepolicy_sha256,
                                                           config, domain, policy, coverage_target=coverage_target)
            if (origin != expected_origin or table_fingerprint(learner, prefix_groups=3) != origin["table_fingerprint"]
                    or learner.learning_rate != source.learning_rate or learner.discount != source.discount
                    or learner.learning_rate_power != source.learning_rate_power):
                raise ValueError("common pre-policy table or provenance mismatch")
            used = set(origin["training_seeds"] + origin["validation_seeds"])
            current = learning_metadata.get("budget", {})
            if (config.seed in used | set(origin["reserved_seeds"])
                    or used & set(current.get("training_seeds", []) + current.get("validation_seeds", []) + current.get("reserved_seeds", []))
                    or set(origin["reserved_seeds"]) & set(current.get("training_seeds", []) + current.get("validation_seeds", []))):
                raise ValueError("common pre-policy seed lineage overlaps the current budget")
        learning = {"training": training, "metadata": learning_metadata,
                    "initial_table_fingerprint": table_fingerprint(learner), "epsilon": learner.epsilon,
                    "learning_rate": learner.learning_rate, "discount": learner.discount}
        if learner.learning_rate_power:
            learning["learning_rate_power"] = learner.learning_rate_power
        if coverage_target:
            learning["coverage_target"] = coverage_target
        if common_prepolicy_sha256 is not None:
            learning.update(common_options)
            learning["prepolicy_protocol"] = "frozen_baseline_greedy_prefix_post_zero_v1"
    elif learner is not None or training or learning_metadata is not None or coverage_target or common_prepolicy_sha256 is not None:
        raise ValueError("learning arguments require the q_learning firm policy")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    identity = run_identity(config, domain, policy, learning=learning)
    manifest = {**identity, "purpose": purpose, "status": "running"}
    write_json(destination / "manifest.json", manifest)
    started = time.perf_counter()
    market = None
    try:
        model_artifacts = []
        if learner is not None:
            save_frozen(destination / "input_model.npz", learner, learning_metadata)
            model_artifacts.append("input_model.npz")
        package_parent = Path(__file__).resolve().parents[2]
        with zipfile.ZipFile(destination / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative in identity["source_hashes"]:
                bundle.write(package_parent / relative, relative)
        if source_hashes() != identity["source_hashes"]:
            raise RuntimeError("source changed while archiving the run")
        market = DisclosureMarket(config, domain, policy, learner=learner, training=training,
                                  coverage_target=coverage_target, **common_options)
        result = market.run()
        if learner is not None:
            final_fingerprint = table_fingerprint(learner)
            if common_prepolicy_sha256 is not None and (table_fingerprint(learner, prefix_groups=3) != origin["table_fingerprint"]
                                                        or sha256_file(origin["model_path"]) != common_prepolicy_sha256):
                raise RuntimeError("conditional deployment changed the frozen pre-policy table")
            if not training and final_fingerprint != learning["initial_table_fingerprint"]:
                raise RuntimeError("evaluation changed the frozen learning table")
            save_frozen(destination / "output_model.npz", learner, learning_metadata)
            model_artifacts.append("output_model.npz")
        if config.firm_policy != "npv":
            report = learning_report(result, config, learner, training=training, coverage_target=coverage_target)
            if learner is not None:
                report.update({"initial_table_fingerprint": learning["initial_table_fingerprint"],
                               "final_table_fingerprint": final_fingerprint,
                               "final_transition_count": int(learner.counts.sum())})
            write_json(destination / "learning.json", report)
            model_artifacts.append("learning.json")
        metrics = compute_metrics(result, config, domain)
        validation = validate_market(result.market, config)
        if source_hashes() != identity["source_hashes"]:
            raise RuntimeError("source changed during the run")
        omitted = compact_omitted(result) if archive in ("compact", "sampled") else ()
        extra = sampled_arrays(result) if archive == "sampled" else None
        save_result(destination / "result.npz", result, omit=omitted, extra=extra)
        write_json(destination / "metrics.json", metrics)
        write_json(destination / "validation.json", validation)
        if archive in ("compact", "sampled"):
            manifest["compact_archive"] = {"omitted_result_arrays": list(omitted),
                                           "note": "The result fingerprint covers the full result; replay to regenerate."}
        if archive == "sampled":
            manifest["sampled_archive"] = {"start_day": SAMPLE_START_DAY, "interval_days": SAMPLE_INTERVAL_DAYS,
                                           "days": len(extra["sampled_days"]),
                                           "arrays": sorted(name for name in extra if name != "sampled_days")}
        manifest.update({
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "result_fingerprint": result.fingerprint(),
            "artifacts": {name: sha256_file(destination / name)
                          for name in ("source.zip", "result.npz", "metrics.json", "validation.json", *model_artifacts)},
        })
        write_json(destination / "manifest.json", manifest)
    except Exception as error:
        if learner is not None and market is not None:
            try:
                digest = save_frozen(destination / "failed_model.npz", learner, learning_metadata)
                manifest["failed_learning_checkpoint"] = {
                    "path": "failed_model.npz", "sha256": digest,
                    "last_prepared_day": market.signal_day,
                    "table_fingerprint": table_fingerprint(learner),
                    "resumable": False,
                }
            except Exception as checkpoint_error:
                manifest["failed_learning_checkpoint"] = {
                    "error": str(checkpoint_error), "resumable": False,
                    "last_prepared_day": market.signal_day,
                }
        write_json(destination / "manifest.json", {
            **manifest, "status": "failed", "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
    render_run(destination)
    return manifest


def render_run(output: str | Path) -> Path:
    """Rebuild a report from verified archived artifacts, without simulating."""
    destination = Path(output)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("only a completed run can be rendered")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(destination / name) != digest:
            raise ValueError(f"archive hash mismatch: {name}")
    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((destination / "validation.json").read_text(encoding="utf-8"))
    lines = [
        "# Disclosure-market run", "", f"Purpose: {manifest['purpose']}",
        f"Domain: {manifest['domain']['name']}", f"Seed: {manifest['config']['seed']}",
        f"Run ID: `{manifest['run_id']}`", f"Result fingerprint: `{manifest['result_fingerprint']}`", "",
        "A completed simulation is not evidence of empirical calibration or a policy ranking.", "",
        "## Metrics", "", "```json", json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), "```", "",
        "## Validation", "", "```json", json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), "```", "",
    ]
    if "learning.json" in manifest["artifacts"]:
        report = json.loads((destination / "learning.json").read_text(encoding="utf-8"))
        coverage = report["coverage"]
        lines.extend([
            "## Learning diagnostics", "", f"Firm policy: {report['firm_policy']}",
            f"Training: {report['training']}", f"Coverage status: {coverage['status']}",
            f"Unseen state share: {coverage['unseen_state_fraction']}",
            f"Unclosed decisions: {report['tail']['unclosed_decisions']}",
            "Ledger, coverage counts and pending liabilities: [learning.json](learning.json).", "",
        ])
    report = destination / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--domain", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--model", type=Path, help="Frozen Q table for evaluation")
    parser.add_argument("--model-sha256")
    parser.add_argument("--coverage-target", type=int, default=0)
    parser.add_argument("--common-prepolicy-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    if args.render_only:
        print(render_run(args.output))
        return 0
    if args.domain is None:
        parser.error("--domain is required for a new run")
    disable_power_throttling()
    config = load_market_config(args.config) if args.config else MarketConfig()
    domain = load_domain(args.domain)
    policy = load_policy(args.policy, domain) if args.policy else None
    if args.model_sha256 and args.model is None:
        parser.error("--model-sha256 requires --model")
    learner, metadata = load_frozen(args.model, expected_sha256=args.model_sha256) if args.model else (None, None)
    result = run_and_archive(config, domain, args.output, policy, learner=learner, learning_metadata=metadata,
                             coverage_target=args.coverage_target,
                             **({"common_prepolicy_sha256": args.common_prepolicy_sha256} if args.common_prepolicy_sha256 is not None else {}))
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
