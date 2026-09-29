"""Paired development diagnostics for the Stage 1 position and depth bases."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import STRATEGY_NAMES, Stage1Config, load_stage1_config
from .harness import MarketHarness
from .mechanism_validation import MechanismProtocol, build_scenarios, result_metrics
from .runner import resolve_code_revision
from .population import VALUE_STRATEGY


def diagnostic_scenarios(config: Stage1Config) -> dict[str, Stage1Config]:
    legacy = replace(
        config, reference_position_basis="shares", liquidity_depth_basis="shares"
    )
    candidate = replace(
        config, reference_position_basis="notional", liquidity_depth_basis="notional"
    )
    return {
        "legacy": legacy,
        "reference_notional": replace(legacy, reference_position_basis="notional"),
        "depth_notional": replace(legacy, liquidity_depth_basis="notional"),
        "candidate": candidate,
        "legacy_fixed_liquidity": build_scenarios(legacy)["fixed_liquidity"],
        "candidate_fixed_liquidity": build_scenarios(candidate)["fixed_liquidity"],
    }


class _ObservedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.row: dict[str, float] = {}

    @property
    def diagnostics(self):
        return self.policy.diagnostics

    def act(self, observation, population, rng):
        config = self.policy.config
        center = population.reference_positions
        if config.reference_position_basis == "notional":
            center = center * config.initial_price / observation.price
        limit = np.minimum(
            config.target_position_fraction * population.reference_wealth
            / observation.price,
            config.position_cap,
        )
        ratio = center / limit
        self.row = {
            "reference_center_mean": float(np.mean(center)),
            "position_limit_mean": float(np.mean(limit)),
            "center_to_limit_minimum": float(np.min(ratio)),
            "center_to_limit_mean": float(np.mean(ratio)),
            "center_to_limit_maximum": float(np.max(ratio)),
            "center_to_supply_cap_maximum": float(np.max(center / config.position_cap)),
            "realized_volatility": observation.realized_volatility,
            "realized_volatility_reference": observation.realized_volatility_reference,
        }
        submitted = self.policy.act(observation, population, rng)
        gross = float(np.abs(submitted).sum())
        self.row.update(
            submitted_gross_shares=gross,
            submitted_gross_notional=gross * observation.price,
            submitted_net_notional=float(submitted.sum()) * observation.price,
        )
        return submitted


def _run_diagnostic(
    scenario: str, config: Stage1Config, output: Path
) -> dict[str, object]:
    harness = MarketHarness(config)
    observer = _ObservedPolicy(harness.policy)
    harness.policy = observer
    cap_events = []
    maximum_error = 0.0
    daily_path = output / "daily" / f"{scenario}_{config.seed}.csv"
    with daily_path.open("x", encoding="utf-8", newline="") as destination:
        writer = None
        for day_index in range(config.trading_days):
            audit = harness.step(day_index)
            raw = audit.permanent_impact + audit.transient_impact_change
            adjustment = audit.demand_log_return - raw
            total = float(np.log(audit.mid_price_after / audit.price_before))
            error = total - (audit.public_news_impact + raw + adjustment)
            pressure = observer.row["submitted_gross_shares"] / audit.depth
            row = {
                "day": audit.day,
                "evaluation_day": audit.day > config.burn_in_days,
                "price_before": audit.price_before,
                "price_after": audit.mid_price_after,
                **observer.row,
                "depth_shares": audit.depth,
                "depth_notional": audit.depth * audit.price_before,
                "participation_pressure": pressure,
                "capped_participation": min(pressure, 3.0),
                "amplified_participation": min(pressure, 3.0) ** config.participation_pressure_exponent,
                "public_news_impact": audit.public_news_impact,
                "permanent_impact": audit.permanent_impact,
                "transient_impact": audit.transient_impact,
                "transient_impact_change": audit.transient_impact_change,
                "raw_demand_log_return": raw,
                "price_cap_adjustment": adjustment,
                "demand_log_return": audit.demand_log_return,
                "total_log_return": total,
                "price_decomposition_error": error,
                "price_cap_hit": audit.price_cap_hit,
                "cash_error": audit.cash_error,
                "share_error": audit.share_error,
            }
            if not all(np.isfinite(value) for value in row.values()):
                raise RuntimeError(f"nonfinite diagnostic on day {audit.day}")
            if audit.mid_price_after <= 0 or abs(error) > 1e-12:
                raise RuntimeError(f"price decomposition failed on day {audit.day}: {error}")
            maximum_error = max(maximum_error, abs(error))
            if writer is None:
                writer = csv.DictWriter(destination, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if audit.price_cap_hit:
                cap_events.append(row)
    result = harness.run()
    evaluation = result.daily_audits[config.burn_in_days:]
    news = np.array([audit.public_news_impact for audit in evaluation])
    demand = np.array([audit.demand_log_return for audit in evaluation])
    total = np.diff(np.log(result.prices[config.burn_in_days:]))
    news_variance = float(np.var(news, ddof=0))
    demand_variance = float(np.var(demand, ddof=0))
    covariance = float(np.mean((news - news.mean()) * (demand - demand.mean())))
    total_variance = float(np.var(total, ddof=0))
    variance_error = total_variance - (news_variance + demand_variance + 2 * covariance)
    if not np.isfinite(variance_error) or abs(variance_error) > 1e-12:
        raise RuntimeError(f"variance decomposition failed: {variance_error}")
    metrics = result_metrics(result, config.burn_in_days)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("nonfinite result metrics")
    return {
        "scenario": scenario,
        "seed": config.seed,
        "metrics": metrics,
        "variance_decomposition": {
            "ddof": 0,
            "evaluation_days": len(evaluation),
            "news_variance": news_variance,
            "demand_variance": demand_variance,
            "news_demand_covariance": covariance,
            "total_variance": total_variance,
            "reconstruction_error": variance_error,
        },
        "maximum_price_decomposition_error": maximum_error,
        "cap_events": cap_events,
        "fingerprint": result.fingerprint(),
        "daily_file": str(daily_path.relative_to(output)),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_diagnostics(
    config: Stage1Config,
    protocol: MechanismProtocol,
    output: Path,
    *,
    workers: int = 4,
) -> dict[str, object]:
    if protocol.stage != "development":
        raise ValueError("diagnostics require a development protocol")
    if workers < 1:
        raise ValueError("workers must be positive")
    if config.burn_in_days < protocol.minimum_burn_in_days:
        raise ValueError("config burn-in is below protocol minimum")
    if config.trading_days - config.burn_in_days < protocol.minimum_evaluation_days:
        raise ValueError("config evaluation days are below protocol minimum")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "daily").mkdir()
    scenarios = diagnostic_scenarios(config)
    tasks = [
        (scenario, replace(scenario_config, seed=seed), output)
        for scenario, scenario_config in scenarios.items()
        for seed in protocol.seeds
    ]
    _write_json(output / "manifest.json", {
        "config": config.to_dict(),
        "protocol": asdict(protocol),
        "scenario_configs": {name: value.to_dict() for name, value in scenarios.items()},
        "code_revision": resolve_code_revision(),
        "decision": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "stage1_complete": False,
        "may_enter_stage2": False,
    })
    results: dict[tuple[str, int], dict[str, object]] = {}
    checkpoint = {
        "status": "RUNNING", "expected_runs": len(tasks),
        "completed_runs": [], "errors": [],
        "unfinished_runs": [
            {"scenario": name, "seed": task_config.seed}
            for name, task_config, _ in tasks
        ],
    }
    _write_json(output / "checkpoint.json", checkpoint)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_diagnostic, *task): (task[0], task[1].seed)
            for task in tasks
        }
        try:
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                _write_json(output / f"{key[0]}_{key[1]}.json", result)
                results[key] = result
                checkpoint["completed_runs"] = [
                    {"scenario": name, "seed": seed} for name, seed in sorted(results)
                ]
                checkpoint["unfinished_runs"] = [
                    {"scenario": name, "seed": task_config.seed}
                    for name, task_config, _ in tasks
                    if (name, task_config.seed) not in results
                ]
                _write_json(output / "checkpoint.json", checkpoint)
        except Exception as error:
            checkpoint["status"] = "ENGINEERING_ERROR"
            checkpoint["errors"] = [{
                "scenario": key[0], "seed": key[1],
                "type": type(error).__name__, "message": str(error),
            }]
            checkpoint["unfinished_runs"] = [
                {"scenario": name, "seed": task_config.seed}
                for name, task_config, _ in tasks
                if (name, task_config.seed) not in results
            ]
            _write_json(output / "checkpoint.json", checkpoint)
            for pending in futures:
                pending.cancel()
            raise
    ordered = [results[(name, task_config.seed)] for name, task_config, _ in tasks]
    report = {
        "decision": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "stage1_complete": False,
        "may_enter_stage2": False,
        "runs": ordered,
    }
    _write_json(output / "diagnostics.json", report)
    with (output / "per_seed_metrics.csv").open("x", encoding="utf-8", newline="") as destination:
        rows = [{
            "scenario": run["scenario"], "seed": run["seed"],
            **run["metrics"], **run["variance_decomposition"],
            "maximum_price_decomposition_error": run["maximum_price_decomposition_error"],
        } for run in ordered]
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint["status"] = "COMPLETE"
    _write_json(output / "checkpoint.json", checkpoint)
    return report


def reconstruct_price_gap(rows, config: Stage1Config) -> dict[str, object]:
    alpha = config.public_news_price_pass_through
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("price-gap reconstruction requires positive news pass-through")
    if len(rows) != config.trading_days or config.burn_in_days >= len(rows):
        raise ValueError("daily rows do not match the configured evaluation window")
    initial_gap = float(np.log(config.initial_price / config.initial_fundamental))
    initial_transient = float(rows[0]["transient_impact"]) - float(rows[0]["transient_impact_change"])
    previous_price = config.initial_price
    previous_transient = initial_transient
    news_sum = permanent_sum = cap_sum = fundamental_news_sum = 0.0
    daily = []
    for day, row in enumerate(rows, 1):
        if int(row["day"]) != day:
            raise ValueError("daily rows must start at day one and be contiguous")
        opening, closing = float(row["price_before"]), float(row["price_after"])
        news = float(row["public_news_impact"])
        permanent = float(row["permanent_impact"])
        transient = float(row["transient_impact"])
        transient_change = float(row["transient_impact_change"])
        cap = float(row["price_cap_adjustment"])
        if not all(np.isfinite(value) for value in (opening, closing, news, permanent, transient, transient_change, cap)) or min(opening, closing) <= 0:
            raise ValueError(f"invalid daily values on day {day}")
        if not np.isclose(opening, previous_price, rtol=1e-12, atol=1e-12):
            raise ValueError(f"discontinuous price path on day {day}")
        if abs(transient - previous_transient - transient_change) > 1e-12:
            raise ValueError(f"discontinuous transient impact on day {day}")
        if abs(np.log(closing / opening) - (news + permanent + transient_change + cap)) > 1e-12:
            raise ValueError(f"price decomposition failed on day {day}")
        fundamental_news = news / alpha
        fundamental_news_sum += fundamental_news
        news_sum += (alpha - 1.0) * fundamental_news
        permanent_sum += permanent
        cap_sum += cap
        gap = float(np.log(closing / config.initial_fundamental) - fundamental_news_sum)
        reconstructed = initial_gap + news_sum + permanent_sum + transient - initial_transient + cap_sum
        if abs(gap - reconstructed) > 1e-11:
            raise ValueError(f"gap reconstruction failed on day {day}")
        daily.append({
            "day": day, "evaluation_day": day > config.burn_in_days,
            "price_before": opening, "price_after": closing,
            "fundamental_news_return": fundamental_news,
            "fundamental_reconstructed": float(config.initial_fundamental * np.exp(fundamental_news_sum)),
            "initial_log_price_gap": initial_gap,
            "news_gap_cumulative": news_sum,
            "permanent_impact_cumulative": permanent_sum,
            "transient_impact_change_cumulative": transient - initial_transient,
            "price_cap_adjustment_cumulative": cap_sum,
            "signed_log_price_gap": gap,
            "reconstructed_log_price_gap": reconstructed,
            "gap_reconstruction_error": gap - reconstructed,
        })
        previous_price, previous_transient = closing, transient
    evaluation = daily[config.burn_in_days:]
    gaps = np.array([row["signed_log_price_gap"] for row in evaluation])
    summary = {
        "evaluation_days": len(evaluation), "initial_log_price_gap": initial_gap,
        "initial_transient_impact": initial_transient,
        "g_burn_in": daily[config.burn_in_days - 1]["signed_log_price_gap"] if config.burn_in_days else initial_gap,
        "g1000": daily[999]["signed_log_price_gap"] if len(daily) >= 1000 else None,
        "mean_signed_log_price_gap": float(gaps.mean()),
        "mean_absolute_log_price_gap": float(np.abs(gaps).mean()),
        "signed_gap_quantiles": {str(q): float(np.quantile(gaps, q)) for q in (.05, .25, .5, .75, .95)},
        "maximum_gap_reconstruction_error": max(abs(row["gap_reconstruction_error"]) for row in daily),
    }
    for name, mask in (("positive", gaps > .1), ("negative", gaps < -.1)):
        longest = current = 0
        for flag in mask:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        summary[f"{name}_gap_over_0_1_fraction"] = float(mask.mean())
        summary[f"longest_{name}_gap_over_0_1_run"] = longest
    summary["components"] = {
        name: {"evaluation_mean": float(np.mean([row[name] for row in evaluation])),
               "burn_in": daily[config.burn_in_days - 1][name] if config.burn_in_days else 0.0,
               "final": daily[-1][name]}
        for name in ("news_gap_cumulative", "permanent_impact_cumulative", "transient_impact_change_cumulative", "price_cap_adjustment_cumulative")
    }
    return {"daily": daily, "summary": summary}


def reconstruct_price_gaps(source: Path, output: Path) -> dict[str, object]:
    source, output = Path(source), Path(output)
    prior = source / "diagnostics"
    manifest = json.loads((prior / "manifest.json").read_text(encoding="utf-8"))
    source_config = Stage1Config.from_dict(manifest["config"])
    scenarios = diagnostic_scenarios(source_config)
    if {name: Stage1Config.from_dict(value).to_dict() for name, value in manifest["scenario_configs"].items()} != {name: config.to_dict() for name, config in scenarios.items()}:
        raise ValueError("source diagnostic scenarios do not match their configuration")
    seeds = manifest["protocol"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("source seeds must be nonempty and unique")
    with (prior / "per_seed_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    metric_keys = [(row["scenario"], int(row["seed"])) for row in metric_rows]
    expected_keys = {(name, seed) for name in scenarios for seed in seeds}
    if len(set(metric_keys)) != len(metric_keys) or set(metric_keys) != expected_keys:
        raise ValueError("source metric scenarios and seeds are missing, duplicated or incompatible")
    csv_metrics = dict(zip(metric_keys, metric_rows))
    target = output / "reconstruction"
    target.mkdir(parents=True, exist_ok=False)
    summaries = []
    for scenario, config in scenarios.items():
        for seed in seeds:
            result = json.loads((prior / f"{scenario}_{seed}.json").read_text(encoding="utf-8"))
            if result["scenario"] != scenario or result["seed"] != seed:
                raise ValueError("source result scenario or seed mismatch")
            if any(float(csv_metrics[(scenario, seed)][name]) != value for name, value in result["metrics"].items()):
                raise ValueError(f"source JSON and CSV metrics differ: {scenario}/{seed}")
            with (prior / result["daily_file"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            reconstructed = reconstruct_price_gap(rows, replace(config, seed=seed))
            summary = reconstructed["summary"]
            if not np.isclose(summary["mean_absolute_log_price_gap"], result["metrics"]["mean_absolute_log_price_gap"], rtol=0, atol=1e-11):
                raise ValueError(f"source mean absolute price gap mismatch: {scenario}/{seed}")
            path = target / f"{scenario}_{seed}.csv"
            with path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(reconstructed["daily"][0]))
                writer.writeheader()
                writer.writerows(reconstructed["daily"])
            summaries.append({"scenario": scenario, "seed": seed, **summary,
                              "daily_file": str(path.relative_to(output))})
    indexed = {(row["scenario"], row["seed"]): row for row in summaries}
    comparisons = []
    for left, right in (("reference_notional", "legacy"), ("depth_notional", "legacy"),
                        ("candidate", "legacy"), ("candidate", "reference_notional"),
                        ("candidate", "depth_notional"),
                        ("candidate_fixed_liquidity", "legacy_fixed_liquidity"),
                        ("legacy_fixed_liquidity", "legacy"),
                        ("candidate_fixed_liquidity", "candidate")):
        for seed in seeds:
            comparisons.append({
                "left": left, "right": right, "seed": seed,
                **{f"delta_{metric}": indexed[(left, seed)][metric] - indexed[(right, seed)][metric]
                   for metric in ("g_burn_in", "mean_signed_log_price_gap", "mean_absolute_log_price_gap")},
            })
    report = {"runs": summaries, "paired_comparisons": comparisons}
    _write_json(target / "summary.json", report)
    return report


def _strategy_orders(orders, strategies, price):
    rows = {}
    for code, name in enumerate(STRATEGY_NAMES):
        selected = orders[strategies == code]
        buy = float(np.maximum(selected, 0).sum())
        sell = float(-np.minimum(selected, 0).sum())
        rows[name] = {"buy_shares": buy, "sell_shares": sell,
                      "net_shares": float(selected.sum()),
                      "buy_notional": buy * price, "sell_notional": sell * price,
                      "net_notional": float(selected.sum()) * price}
    return rows


class _PriceDiscoveryPolicy(_ObservedPolicy):
    def __init__(self, policy, *, probes_enabled: bool = True):
        super().__init__(policy)
        self.day = 0
        self.probes = []
        self.probes_enabled = probes_enabled

    def act(self, observation, population, rng):
        self.day += 1
        probe_day = self.probes_enabled and self.day in (1, 1001, 1501, 2001, 2501, 3001, 3500)
        if probe_day:
            snapshot = deepcopy((self.policy, population, observation, rng))
        self.before_positions = population.positions.copy()
        before_desired = population.desired_positions.copy()
        before_cash = population.cash.copy()
        self.strategies = population.strategies.copy()
        self.submitted = super().act(observation, population, rng)
        config, price = self.policy.config, observation.price
        limit = np.minimum(config.target_position_fraction * population.reference_wealth / price, config.position_cap)
        minimum = np.maximum(-config.max_short_leverage * (before_cash + self.before_positions * price)
                             / (price * np.exp(config.max_log_return + 4 * config.fundamental_volatility)), -config.position_cap)
        self.minimum_positions = minimum
        affordable = before_cash / (price * np.exp(config.max_log_return) * (1 + config.transaction_cost_rate))
        relative_mispricing = (population.subjective_values - price) / price
        band = (self.strategies == VALUE_STRATEGY) & (np.abs(relative_mispricing) < config.value_no_trade_band)
        subjective_error = np.log(population.subjective_values / observation.fundamental_value)
        submitted_by_strategy = _strategy_orders(self.submitted, self.strategies, price)
        remaining = population.desired_positions - self.before_positions
        tolerance = 1e-9
        adjustment_rates = np.array([config.value_target_adjustment, config.trend_target_adjustment, config.noise_target_adjustment])[self.strategies]
        raw_target = before_desired.copy()
        np.divide(population.desired_positions - before_desired, adjustment_rates,
                  out=raw_target, where=adjustment_rates > 0)
        raw_target[adjustment_rates > 0] += before_desired[adjustment_rates > 0]
        positive_value_signal = (self.strategies == VALUE_STRATEGY) & (relative_mispricing > 0) & ~band & (config.value_sensitivity > 0)
        capacity_counts = {
            "target_reached_count": np.abs(remaining) <= tolerance,
            "raw_target_at_position_limit_count": (adjustment_rates > 0) & (np.abs(np.abs(raw_target) - limit) <= tolerance),
            "cash_capacity_binding_count": (remaining > affordable + tolerance) & (np.abs(self.submitted - affordable) <= tolerance),
            "sell_capacity_binding_count": (remaining < minimum - self.before_positions - tolerance) & (np.abs(self.submitted - (minimum - self.before_positions)) <= tolerance),
            "positive_value_signal_nonpositive_order_count": positive_value_signal & (self.submitted <= 0),
            "undervalued_value_nonpositive_order_count": (self.strategies == VALUE_STRATEGY) & (relative_mispricing > 0) & (self.submitted <= 0),
            "value_in_band_nonzero_order_count": band & (np.abs(self.submitted) > tolerance),
        }
        for code, name in enumerate(STRATEGY_NAMES):
            mask = self.strategies == code
            count = int(mask.sum())
            fields = {
                "agent_count": count,
                "actual_position_before_shares": float(self.before_positions[mask].sum()),
                "desired_position_before_shares": float(before_desired[mask].sum()),
                "desired_position_after_shares": float(population.desired_positions[mask].sum()),
                "remaining_target_shares": float((population.desired_positions - self.before_positions)[mask].sum()),
                "remaining_target_absolute_shares": float(np.abs(population.desired_positions - self.before_positions)[mask].sum()),
                "position_limit_shares": float(limit[mask].sum()),
                "minimum_position_shares": float(minimum[mask].sum()),
                "buy_position_capacity_shares": float(np.maximum(limit - self.before_positions, 0)[mask].sum()),
                "sell_position_capacity_shares": float(np.maximum(self.before_positions - minimum, 0)[mask].sum()),
                "cash_before": float(before_cash[mask].sum()),
                "affordable_buy_capacity_shares": float(affordable[mask].sum()),
                "mean_subjective_log_error_after_information": float(subjective_error[mask].mean()) if count else 0.0,
                "mean_absolute_subjective_log_error_after_information": float(np.abs(subjective_error[mask]).mean()) if count else 0.0,
                "value_in_band_count": int(band[mask].sum()),
                **{key: int(flags[mask].sum()) for key, flags in capacity_counts.items()},
                **{f"submitted_{key}": value for key, value in submitted_by_strategy[name].items()},
            }
            self.row.update({f"{name}_{key}": value for key, value in fields.items()})
        if probe_day:
            copied_policy, copied_population, copied_observation, copied_rng = deepcopy(snapshot)
            baseline = copied_policy.act(copied_observation, copied_population, copied_rng)
            if not np.array_equal(baseline, self.submitted):
                raise RuntimeError(f"copied baseline orders differ on day {self.day}")
            updated_band = (copied_population.strategies == VALUE_STRATEGY) & (
                np.abs((copied_population.subjective_values - price) / price) < config.value_no_trade_band)
            interventions = {"baseline": submitted_by_strategy}
            for intervention in ("value_sensitivity_zero", "cancel_in_band_target_backlog"):
                copied_policy, copied_population, copied_observation, copied_rng = deepcopy(snapshot)
                if intervention == "value_sensitivity_zero":
                    copied_policy.config = replace(copied_policy.config, value_sensitivity=0.0)
                else:
                    copied_population.desired_positions[updated_band] = copied_population.positions[updated_band]
                orders = copied_policy.act(copied_observation, copied_population, copied_rng)
                interventions[intervention] = _strategy_orders(orders, self.strategies, price)
            for values in interventions.values():
                for name in STRATEGY_NAMES:
                    values[name]["net_notional_delta_from_baseline"] = values[name]["net_notional"] - submitted_by_strategy[name]["net_notional"]
            self.probes.append({"day": self.day, "baseline_orders_exact": True,
                                "fixed_state_local_only": True, "copies_settled": False,
                                "in_band_value_agents": int(updated_band.sum()),
                                "notional_valuation": "opening_price",
                                "interventions": interventions})
        return self.submitted


def _value_direction_fields(orders, value_signal, strategies, price, stage,
                            forced_cover_orders=None):
    voluntary = orders if forced_cover_orders is None else orders - forced_cover_orders
    value = strategies == VALUE_STRATEGY
    fields = {}
    for name, mask in (
        ("positive_signal_voluntary_sell", value & (value_signal > 0) & (voluntary < 0)),
        ("negative_signal_voluntary_buy", value & (value_signal < 0) & (voluntary > 0)),
    ):
        shares = float(np.abs(voluntary[mask]).sum())
        fields.update({f"value_{stage}_{name}_count": int(mask.sum()),
                       f"value_{stage}_{name}_shares": shares,
                       f"value_{stage}_{name}_notional": shares * price})
    for name, signal in (("positive", value_signal > 0), ("negative", value_signal < 0)):
        fields[f"value_{stage}_{name}_signal_zero_order_count"] = int((value & signal & (voluntary == 0)).sum())
    return fields


class _ValueDirectionPolicy(_PriceDiscoveryPolicy):
    def __init__(self, policy):
        policy.instrument_orders = True
        super().__init__(policy, probes_enabled=False)

    def act(self, observation, population, rng):
        submitted = super().act(observation, population, rng)
        self.order_snapshot = self.policy.order_diagnostics
        snapshot = self.order_snapshot
        if snapshot is None or not np.array_equal(snapshot.after_risk, submitted):
            raise RuntimeError("policy order snapshots do not match submitted orders")
        for stage in ("before_constraint", "after_constraint", "after_risk"):
            orders = getattr(snapshot, stage)
            forced = snapshot.forced_cover_orders if stage == "after_risk" else None
            self.row.update(_value_direction_fields(orders, snapshot.value_signal, self.strategies,
                                                   observation.price, stage, forced))
            for name, values in _strategy_orders(orders, self.strategies, observation.price).items():
                self.row.update({f"{name}_{stage}_{key}": value for key, value in values.items()})
        for code, name in enumerate(STRATEGY_NAMES):
            mask = self.strategies == code
            covers = snapshot.forced_cover_orders[mask]
            self.row.update({
                f"{name}_policy_forced_cover_count": int(np.count_nonzero(covers)),
                f"{name}_policy_forced_cover_shares": float(covers.sum()),
                f"{name}_policy_forced_cover_notional": float(covers.sum()) * observation.price,
                f"{name}_actual_gross_notional_before": float(np.abs(self.before_positions[mask]).sum()) * observation.price,
                f"{name}_actual_net_notional_before": float(self.before_positions[mask].sum()) * observation.price,
                f"{name}_wealth_before": float((population.cash[mask] + self.before_positions[mask] * observation.price).sum()),
            })
        return submitted


def _inventory_excess_fields(positions, limits, strategies, price, stage):
    fields = {}
    for code, name in enumerate(STRATEGY_NAMES):
        mask = strategies == code
        selected, selected_limits = positions[mask], limits[mask]
        fields.update({f"{name}_{stage}_position_gross_notional": float(np.abs(selected).sum()) * price,
                       f"{name}_{stage}_position_net_notional": float(selected.sum()) * price,
                       f"{name}_{stage}_target_limit_notional": float(selected_limits.sum()) * price})
        for side, excess in (("long", np.maximum(selected - selected_limits, 0)),
                             ("short", np.maximum(-selected_limits - selected, 0))):
            prefix = f"{name}_{stage}_{side}_excess_"
            fields.update({prefix + "count": int(np.count_nonzero(excess > 1e-9)),
                           prefix + "shares": float(excess.sum()),
                           prefix + "notional": float(excess.sum()) * price})
    return fields


def _budget_inverse_fields(after_direction, after_budget, actual_orders,
                           value_signal, strategies, price, stage, margin_orders=None):
    fields = {}
    margin_orders = np.zeros_like(actual_orders) if margin_orders is None else margin_orders
    for name, sign, signal in (("positive_signal_sell", -1, value_signal > 0),
                               ("negative_signal_buy", 1, value_signal < 0)):
        mask = (strategies == VALUE_STRATEGY) & signal
        direction = np.maximum(sign * after_direction[mask], 0)
        budget = np.maximum(sign * after_budget[mask], 0)
        total = np.maximum(sign * actual_orders[mask], 0)
        margin = np.minimum(total, np.maximum(sign * margin_orders[mask], 0))
        ratio = np.divide(np.maximum(budget - direction, 0), budget,
                          out=np.zeros_like(budget), where=budget > 0)
        attributed_budget = (total - margin) * ratio
        other = total - margin - attributed_budget
        for component, quantities in (("total", total), ("budget", attributed_budget),
                                      ("margin", margin), ("other", other)):
            prefix = f"value_{stage}_{name}_{component}_"
            fields.update({prefix + "count": int(np.count_nonzero(quantities)),
                           prefix + "shares": float(quantities.sum()),
                           prefix + "notional": float(quantities.sum()) * price})
    return fields


class _ValueBudgetPolicy(_PriceDiscoveryPolicy):
    def __init__(self, policy):
        policy.instrument_orders = True
        super().__init__(policy, probes_enabled=False)
        self.excess_streaks = {}
        self.excess_agent_totals = {}

    def observe_inventory(self, row, positions, limits, price, stage):
        row.update(_inventory_excess_fields(positions, limits, self.strategies, price, stage))
        value = self.strategies == VALUE_STRATEGY
        indices = np.flatnonzero(value)
        for side, excess in (("long", np.maximum(positions[value] - limits[value], 0)),
                             ("short", np.maximum(-limits[value] - positions[value], 0))):
            key = f"value_{stage}_{side}"
            streak = self.excess_streaks.setdefault(key, np.zeros(len(excess), dtype=int))
            totals = self.excess_agent_totals.setdefault(key, np.zeros(len(excess)))
            if self.day <= self.policy.config.burn_in_days:
                streak[:] = 0
            else:
                streak[:] = np.where(excess > 1e-9, streak + 1, 0)
                totals += excess * price
            row[f"{key}_longest_current_excess_run"] = int(streak.max()) if len(streak) else 0
            row[f"{key}_longest_current_excess_agent"] = int(indices[np.argmax(streak)]) if len(streak) else -1
            row[f"{key}_maximum_individual_excess_notional"] = float(excess.max()) * price if len(excess) else 0.0
            row[f"{key}_maximum_individual_excess_agent"] = int(indices[np.argmax(excess)]) if len(excess) else -1
            for quantile in (.5, .95, .99):
                row[f"{key}_individual_excess_notional_q{quantile}"] = float(np.quantile(excess, quantile)) * price if len(excess) else 0.0

    def act(self, observation, population, rng):
        submitted = super().act(observation, population, rng)
        snapshot = self.order_snapshot = self.policy.order_diagnostics
        if snapshot is None or not np.array_equal(snapshot.after_risk, submitted):
            raise RuntimeError("policy order snapshots do not match submitted orders")
        if any(getattr(snapshot, field) is None for field in ("after_budget", "budget_adjustment", "budget_limit", "maximum_rebalance")):
            raise RuntimeError("budget diagnostic snapshots are incomplete")
        if not np.array_equal(snapshot.budget_adjustment, snapshot.after_budget - snapshot.after_constraint):
            raise RuntimeError("budget adjustment does not match order snapshots")
        value = self.strategies == VALUE_STRATEGY
        expected = snapshot.after_constraint.copy()
        if self.policy.config.value_inventory_control == "budget_priority":
            lower = np.maximum(-snapshot.budget_limit - self.before_positions, -snapshot.maximum_rebalance)
            upper = np.minimum(snapshot.budget_limit - self.before_positions, snapshot.maximum_rebalance)
            reachable = lower <= upper
            projected = -np.sign(self.before_positions) * snapshot.maximum_rebalance
            projected[reachable] = np.clip(expected[reachable], lower[reachable], upper[reachable])
            expected[value] = projected[value]
        if not np.array_equal(expected, snapshot.after_budget):
            raise RuntimeError("budget orders do not match the registered projection")
        for stage, orders in (("before_constraint", snapshot.before_constraint),
                              ("after_direction", snapshot.after_constraint),
                              ("after_budget", snapshot.after_budget), ("after_risk", snapshot.after_risk)):
            for code, name in enumerate(STRATEGY_NAMES):
                selected = orders[self.strategies == code]
                self.row.update({f"{name}_{stage}_{side}_count": int(np.count_nonzero(flags))
                                 for side, flags in (("buy", selected > 0), ("sell", selected < 0), ("zero", selected == 0))})
            for name, fields in _strategy_orders(orders, self.strategies, observation.price).items():
                self.row.update({f"{name}_{stage}_{key}": number for key, number in fields.items()})
            reference = orders if stage in ("before_constraint", "after_direction") else snapshot.after_budget
            direction = orders if stage == "before_constraint" else snapshot.after_constraint
            margin = snapshot.forced_cover_orders if stage == "after_risk" else None
            self.row.update(_budget_inverse_fields(direction, reference, orders, snapshot.value_signal,
                                                  self.strategies, observation.price, stage, margin))
        for code, name in enumerate(STRATEGY_NAMES):
            mask = self.strategies == code
            covers = snapshot.forced_cover_orders[mask]
            adjustment = snapshot.budget_adjustment[mask]
            self.row.update({
                f"{name}_budget_adjustment_count": int(np.count_nonzero(adjustment)),
                f"{name}_budget_adjustment_absolute_shares": float(np.abs(adjustment).sum()),
                f"{name}_budget_adjustment_absolute_notional": float(np.abs(adjustment).sum()) * observation.price,
                f"{name}_budget_adjustment_net_shares": float(adjustment.sum()),
                f"{name}_budget_adjustment_net_notional": float(adjustment.sum()) * observation.price,
                f"{name}_policy_forced_cover_count": int(np.count_nonzero(covers)),
                f"{name}_policy_forced_cover_shares": float(covers.sum()),
                f"{name}_policy_forced_cover_notional": float(covers.sum()) * observation.price,
                f"{name}_actual_gross_notional_before": float(np.abs(self.before_positions[mask]).sum()) * observation.price,
                f"{name}_actual_net_notional_before": float(self.before_positions[mask].sum()) * observation.price,
                f"{name}_wealth_before": float((population.cash[mask] + self.before_positions[mask] * observation.price).sum()),
            })
        self.observe_inventory(self.row, self.before_positions, snapshot.budget_limit, observation.price, "opening")
        self.observe_inventory(self.row, self.before_positions + snapshot.after_budget, snapshot.budget_limit, observation.price, "after_budget")
        return submitted


class _ObservedSettlement:
    def __init__(self, settlement):
        self.settlement = settlement

    def settle(self, **kwargs):
        self.result = self.settlement.settle(**kwargs)
        return self.result


def _run_price_discovery(scenario: str, config: Stage1Config, output: Path,
                         *, value_direction: bool = False, value_budget: bool = False) -> dict[str, object]:
    if config.learning_enabled:
        raise ValueError("price-discovery replay requires fixed strategy membership")
    harness = MarketHarness(config)
    observer = (_ValueBudgetPolicy(harness.policy) if value_budget else
                _ValueDirectionPolicy(harness.policy) if value_direction else _PriceDiscoveryPolicy(harness.policy))
    settlement = _ObservedSettlement(harness.settlement)
    harness.policy, harness.settlement = observer, settlement
    daily_path = Path(output) / "daily" / f"{scenario}_{config.seed}.csv"
    maximum_position_error = maximum_price_error = 0.0
    with daily_path.open("x", encoding="utf-8", newline="") as handle:
        writer = None
        for index in range(config.trading_days):
            audit = harness.step(index)
            filled = settlement.result.executed_orders
            changes = harness.population.positions - observer.before_positions
            position_error = float(np.max(np.abs(changes - filled)))
            if not np.allclose(changes, filled, rtol=0, atol=1e-9):
                raise RuntimeError(f"filled orders do not match position changes on day {audit.day}")
            raw = audit.permanent_impact + audit.transient_impact_change
            adjustment = audit.demand_log_return - raw
            total = float(np.log(audit.mid_price_after / audit.price_before))
            price_error = total - (audit.public_news_impact + raw + adjustment)
            if abs(price_error) > 1e-12:
                raise RuntimeError(f"price decomposition failed on day {audit.day}")
            row = {"day": audit.day, "evaluation_day": audit.day > config.burn_in_days,
                   "fundamental_value": audit.fundamental_value,
                   "price_before": audit.price_before, "price_after": audit.mid_price_after,
                   "execution_price": audit.execution_price,
                   "signed_log_price_gap": float(np.log(audit.mid_price_after / audit.fundamental_value)),
                   **observer.row,
                   "depth_shares": audit.depth,
                   "depth_notional": audit.depth * audit.price_before,
                   "participation_pressure": observer.row["submitted_gross_shares"] / audit.depth,
                   "capped_participation": min(observer.row["submitted_gross_shares"] / audit.depth, 3.0),
                   "amplified_participation": min(observer.row["submitted_gross_shares"] / audit.depth, 3.0) ** config.participation_pressure_exponent,
                   "public_news_impact": audit.public_news_impact,
                   "permanent_impact": audit.permanent_impact,
                   "transient_impact": audit.transient_impact,
                   "transient_impact_change": audit.transient_impact_change,
                   "price_cap_adjustment": adjustment,
                   "raw_demand_log_return": raw,
                   "demand_log_return": audit.demand_log_return, "total_log_return": total,
                   "price_decomposition_error": price_error, "price_cap_hit": audit.price_cap_hit,
                   "maximum_position_change_error": position_error,
                   "cash_error": audit.cash_error, "share_error": audit.share_error}
            filled_by_strategy = _strategy_orders(filled, observer.strategies, audit.execution_price)
            for code, name in enumerate(STRATEGY_NAMES):
                mask = observer.strategies == code
                row.update({f"{name}_filled_{key}": value for key, value in filled_by_strategy[name].items()})
                row[f"{name}_actual_position_change_shares"] = float(changes[mask].sum())
                row[f"{name}_submitted_minus_filled_shares"] = float((observer.submitted - filled)[mask].sum())
                row[f"{name}_submitted_minus_filled_notional_at_open"] = float((observer.submitted - filled)[mask].sum()) * audit.price_before
                row[f"{name}_submitted_minus_filled_absolute_shares"] = float(np.abs(observer.submitted - filled)[mask].sum())
            if value_direction or value_budget:
                snapshot = observer.order_snapshot
                additional_covers = np.maximum(filled - np.maximum(snapshot.after_risk, 0), 0)
                filled_policy_covers = np.minimum(snapshot.forced_cover_orders, np.maximum(filled, 0))
                if value_direction:
                    row.update(_value_direction_fields(filled, snapshot.value_signal, observer.strategies,
                                                       audit.execution_price, "filled", filled_policy_covers + additional_covers))
                if value_budget:
                    row.update(_budget_inverse_fields(snapshot.after_constraint, snapshot.after_budget, filled,
                        snapshot.value_signal, observer.strategies, audit.execution_price, "filled",
                        filled_policy_covers + additional_covers))
                    observer.observe_inventory(row, harness.population.positions, snapshot.budget_limit,
                                               audit.price_before, "after_fill_decision_limit")
                    closing_limits = np.minimum(config.target_position_fraction * harness.population.reference_wealth / audit.mid_price_after, config.position_cap)
                    observer.observe_inventory(row, harness.population.positions, closing_limits,
                                               audit.mid_price_after, "closing")
                for code, name in enumerate(STRATEGY_NAMES):
                    mask = observer.strategies == code
                    covers = additional_covers[mask]
                    positions = harness.population.positions[mask]
                    row.update({
                        f"{name}_settlement_additional_cover_count": int(np.count_nonzero(covers)),
                        f"{name}_settlement_additional_cover_shares": float(covers.sum()),
                        f"{name}_settlement_additional_cover_notional": float(covers.sum()) * audit.execution_price,
                        f"{name}_filled_policy_forced_cover_shares": float(filled_policy_covers[mask].sum()),
                        f"{name}_wealth_after": float((harness.population.cash[mask] + positions * audit.mid_price_after).sum()),
                        f"{name}_actual_gross_notional_after": float(np.abs(positions).sum()) * audit.mid_price_after,
                        f"{name}_actual_net_notional_after": float(positions.sum()) * audit.mid_price_after,
                        f"{name}_absolute_target_gap_notional_after_fill": float(np.abs(harness.population.desired_positions[mask] - positions).sum()) * audit.mid_price_after,
                    })
                    if value_budget:
                        row.update({f"{name}_filled_{side}_count": int(np.count_nonzero(flags))
                                    for side, flags in (("buy", filled[mask] > 0), ("sell", filled[mask] < 0), ("zero", filled[mask] == 0))})
                        row[f"{name}_filled_policy_forced_cover_count"] = int(np.count_nonzero(filled_policy_covers[mask]))
                        row[f"{name}_filled_policy_forced_cover_notional"] = float(filled_policy_covers[mask].sum()) * audit.execution_price
                agent_wealth = sum(row[f"{name}_wealth_after"] for name in STRATEGY_NAMES)
                maker_wealth = audit.market_maker_cash + audit.market_maker_inventory * audit.mid_price_after
                expected_wealth = audit.total_cash + audit.total_shares * audit.mid_price_after
                row.update(total_agent_wealth=agent_wealth, market_maker_wealth=maker_wealth,
                           total_cash=audit.total_cash, total_shares=audit.total_shares,
                           wealth_accounting_error=agent_wealth + maker_wealth - expected_wealth)
                if not np.isclose(agent_wealth + maker_wealth, expected_wealth, rtol=1e-12, atol=1e-7):
                    raise RuntimeError(f"wealth accounting failed on day {audit.day}")
            if not all(np.isfinite(value) for value in row.values()):
                raise RuntimeError(f"nonfinite price-discovery diagnostic on day {audit.day}")
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_price_error = max(maximum_price_error, abs(price_error))
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    result = harness.run()
    metrics = result_metrics(result, config.burn_in_days)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("nonfinite replay metrics")
    report = {"scenario": scenario, "seed": config.seed, "metrics": metrics,
            "fingerprint": result.fingerprint(), "daily_file": str(daily_path.relative_to(output)),
            "probes": observer.probes, "maximum_position_change_error": maximum_position_error,
            "maximum_price_decomposition_error": maximum_price_error}
    if value_budget:
        report["individual_inventory_excess"] = {
            f"{name}_mean_daily_excess_notional_q{quantile}": float(np.quantile(values / (config.trading_days - config.burn_in_days), quantile)) if len(values) else 0.0
            for name, values in observer.excess_agent_totals.items() for quantile in (.5, .95, .99, 1.0)
        }
    return report


def run_price_discovery(config: Stage1Config, protocol: MechanismProtocol, output: Path,
                        *, source: Path, workers: int = 4) -> dict[str, object]:
    if protocol.stage != "development" or config.learning_enabled:
        raise ValueError("price discovery requires a development protocol without learning")
    if workers < 1 or not protocol.seeds or len(set(protocol.seeds)) != len(protocol.seeds):
        raise ValueError("workers must be positive and seeds nonempty and unique")
    if config.burn_in_days < protocol.minimum_burn_in_days or config.trading_days - config.burn_in_days < protocol.minimum_evaluation_days:
        raise ValueError("config does not meet the protocol evaluation window")
    source, output = Path(source), Path(output)
    prior = json.loads((source / "diagnostics" / "manifest.json").read_text(encoding="utf-8"))
    development = json.loads((source / "development" / "mechanism_report.json").read_text(encoding="utf-8"))
    expected_protocol = json.loads(json.dumps(asdict(protocol)))
    for manifest in (prior, development):
        if Stage1Config.from_dict(manifest["config"]).to_dict() != config.to_dict() or manifest["protocol"] != expected_protocol:
            raise ValueError("source config, protocol, seeds or evaluation window mismatch")
    scenarios = build_scenarios(config)
    names = ("full", "fixed_liquidity", "no_agent_information")
    tasks = [(name, replace(scenarios[name], seed=seed), output) for name in names for seed in protocol.seeds]
    output.mkdir(parents=True, exist_ok=False)
    (output / "daily").mkdir()
    checkpoint = {"status": "RUNNING", "expected_runs": len(tasks), "completed_runs": [],
                  "unfinished_runs": [{"scenario": name, "seed": item.seed} for name, item, _ in tasks],
                  "reconstruction_complete": False, "errors": []}
    _write_json(output / "checkpoint.json", checkpoint)
    _write_json(output / "manifest.json", {
        "config": config.to_dict(), "protocol": asdict(protocol), "source": str(source.resolve()),
        "scenario_configs": {name: scenarios[name].to_dict() for name in names},
        "code_revision": resolve_code_revision(), "decision": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "stage1_complete": False, "may_enter_stage2": False,
        "probe_days": [1, 1001, 1501, 2001, 2501, 3001, 3500],
        "probe_interpretation": "Local fixed-state interventions; not full-path or additive causal contributions.",
        "submitted_notional_valuation": "opening_price", "filled_notional_valuation": "execution_price",
        "diagnostic_share_tolerance": 1e-9,
    })
    results = {}
    key = ("reconstruction", None)
    futures = {}
    try:
        reconstruction = reconstruct_price_gaps(source, output)
        checkpoint["reconstruction_complete"] = True
        _write_json(output / "checkpoint.json", checkpoint)
        with (source / "development" / "per_seed_metrics.csv").open(encoding="utf-8", newline="") as handle:
            development_rows = list(csv.DictReader(handle))
        expected = {}
        for name, item, _ in tasks:
            if name in ("full", "fixed_liquidity"):
                source_name = "candidate" if name == "full" else "candidate_fixed_liquidity"
                value = json.loads((source / "diagnostics" / f"{source_name}_{item.seed}.json").read_text(encoding="utf-8"))
                if value["scenario"] != source_name or value["seed"] != item.seed:
                    raise ValueError("source replay identity mismatch")
                expected[(name, item.seed)] = value
            else:
                matching = [row for row in development_rows if row["scenario"] == name and int(row["seed"]) == item.seed]
                if len(matching) != 1:
                    raise ValueError(f"missing or duplicate source metrics: {name}/{item.seed}")
                expected[(name, item.seed)] = {"metrics": {metric: float(value) for metric, value in matching[0].items() if metric not in ("scenario", "seed")}}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_price_discovery, *task): (task[0], task[1].seed) for task in tasks}
            try:
                for future in as_completed(futures):
                    key = futures[future]
                    result = future.result()
                    reference = expected[key]
                    if result["metrics"] != reference["metrics"]:
                        raise RuntimeError(f"replay metrics differ from source: {key}")
                    if "fingerprint" in reference and result["fingerprint"] != reference["fingerprint"]:
                        raise RuntimeError(f"replay fingerprint differs from source: {key}")
                    if "daily_file" in reference:
                        with (source / "diagnostics" / reference["daily_file"]).open(encoding="utf-8", newline="") as handle:
                            old_daily = list(csv.DictReader(handle))
                        with (output / result["daily_file"]).open(encoding="utf-8", newline="") as handle:
                            new_daily = list(csv.DictReader(handle))
                        if len(old_daily) != len(new_daily) or any(
                            old_value != new_row[name]
                            for old_row, new_row in zip(old_daily, new_daily)
                            for name, old_value in old_row.items()
                        ):
                            raise RuntimeError(f"replay daily diagnostics differ from source: {key}")
                    result["source_verification"] = {"all_metrics_exact": True,
                        "fingerprint_exact": True if "fingerprint" in reference else None,
                        "original_daily_columns_exact": True if "daily_file" in reference else None}
                    _write_json(output / f"{key[0]}_{key[1]}.json", result)
                    results[key] = result
                    checkpoint["completed_runs"] = [{"scenario": name, "seed": seed} for name, seed in sorted(results)]
                    checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
                    _write_json(output / "checkpoint.json", checkpoint)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        report = {"decision": "DEVELOPMENT_DIAGNOSTICS_ONLY", "stage1_complete": False,
                  "may_enter_stage2": False, "reconstruction": reconstruction,
                  "runs": [results[(name, item.seed)] for name, item, _ in tasks]}
        _write_json(output / "diagnostics.json", report)
        checkpoint["status"] = "COMPLETE"
        _write_json(output / "checkpoint.json", checkpoint)
        return report
    except Exception as error:
        checkpoint["status"] = "ENGINEERING_ERROR"
        checkpoint["errors"] = [{"scenario": key[0], "seed": key[1], "type": type(error).__name__, "message": str(error)}]
        checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
        _write_json(output / "checkpoint.json", checkpoint)
        for future in futures:
            future.cancel()
        raise


def _run_value_direction(scenario: str, config: Stage1Config, output: Path,
                         *, value_budget: bool = False) -> dict[str, object]:
    result = _run_price_discovery(scenario, config, output,
                                  value_direction=not value_budget, value_budget=value_budget)
    with (Path(output) / result["daily_file"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    evaluation = rows[config.burn_in_days:]
    exposure = {}
    for name in STRATEGY_NAMES:
        for metric, field in (
            ("mean_gross_notional", "actual_gross_notional_before"),
            ("mean_net_notional", "actual_net_notional_before"),
            ("mean_absolute_target_gap_notional", "absolute_target_gap_notional_after_fill"),
            ("mean_wealth", "wealth_after"),
        ):
            exposure[f"{name}_{metric}"] = float(np.mean([float(row[f"{name}_{field}"]) for row in evaluation]))
        exposure[f"{name}_final_wealth"] = float(rows[-1][f"{name}_wealth_after"])
        for stage in ("submitted", "filled"):
            exposure[f"{name}_gross_{stage}_notional"] = sum(
                float(row[f"{name}_{stage}_buy_notional"]) + float(row[f"{name}_{stage}_sell_notional"])
                for row in evaluation
            )
    direction_fields = [name for name in rows[0] if any(
        part in name for part in ("_signal_voluntary_", "_signal_zero_order_", "_policy_forced_cover_", "_settlement_additional_cover_")
    )]
    if value_budget:
        stages = ("before_constraint", "after_direction", "after_budget", "after_risk", "filled")
        direction_fields = [name for name in rows[0] if any(
            part in name for part in ("_positive_signal_sell_", "_negative_signal_buy_", "_budget_adjustment_",
                                      "_policy_forced_cover_", "_settlement_additional_cover_")
        ) or any(name == f"{strategy}_{stage}_{side}_{unit}"
                 for strategy in STRATEGY_NAMES for stage in stages
                 for side in ("buy", "sell", "net", "zero") for unit in ("count", "shares", "notional"))]
        inventory = dict(result.pop("individual_inventory_excess"))
        for strategy in STRATEGY_NAMES:
            wealth = np.array([float(row[f"{strategy}_wealth_after"]) for row in evaluation])
            starting_wealth = float(evaluation[0][f"{strategy}_wealth_before"])
            peaks = np.maximum.accumulate(np.concatenate(([starting_wealth], wealth)))[1:]
            drawdowns = np.divide(peaks - wealth, peaks, out=np.zeros_like(wealth), where=peaks > 0)
            exposure[f"{strategy}_maximum_wealth_drawdown"] = float(drawdowns.max())
            exposure[f"{strategy}_maximum_wealth_drawdown_day"] = int(evaluation[int(np.argmax(drawdowns))]["day"])
            for stage in ("opening", "after_budget", "after_fill_decision_limit", "closing"):
                denominator = np.array([float(row[f"{strategy}_{stage}_target_limit_notional"]) for row in evaluation])
                for side in ("long", "short"):
                    prefix = f"{strategy}_{stage}_{side}"
                    for unit in ("count", "shares", "notional"):
                        numbers = np.array([float(row[f"{prefix}_excess_{unit}"]) for row in evaluation])
                        inventory[f"{prefix}_mean_excess_{unit}"] = float(numbers.mean())
                        inventory[f"{prefix}_maximum_excess_{unit}"] = float(numbers.max())
                        inventory[f"{prefix}_maximum_excess_{unit}_day"] = int(evaluation[int(np.argmax(numbers))]["day"])
                        for quantile in (.5, .95, .99):
                            inventory[f"{prefix}_excess_{unit}_q{quantile}"] = float(np.quantile(numbers, quantile))
                    amounts = np.array([float(row[f"{prefix}_excess_notional"]) for row in evaluation])
                    valid = denominator > 0
                    ratios = amounts[valid] / denominator[valid]
                    inventory[f"{prefix}_ratio_valid_days"] = int(valid.sum())
                    inventory[f"{prefix}_mean_excess_ratio"] = float(ratios.mean()) if len(ratios) else None
                    inventory[f"{prefix}_maximum_excess_ratio"] = float(ratios.max()) if len(ratios) else None
                    inventory[f"{prefix}_days_with_excess"] = sum(int(float(row[f"{prefix}_excess_count"]) > 0) for row in evaluation)
                    if strategy == "value":
                        for field, output_name in (("longest_current_excess_run", "longest_individual_excess_run"),
                                                   ("maximum_individual_excess_notional", "maximum_individual_excess_notional")):
                            numbers = np.array([float(row[f"{prefix}_{field}"]) for row in evaluation])
                            index = int(np.argmax(numbers))
                            agent_field = "longest_current_excess_agent" if field == "longest_current_excess_run" else "maximum_individual_excess_agent"
                            inventory[f"{prefix}_{output_name}"] = float(numbers[index])
                            inventory[f"{prefix}_{output_name}_day"] = int(evaluation[index]["day"])
                            inventory[f"{prefix}_{output_name}_agent"] = int(evaluation[index][f"{prefix}_{agent_field}"])
        result["inventory_excess"] = inventory
    result["direction_counts"] = {
        name: sum(int(row[name]) for row in evaluation)
        for name in direction_fields if name.endswith("_count")
    }
    result["direction_flows"] = {
        name: sum(float(row[name]) for row in evaluation)
        for name in direction_fields if not name.endswith("_count")
    }
    result["exposure"] = exposure
    result["maximum_wealth_accounting_error"] = max(abs(float(row["wealth_accounting_error"])) for row in rows)
    reconstructed = reconstruct_price_gap(rows, config)
    if not np.isclose(reconstructed["summary"]["mean_absolute_log_price_gap"], result["metrics"]["mean_absolute_log_price_gap"], rtol=0, atol=1e-11):
        raise RuntimeError("value-direction price-gap reconstruction differs from simulation metrics")
    destination = Path(output) / "reconstruction"
    destination.mkdir(exist_ok=True)
    path = destination / f"{scenario}_{config.seed}.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reconstructed["daily"][0]))
        writer.writeheader()
        writer.writerows(reconstructed["daily"])
    result["gap_reconstruction"] = reconstructed["summary"]
    result["gap_reconstruction"]["final_signed_log_price_gap"] = reconstructed["daily"][-1]["signed_log_price_gap"]
    result["reconstruction_daily_file"] = str(path.relative_to(output))
    return result


def _run_value_budget(scenario: str, config: Stage1Config, output: Path) -> dict[str, object]:
    return _run_value_direction(scenario, config, output, value_budget=True)


def run_value_direction(config: Stage1Config, protocol: MechanismProtocol, output: Path,
                        *, source: Path, workers: int = 4) -> dict[str, object]:
    if protocol.stage != "development" or config.learning_enabled:
        raise ValueError("value direction requires a development protocol without learning")
    if config.value_order_constraint != "valuation_direction":
        raise ValueError("candidate requires the valuation_direction constraint")
    if workers < 1 or not protocol.seeds or len(set(protocol.seeds)) != len(protocol.seeds):
        raise ValueError("workers must be positive and seeds nonempty and unique")
    if config.burn_in_days < protocol.minimum_burn_in_days or config.trading_days - config.burn_in_days < protocol.minimum_evaluation_days:
        raise ValueError("config does not meet the protocol evaluation window")
    source, output = Path(source), Path(output)
    prior = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    baseline = replace(config, value_order_constraint="none")
    if Stage1Config.from_dict(prior["config"]).to_dict() != baseline.to_dict():
        raise ValueError("source baseline config or evaluation window mismatch")
    expected_protocol = json.loads(json.dumps(asdict(protocol)))
    if {key: value for key, value in prior["protocol"].items() if key not in ("version", "purpose")} != {
        key: value for key, value in expected_protocol.items() if key not in ("version", "purpose")
    }:
        raise ValueError("source protocol criteria, seeds or evaluation window mismatch")
    if "scenario_configs" in prior and Stage1Config.from_dict(prior["scenario_configs"]["full"]).to_dict() != baseline.to_dict():
        raise ValueError("source full scenario config mismatch")
    scenarios = {"baseline": baseline, "candidate": config}
    tasks = [(name, replace(item, seed=seed), output) for name, item in scenarios.items() for seed in protocol.seeds]
    output.mkdir(parents=True, exist_ok=False)
    (output / "daily").mkdir()
    (output / "reconstruction").mkdir()
    checkpoint = {"status": "RUNNING", "expected_runs": len(tasks), "completed_runs": [],
                  "unfinished_runs": [{"scenario": name, "seed": item.seed} for name, item, _ in tasks],
                  "errors": []}
    _write_json(output / "checkpoint.json", checkpoint)
    _write_json(output / "manifest.json", {
        "config": config.to_dict(), "protocol": asdict(protocol), "source": str(source.resolve()),
        "scenario_configs": {name: item.to_dict() for name, item in scenarios.items()},
        "code_revision": resolve_code_revision(), "decision": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "stage1_complete": False, "may_enter_stage2": False,
        "direction_classification": "Strict sign of policy value_signal and voluntary orders for value traders; zero voluntary orders are counted separately.",
        "policy_forced_cover": "Original policy sell-or-zero branch cover; voluntary positive buys are not reclassified.",
        "settlement_additional_cover": "Positive executed increment above max(policy after_risk order, 0); recorded separately from policy covers.",
        "submitted_notional_valuation": "opening_price", "filled_notional_valuation": "execution_price",
        "exposure_valuation": "Mean actual holdings at open; target deviation and wealth after settlement at close; gross flows use their stage prices.",
        "interpretation": "Paired full-path observations; cancelled orders are not additive price contributions or formal acceptance evidence.",
    })
    results = {}
    futures = {}
    key = ("source", None)
    try:
        references = {}
        for seed in protocol.seeds:
            key = ("baseline", seed)
            reference = json.loads((source / f"full_{seed}.json").read_text(encoding="utf-8"))
            if reference["scenario"] != "full" or reference["seed"] != seed:
                raise ValueError("source baseline result identity mismatch")
            references[seed] = reference
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_value_direction, *task): (task[0], task[1].seed) for task in tasks}
            try:
                for future in as_completed(futures):
                    key = futures[future]
                    result = future.result()
                    if key[0] == "baseline":
                        reference = references[key[1]]
                        if result["metrics"] != reference["metrics"]:
                            raise RuntimeError(f"baseline metrics differ from source: {key}")
                        if result["fingerprint"] != reference["fingerprint"]:
                            raise RuntimeError(f"baseline fingerprint differs from source: {key}")
                        result["source_verification"] = {"all_metrics_exact": True, "fingerprint_exact": True}
                    _write_json(output / f"{key[0]}_{key[1]}.json", result)
                    results[key] = result
                    checkpoint["completed_runs"] = [{"scenario": name, "seed": seed} for name, seed in sorted(results)]
                    checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
                    _write_json(output / "checkpoint.json", checkpoint)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        comparisons = []
        for seed in protocol.seeds:
            left, right = results[("baseline", seed)], results[("candidate", seed)]
            comparison = {"seed": seed}
            for category in ("metrics", "exposure", "direction_counts", "direction_flows"):
                if set(left[category]) != set(right[category]):
                    raise RuntimeError(f"paired diagnostic fields differ: {seed}/{category}")
                comparison[category] = {name: {"baseline": left[category][name], "candidate": right[category][name],
                                              "delta": right[category][name] - left[category][name]}
                                        for name in left[category]}
            gap_values = []
            for run in (left, right):
                summary = run["gap_reconstruction"]
                values = {name: value for name, value in summary.items() if isinstance(value, (int, float))}
                values.update({f"signed_gap_quantile_{quantile}": value for quantile, value in summary["signed_gap_quantiles"].items()})
                values.update({f"{component}_{statistic}": value
                               for component, statistics in summary["components"].items()
                               for statistic, value in statistics.items()})
                gap_values.append(values)
            comparison["gap_reconstruction"] = {
                name: {"baseline": gap_values[0][name], "candidate": gap_values[1][name],
                       "delta": gap_values[1][name] - gap_values[0][name]}
                for name in gap_values[0]
            }
            comparisons.append(comparison)
        report = {"decision": "DEVELOPMENT_DIAGNOSTICS_ONLY", "stage1_complete": False,
                  "may_enter_stage2": False, "paired_seed_count": len(comparisons),
                  "runs": [results[(name, item.seed)] for name, item, _ in tasks],
                  "paired_comparisons": comparisons}
        _write_json(output / "diagnostics.json", report)
        with (output / "per_seed_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            rows = [{"scenario": run["scenario"], "seed": run["seed"], **run["metrics"], **run["exposure"]} for run in report["runs"]]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint["status"] = "COMPLETE"
        _write_json(output / "checkpoint.json", checkpoint)
        return report
    except Exception as error:
        checkpoint["status"] = "ENGINEERING_ERROR"
        checkpoint["errors"] = [{"scenario": key[0], "seed": key[1], "type": type(error).__name__, "message": str(error)}]
        checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
        _write_json(output / "checkpoint.json", checkpoint)
        raise


def run_value_budget(config: Stage1Config, protocol: MechanismProtocol, output: Path,
                     *, source: Path, workers: int = 4) -> dict[str, object]:
    if protocol.stage != "development" or config.learning_enabled:
        raise ValueError("value budget requires a development protocol without learning")
    if config.value_order_constraint != "valuation_direction" or config.value_inventory_control != "budget_priority":
        raise ValueError("candidate requires valuation_direction and budget_priority")
    if workers < 1 or not protocol.seeds or len(set(protocol.seeds)) != len(protocol.seeds):
        raise ValueError("workers must be positive and seeds nonempty and unique")
    if config.burn_in_days < protocol.minimum_burn_in_days or config.trading_days - config.burn_in_days < protocol.minimum_evaluation_days:
        raise ValueError("config does not meet the protocol evaluation window")
    source, output = Path(source), Path(output)
    scenarios = {
        "baseline": replace(config, value_order_constraint="none", value_inventory_control="none"),
        "direction_only": replace(config, value_inventory_control="none"),
        "budget_only": replace(config, value_order_constraint="none"),
        "candidate": config,
    }
    prior = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if Stage1Config.from_dict(prior["config"]).to_dict() != scenarios["direction_only"].to_dict():
        raise ValueError("source direction-only config or evaluation window mismatch")
    source_names = {"baseline": "baseline", "direction_only": "candidate"}
    if {name: Stage1Config.from_dict(item).to_dict() for name, item in prior["scenario_configs"].items()} != {
        old_name: scenarios[name].to_dict() for name, old_name in source_names.items()
    }:
        raise ValueError("source baseline or direction-only scenario config mismatch")
    expected_protocol = json.loads(json.dumps(asdict(protocol)))
    if {key: value for key, value in prior["protocol"].items() if key not in ("version", "purpose")} != {
        key: value for key, value in expected_protocol.items() if key not in ("version", "purpose")
    }:
        raise ValueError("source protocol criteria, seeds or evaluation window mismatch")
    tasks = [(name, replace(item, seed=seed), output) for name, item in scenarios.items() for seed in protocol.seeds]
    output.mkdir(parents=True, exist_ok=False)
    (output / "daily").mkdir()
    (output / "reconstruction").mkdir()
    checkpoint = {"status": "RUNNING", "expected_runs": len(tasks), "completed_runs": [],
                  "unfinished_runs": [{"scenario": name, "seed": item.seed} for name, item, _ in tasks],
                  "errors": []}
    _write_json(output / "checkpoint.json", checkpoint)
    _write_json(output / "manifest.json", {
        "config": config.to_dict(), "protocol": asdict(protocol), "source": str(source.resolve()),
        "scenario_configs": {name: item.to_dict() for name, item in scenarios.items()},
        "code_revision": resolve_code_revision(), "decision": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "stage1_complete": False, "may_enter_stage2": False,
        "order_stages": ["before_constraint", "after_direction", "after_budget", "after_risk", "filled"],
        "after_direction_snapshot": "PolicyOrderDiagnostics.after_constraint",
        "inverse_order_accounting": "Strict nonzero value-signal sign. Budget-added inverse shares are max(inverse(after_budget)-inverse(after_direction),0). After removing explicit margin shares, actual inverse shares are allocated in that budget-added fraction; the remainder is other. Component counts can overlap; shares and notionals add to total. Quantity attribution is not an independent causal effect or physically split order.",
        "policy_forced_cover": "Original policy sell-or-zero after_budget branch cover; existing positive orders are not reclassified as margin.",
        "settlement_additional_cover": "Positive executed increment above max(policy after_risk order,0), separate from filled policy covers.",
        "inventory_excess": "Per-agent max(h-L,0) and max(-L-h,0), aggregated without offsetting long and short. Opening, hypothetical after_budget, and actual after_fill_decision_limit use opening L and price; closing uses actual holdings and L recomputed at closing price. L is a soft reference-wealth target, not a hard margin limit. Budget controls value agents only.",
        "inventory_statistics": "Evaluation-window daily aggregate quantiles and each value agent's mean daily excess-notional quantiles. Individual consecutive runs reset at evaluation start; daily agent ids are population indices. Raw excess amounts retain floating-point residuals; positive counts and consecutive runs use a 1e-9 share tolerance. Ratios use positive aggregate target limits only; no valid denominator gives null.",
        "diagnostic_share_tolerance": 1e-9,
        "submitted_notional_valuation": "opening_price", "filled_notional_valuation": "execution_price",
        "exposure_valuation": "Mean actual holdings at open; target deviation and wealth after settlement at close; gross flows use stage prices. Wealth drawdown starts at evaluation-window opening wealth.",
        "interpretation": "All-seed full-path comparisons of the preregistered candidate against each other branch. Order quantities are not additive price contributions or formal acceptance evidence. No local probes.",
    })
    results, futures = {}, {}
    key = ("source", None)
    try:
        references = {}
        for name, old_name in source_names.items():
            for seed in protocol.seeds:
                key = (name, seed)
                reference = json.loads((source / f"{old_name}_{seed}.json").read_text(encoding="utf-8"))
                if reference["scenario"] != old_name or reference["seed"] != seed:
                    raise ValueError("source result identity mismatch")
                references[key] = reference
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_value_budget, *task): (task[0], task[1].seed) for task in tasks}
            try:
                for future in as_completed(futures):
                    key = futures[future]
                    result = future.result()
                    if key in references:
                        reference = references[key]
                        if result["metrics"] != reference["metrics"]:
                            raise RuntimeError(f"replay metrics differ from source: {key}")
                        if result["fingerprint"] != reference["fingerprint"]:
                            raise RuntimeError(f"replay fingerprint differs from source: {key}")
                        result["source_verification"] = {"all_metrics_exact": True, "fingerprint_exact": True}
                    _write_json(output / f"{key[0]}_{key[1]}.json", result)
                    results[key] = result
                    checkpoint["completed_runs"] = [{"scenario": name, "seed": seed} for name, seed in sorted(results)]
                    checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
                    _write_json(output / "checkpoint.json", checkpoint)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        comparisons = []
        for seed in protocol.seeds:
            for reference_name in ("direction_only", "budget_only", "baseline"):
                left, right = results[(reference_name, seed)], results[("candidate", seed)]
                comparison = {"comparison": f"candidate_minus_{reference_name}", "seed": seed}
                for category in ("metrics", "exposure", "direction_counts", "direction_flows", "inventory_excess"):
                    if set(left[category]) != set(right[category]):
                        raise RuntimeError(f"paired diagnostic fields differ: {seed}/{category}")
                    comparison[category] = {
                        name: {"reference": left[category][name], "candidate": right[category][name],
                               "delta": right[category][name] - left[category][name]
                               if left[category][name] is not None and right[category][name] is not None else None}
                        for name in left[category]
                    }
                gap_values = []
                for run in (left, right):
                    summary = run["gap_reconstruction"]
                    values = {name: value for name, value in summary.items() if isinstance(value, (int, float))}
                    values.update({f"signed_gap_quantile_{quantile}": value for quantile, value in summary["signed_gap_quantiles"].items()})
                    values.update({f"{component}_{statistic}": value
                                   for component, statistics in summary["components"].items()
                                   for statistic, value in statistics.items()})
                    gap_values.append(values)
                comparison["gap_reconstruction"] = {
                    name: {"reference": gap_values[0][name], "candidate": gap_values[1][name],
                           "delta": gap_values[1][name] - gap_values[0][name]}
                    for name in gap_values[0]
                }
                comparisons.append(comparison)
        report = {"decision": "DEVELOPMENT_DIAGNOSTICS_ONLY", "stage1_complete": False,
                  "may_enter_stage2": False, "paired_seed_count": len(protocol.seeds),
                  "paired_comparison_count": len(comparisons),
                  "runs": [results[(name, item.seed)] for name, item, _ in tasks],
                  "paired_comparisons": comparisons}
        _write_json(output / "diagnostics.json", report)
        with (output / "per_seed_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            rows = [{"scenario": run["scenario"], "seed": run["seed"], **run["metrics"], **run["exposure"]} for run in report["runs"]]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint["status"] = "COMPLETE"
        _write_json(output / "checkpoint.json", checkpoint)
        return report
    except Exception as error:
        checkpoint["status"] = "ENGINEERING_ERROR"
        checkpoint["errors"] = [{"scenario": key[0], "seed": key[1], "type": type(error).__name__, "message": str(error)}]
        checkpoint["unfinished_runs"] = [{"scenario": name, "seed": item.seed} for name, item, _ in tasks if (name, item.seed) not in results]
        _write_json(output / "checkpoint.json", checkpoint)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=("paired", "price-discovery", "value-direction", "value-budget"), default="paired")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    config = load_stage1_config(args.config)
    protocol = MechanismProtocol.from_json(args.protocol)
    if args.mode in ("price-discovery", "value-direction", "value-budget"):
        if args.source is None:
            parser.error(f"--source is required for {args.mode} mode")
        run = (run_value_budget if args.mode == "value-budget" else
               run_value_direction if args.mode == "value-direction" else run_price_discovery)
        report = run(config, protocol, args.output, source=args.source, workers=args.workers)
    else:
        report = run_diagnostics(config, protocol, args.output, workers=args.workers)
    print(json.dumps({"output": str(args.output), "runs": len(report["runs"]),
                      "decision": report["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
