"""Cross-seed mechanism attribution for the Stage 1 artificial market."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import Stage1Config, load_stage1_config
from .harness import SimulationResult, run_stage1
from .manifest import canonical_json_bytes, sha256_file
from .runner import resolve_code_revision
from .stage1_freeze import verify_freeze_manifest

FloatArray = NDArray[np.float64]

PROTOCOL_KEYS = {
    "version",
    "stage",
    "purpose",
    "seeds",
    "excluded_development_seeds",
    "criteria",
    "minimum_burn_in_days",
    "minimum_evaluation_days",
}
CRITERION_NAMES = {
    "maximum_conservation_relative_error",
    "minimum_daily_volatility",
    "maximum_daily_volatility",
    "maximum_absolute_return_acf_1",
    "maximum_absolute_return_acf_1_20",
    "minimum_absolute_return_acf_1",
    "minimum_absolute_return_acf_5",
    "minimum_excess_kurtosis",
    "minimum_volume_absolute_return_correlation",
    "maximum_mean_absolute_log_price_gap",
    "minimum_value_only_volatility",
    "minimum_trend_only_volatility",
    "minimum_noise_only_volatility",
    "maximum_ablation_absolute_return_acf_1",
    "minimum_order_flow_imbalance_acf_1",
    "minimum_mean_spread_bps",
    "maximum_mean_spread_bps",
    "minimum_seed_pass_rate",
    "minimum_paired_seed_pass_rate",
    "minimum_news_channel_price_response_ratio",
    "minimum_agent_channel_price_response_ratio",
    "minimum_endogenous_tail_excess",
    "minimum_logit_strategy_turnover",
    "minimum_logit_fitness_alignment",
    "maximum_logit_daily_volatility",
}
SCENARIO_NAMES = (
    "full",
    "no_direct_news",
    "no_agent_information",
    "no_information_channels",
    "value_only",
    "trend_only",
    "noise_only",
    "no_logit",
    "no_activity_persistence",
    "independent_signals",
    "fixed_participation",
    "fixed_liquidity",
    "liquidity_stress",
    "concentrated_wealth",
    "garch_t_control",
    "logit_learning",
)

METRIC_NAMES = (
    "mean_return",
    "daily_volatility",
    "return_acf_1",
    "maximum_absolute_return_acf_1_20",
    "mean_absolute_return_acf_1_20",
    "absolute_return_acf_1",
    "absolute_return_acf_5",
    "absolute_return_acf_20",
    "absolute_return_acf_50",
    "excess_kurtosis",
    "volume_absolute_return_correlation",
    "volume_acf_1",
    "order_flow_imbalance_acf_1",
    "order_flow_imbalance_return_correlation",
    "mean_spread_bps",
    "mean_depth",
    "three_sigma_tail_fraction",
    "mean_absolute_log_price_gap",
    "maximum_drawdown",
    "bubble_day_fraction",
    "crash_day_fraction",
    "strategy_turnover",
    "final_wealth_gini",
    "price_cap_hits",
    "maximum_cash_relative_error",
    "maximum_share_relative_error",
    "return_skewness",
    "leverage_correlation",
    "learning_fitness_alignment",
)


@dataclass(frozen=True, slots=True)
class MechanismProtocol:
    version: str
    purpose: str
    seeds: tuple[int, ...]
    excluded_development_seeds: tuple[int, ...]
    criteria: Mapping[str, float]
    stage: str = "development"
    minimum_burn_in_days: int = 0
    minimum_evaluation_days: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "MechanismProtocol":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("mechanism protocol root must be a JSON object")
        missing = {"version", "purpose", "seeds", "excluded_development_seeds", "criteria"} - set(payload)
        unknown = set(payload) - PROTOCOL_KEYS
        if missing or unknown:
            raise ValueError(
                f"invalid mechanism protocol keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if not isinstance(payload["criteria"], dict):
            raise TypeError("protocol criteria must be a JSON object")
        criterion_names = set(payload["criteria"])
        unknown_criteria = criterion_names - CRITERION_NAMES
        if unknown_criteria:
            raise ValueError(
                f"unknown mechanism criteria: {sorted(unknown_criteria)}"
            )
        criteria = {
            str(name): float(value)
            for name, value in payload["criteria"].items()
        }
        if any(not np.isfinite(value) or value < 0.0 for value in criteria.values()):
            raise ValueError("protocol criteria must be finite and nonnegative")
        for rate_name in (
            "minimum_seed_pass_rate",
            "minimum_paired_seed_pass_rate",
        ):
            if criteria.get(rate_name, 1.0) > 1.0:
                raise ValueError(f"{rate_name} must not exceed one")
        seeds = tuple(int(value) for value in payload["seeds"])
        excluded = tuple(
            int(value) for value in payload["excluded_development_seeds"]
        )
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("protocol seeds must be nonempty and unique")
        if set(seeds) & set(excluded):
            raise ValueError(
                "formal seeds must be disjoint from development seeds"
            )
        minimum_burn_in_days = int(
            payload.get("minimum_burn_in_days", 0)
        )
        minimum_evaluation_days = int(
            payload.get("minimum_evaluation_days", 0)
        )
        stage = str(payload.get("stage", "development"))
        if stage not in ("development", "formal"):
            raise ValueError("protocol stage must be development or formal")
        if minimum_burn_in_days < 0 or minimum_evaluation_days < 0:
            raise ValueError("protocol day minimums must be nonnegative")
        return cls(
            version=str(payload["version"]),
            purpose=str(payload["purpose"]),
            seeds=seeds,
            excluded_development_seeds=excluded,
            criteria=criteria,
            stage=stage,
            minimum_burn_in_days=minimum_burn_in_days,
            minimum_evaluation_days=minimum_evaluation_days,
        )


def _correlation(left: FloatArray, right: FloatArray) -> float:
    centered_left = left - float(left.mean())
    centered_right = right - float(right.mean())
    denominator = float(
        np.sqrt(
            np.dot(centered_left, centered_left)
            * np.dot(centered_right, centered_right)
        )
    )
    if denominator <= 0:
        return 0.0
    return float(np.dot(centered_left, centered_right) / denominator)


def _autocorrelation(values: FloatArray, lag: int) -> float:
    if lag < 1:
        raise ValueError("autocorrelation lag must be positive")
    if values.size <= lag + 1:
        return 0.0
    return _correlation(values[:-lag], values[lag:])


def _lag_one_autocorrelation(values: FloatArray) -> float:
    return _autocorrelation(values, 1)


def _gini(values: FloatArray) -> float:
    if values.size == 0:
        raise ValueError("gini values must be nonempty")
    if np.any(values < -1e-6):
        raise ValueError("gini values contain material negative wealth")
    # Final marked wealth may carry float crumbs below zero within the
    # harness audit tolerance; clamp before measuring inequality.
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        2.0 * np.dot(ranks, ordered) / (ordered.size * total)
        - (ordered.size + 1.0) / ordered.size
    )


def result_metrics(
    result: SimulationResult,
    burn_in_days: int = 0,
) -> dict[str, float]:
    if burn_in_days < 0 or burn_in_days >= result.volumes.size:
        raise ValueError("burn_in_days must be in [0, trading_days)")
    evaluation_days = slice(burn_in_days, result.volumes.size)
    boundary_prices = result.prices[burn_in_days:]
    boundary_fundamentals = result.fundamentals[burn_in_days:]
    returns = np.diff(np.log(boundary_prices))
    volumes = result.volumes[evaluation_days]
    order_flow_imbalance = result.order_flow_imbalance[evaluation_days]
    spreads = result.spreads[evaluation_days]
    depths = result.depths[evaluation_days]
    closing_prices = boundary_prices[1:]
    closing_fundamentals = boundary_fundamentals[1:]
    expected_length = result.volumes.size - burn_in_days
    aligned = (
        returns,
        volumes,
        order_flow_imbalance,
        spreads,
        depths,
        closing_prices,
        closing_fundamentals,
    )
    if any(values.size != expected_length for values in aligned):
        raise ValueError("evaluation arrays are not aligned by trading day")
    volatility = float(np.std(returns, ddof=0))
    if volatility > 0:
        standardized = (returns - float(returns.mean())) / volatility
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
        return_skewness = float(np.mean(standardized**3))
    else:
        excess_kurtosis = 0.0
        return_skewness = 0.0
    if returns.size > 1 and volatility > 0:
        leverage_correlation = _correlation(
            returns[:-1], np.abs(returns[1:])
        )
    else:
        leverage_correlation = 0.0
    running_peak = np.maximum.accumulate(boundary_prices)
    drawdowns = boundary_prices / running_peak - 1.0
    log_gap = np.log(closing_prices / closing_fundamentals)
    strategy_updates = sum(
        audit.updated_agents
        for audit in result.learning_audits
        if audit.day > burn_in_days
    )
    final_wealth = (
        result.final_agent_cash
        + result.final_agent_positions * result.prices[-1]
    )
    initial_cash = float(result.total_cash[0])
    initial_shares = float(result.total_shares[0])
    return_acf_values = np.array(
        [_autocorrelation(returns, lag) for lag in range(1, 21)],
        dtype=np.float64,
    )
    return {
        "mean_return": float(np.mean(returns)),
        "daily_volatility": volatility,
        "return_acf_1": _lag_one_autocorrelation(returns),
        "maximum_absolute_return_acf_1_20": float(
            np.max(np.abs(return_acf_values))
        ),
        "mean_absolute_return_acf_1_20": float(
            np.mean(np.abs(return_acf_values))
        ),
        "absolute_return_acf_1": _lag_one_autocorrelation(
            np.abs(returns)
        ),
        "absolute_return_acf_5": _autocorrelation(
            np.abs(returns), 5
        ),
        "absolute_return_acf_20": _autocorrelation(
            np.abs(returns), 20
        ),
        "absolute_return_acf_50": _autocorrelation(
            np.abs(returns), 50
        ),
        "excess_kurtosis": excess_kurtosis,
        "return_skewness": return_skewness,
        "leverage_correlation": leverage_correlation,
        "volume_absolute_return_correlation": _correlation(
            volumes,
            np.abs(returns),
        ),
        "volume_acf_1": _lag_one_autocorrelation(volumes),
        "order_flow_imbalance_acf_1": _lag_one_autocorrelation(
            order_flow_imbalance
        ),
        "order_flow_imbalance_return_correlation": _correlation(
            order_flow_imbalance,
            returns,
        ),
        "mean_spread_bps": float(
            np.mean(spreads / closing_prices) * 1e4
        ),
        "mean_depth": float(np.mean(depths)),
        "three_sigma_tail_fraction": (
            float(np.mean(np.abs(returns - float(np.mean(returns)))
                          > 3.0 * volatility))
            if volatility > 0
            else 0.0
        ),
        "mean_absolute_log_price_gap": float(np.mean(np.abs(log_gap))),
        "maximum_drawdown": float(-np.min(drawdowns)),
        "bubble_day_fraction": float(np.mean(log_gap > 0.10)),
        "crash_day_fraction": float(np.mean(returns < -0.05)),
        "strategy_turnover": float(strategy_updates),
        "final_wealth_gini": _gini(final_wealth),
        "learning_fitness_alignment": (
            float(
                np.mean(
                    [
                        int(
                            int(np.argmax(audit.choice_probabilities))
                            == int(np.argmax(audit.mean_fitness_by_strategy))
                        )
                        for audit in result.learning_audits
                        if audit.day > burn_in_days
                    ]
                )
            )
            if any(
                audit.day > burn_in_days for audit in result.learning_audits
            )
            else 1.0
        ),
        "price_cap_hits": float(
            sum(int(audit.price_cap_hit) for audit in result.daily_audits)
        ),
        "maximum_cash_relative_error": float(
            np.max(np.abs(result.total_cash - initial_cash)) / initial_cash
        ),
        "maximum_share_relative_error": float(
            np.max(np.abs(result.total_shares - initial_shares))
            / initial_shares
        ),
    }


def build_scenarios(config: Stage1Config) -> dict[str, Stage1Config]:
    """Create one-variable or clearly labelled mechanism controls."""
    no_agent_information = {
        "value_sensitivity": 0.0,
        "value_update_rate": 0.0,
        "initial_subjective_value_dispersion": 0.0,
        "information_response_dispersion": 0.0,
    }
    return {
        "full": config,
        "no_direct_news": replace(
            config,
            public_news_price_pass_through=0.0,
        ),
        "no_agent_information": replace(
            config,
            **no_agent_information,
        ),
        "no_information_channels": replace(
            config,
            public_news_price_pass_through=0.0,
            **no_agent_information,
        ),
        "value_only": replace(
            config,
            strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
            learning_enabled=False,
        ),
        "trend_only": replace(
            config,
            strategy_shares={"value": 0.0, "trend": 1.0, "noise": 0.0},
            learning_enabled=False,
        ),
        "noise_only": replace(
            config,
            strategy_shares={"value": 0.0, "trend": 0.0, "noise": 1.0},
            learning_enabled=False,
        ),
        "no_logit": replace(config, learning_enabled=False),
        "no_activity_persistence": replace(
            config,
            activity_persistence=0.0,
            activity_shock_scale=0.0,
        ),
        "independent_signals": replace(
            config,
            common_signal_correlation=0.0,
        ),
        "fixed_participation": replace(
            config,
            base_activity_rate=1.0,
            activity_rate_dispersion=0.0,
            activity_volatility_sensitivity=0.0,
            activity_signal_sensitivity=0.0,
            activity_persistence=0.0,
            activity_shock_scale=0.0,
        ),
        "fixed_liquidity": replace(
            config,
            liquidity_volatility_sensitivity=0.0,
            minimum_liquidity_fraction=1.0,
        ),
        "liquidity_stress": replace(
            config,
            liquidity_volatility_sensitivity=2.0,
            liquidity_stress_threshold=0.0,
            minimum_liquidity_fraction=0.25,
            depth_resilience=0.08,
        ),
        "concentrated_wealth": replace(
            config,
            initial_wealth_dispersion=1.0,
        ),
        "garch_t_control": replace(
            config,
            fundamental_process="garch_t",
            fundamental_shock_df=5.5,
            fundamental_arch=0.11,
            fundamental_garch=0.88,
        ),
        "logit_learning": replace(
            config,
            learning_enabled=True,
        ),
    }


def _summarize(
    rows: Sequence[Mapping[str, float | int | str]],
) -> dict[str, dict[str, dict[str, float]]]:
    scenario_names = tuple(dict.fromkeys(str(row["scenario"]) for row in rows))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in scenario_names:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in METRIC_NAMES:
            values = np.array(
                [float(row[metric]) for row in scenario_rows],
                dtype=np.float64,
            )
            metric_summary[metric] = {
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q90": float(np.quantile(values, 0.90)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        summary[scenario] = metric_summary
    return summary


def evaluate_gate(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    criteria: Mapping[str, float],
    rows: Sequence[Mapping[str, float | int | str]] | None = None,
    scenario_configs: Mapping[str, Stage1Config] | None = None,
) -> dict[str, object]:
    missing_scenarios = set(SCENARIO_NAMES) - set(summary)
    if missing_scenarios:
        raise ValueError(
            f"mechanism summary is missing scenarios: {sorted(missing_scenarios)}"
        )
    full = summary["full"]

    def median(scenario: str, metric: str) -> float:
        return float(summary[scenario][metric]["median"])

    def criterion(name: str, default: float) -> float:
        return float(criteria.get(name, default))

    minimum_pass_rate = criterion("minimum_seed_pass_rate", 0.8)
    minimum_paired_pass_rate = criterion(
        "minimum_paired_seed_pass_rate", 0.8
    )
    rows_by_scenario = {
        scenario: (
            [row for row in rows if row["scenario"] == scenario]
            if rows is not None
            else []
        )
        for scenario in SCENARIO_NAMES
    }
    evidence: dict[str, dict[str, object]] = {}

    def scenario_evidence(
        name: str,
        scenario: str,
        predicate: Callable[[Mapping[str, float | int | str]], bool],
        *,
        observed: float,
        threshold: object,
    ) -> bool:
        scenario_rows = rows_by_scenario[scenario]
        if rows is not None and not scenario_rows:
            raise ValueError(f"mechanism rows are missing scenario {scenario}")
        outcomes = {
            int(row["seed"]): bool(predicate(row))
            for row in scenario_rows
        }
        pass_rate = (
            float(np.mean(tuple(outcomes.values()))) if outcomes else 1.0
        )
        evidence[name] = {
            "kind": "scenario",
            "scenario": scenario,
            "observed": observed,
            "threshold": threshold,
            "pass_rate": pass_rate,
            "required_pass_rate": minimum_pass_rate,
            "passed_seeds": [seed for seed, ok in outcomes.items() if ok],
            "failed_seeds": [seed for seed, ok in outcomes.items() if not ok],
        }
        return pass_rate >= minimum_pass_rate

    def paired_values(
        left_scenario: str,
        right_scenario: str,
        metric: str,
    ) -> list[tuple[int, float, float]]:
        if rows is None:
            return []
        left = {
            int(row["seed"]): float(row[metric])
            for row in rows_by_scenario[left_scenario]
        }
        right = {
            int(row["seed"]): float(row[metric])
            for row in rows_by_scenario[right_scenario]
        }
        if not left or set(left) != set(right):
            raise ValueError(
                f"paired scenarios {left_scenario} and {right_scenario} "
                "must use identical seeds"
            )
        return [(seed, left[seed], right[seed]) for seed in sorted(left)]

    def paired_ratio_evidence(
        name: str,
        baseline_scenario: str,
        ablated_scenario: str,
        metric: str,
        threshold: float,
    ) -> bool:
        pairs = paired_values(baseline_scenario, ablated_scenario, metric)
        ratios = {
            seed: ablated / max(abs(baseline), 1e-12)
            for seed, baseline, ablated in pairs
        }
        outcomes = {seed: ratio >= threshold for seed, ratio in ratios.items()}
        pass_rate = (
            float(np.mean(tuple(outcomes.values()))) if outcomes else 1.0
        )
        median_ratio = float(np.median(tuple(ratios.values()))) if ratios else 1.0
        evidence[name] = {
            "kind": "paired_ratio",
            "baseline_scenario": baseline_scenario,
            "ablated_scenario": ablated_scenario,
            "metric": metric,
            "median_ratio": median_ratio,
            "threshold": threshold,
            "pass_rate": pass_rate,
            "required_pass_rate": minimum_paired_pass_rate,
            "per_seed": {str(seed): ratio for seed, ratio in ratios.items()},
            "passed_seeds": [seed for seed, ok in outcomes.items() if ok],
            "failed_seeds": [seed for seed, ok in outcomes.items() if not ok],
        }
        return (
            pass_rate >= minimum_paired_pass_rate
            and median_ratio >= threshold
        )

    def paired_difference_evidence(
        name: str,
        stronger_scenario: str,
        weaker_scenario: str,
        metric: str,
        threshold: float,
    ) -> bool:
        pairs = paired_values(stronger_scenario, weaker_scenario, metric)
        differences = {
            seed: stronger - weaker
            for seed, stronger, weaker in pairs
        }
        outcomes = {
            seed: difference >= threshold
            for seed, difference in differences.items()
        }
        pass_rate = (
            float(np.mean(tuple(outcomes.values()))) if outcomes else 1.0
        )
        median_difference = (
            float(np.median(tuple(differences.values())))
            if differences
            else threshold
        )
        evidence[name] = {
            "kind": "paired_difference",
            "stronger_scenario": stronger_scenario,
            "weaker_scenario": weaker_scenario,
            "metric": metric,
            "median_difference": median_difference,
            "threshold": threshold,
            "pass_rate": pass_rate,
            "required_pass_rate": minimum_paired_pass_rate,
            "per_seed": {
                str(seed): difference
                for seed, difference in differences.items()
            },
            "passed_seeds": [seed for seed, ok in outcomes.items() if ok],
            "failed_seeds": [seed for seed, ok in outcomes.items() if not ok],
        }
        return (
            pass_rate >= minimum_paired_pass_rate
            and median_difference >= threshold
        )
    bounded_scenarios = (
        "value_only",
        "trend_only",
        "noise_only",
        "independent_signals",
        "fixed_participation",
        "fixed_liquidity",
        "liquidity_stress",
    )
    volatility_seed_pass = scenario_evidence(
        "full_volatility_in_daily_range",
        "full",
        lambda row: (
            criterion("minimum_daily_volatility", 0.005)
            <= float(row["daily_volatility"])
            <= criterion("maximum_daily_volatility", 0.04)
        ),
        observed=median("full", "daily_volatility"),
        threshold={
            "minimum": criterion("minimum_daily_volatility", 0.005),
            "maximum": criterion("maximum_daily_volatility", 0.04),
        },
    )
    autocorrelation_seed_pass = scenario_evidence(
        "full_return_autocorrelation_near_zero_across_lags",
        "full",
        lambda row: (
            abs(float(row["return_acf_1"]))
            <= criterion("maximum_absolute_return_acf_1", 0.15)
            and float(row["maximum_absolute_return_acf_1_20"])
            <= criterion("maximum_absolute_return_acf_1_20", 0.20)
        ),
        observed=median("full", "maximum_absolute_return_acf_1_20"),
        threshold=criterion("maximum_absolute_return_acf_1_20", 0.20),
    )
    clustering_seed_pass = scenario_evidence(
        "full_volatility_clustering_has_decay",
        "full",
        lambda row: (
            float(row["absolute_return_acf_1"])
            >= criterion("minimum_absolute_return_acf_1", 0.03)
            and float(row["absolute_return_acf_5"])
            >= criterion("minimum_absolute_return_acf_5", 0.01)
            and float(row["absolute_return_acf_20"])
            < float(row["absolute_return_acf_1"])
        ),
        observed=median("full", "absolute_return_acf_1"),
        threshold=criterion("minimum_absolute_return_acf_1", 0.03),
    )
    tail_seed_pass = scenario_evidence(
        "full_heavier_than_gaussian",
        "full",
        lambda row: float(row["excess_kurtosis"])
        >= criterion("minimum_excess_kurtosis", 0.10),
        observed=median("full", "excess_kurtosis"),
        threshold=criterion("minimum_excess_kurtosis", 0.10),
    )
    volume_relation_seed_pass = scenario_evidence(
        "full_volume_volatility_relation_positive",
        "full",
        lambda row: float(row["volume_absolute_return_correlation"])
        >= criterion("minimum_volume_absolute_return_correlation", 0.10),
        observed=median("full", "volume_absolute_return_correlation"),
        threshold=criterion("minimum_volume_absolute_return_correlation", 0.10),
    )
    price_discovery_seed_pass = scenario_evidence(
        "full_price_discovery_bounded",
        "full",
        lambda row: float(row["mean_absolute_log_price_gap"])
        <= criterion("maximum_mean_absolute_log_price_gap", 0.10),
        observed=median("full", "mean_absolute_log_price_gap"),
        threshold=criterion("maximum_mean_absolute_log_price_gap", 0.10),
    )
    endogenous_tail_pass = paired_difference_evidence(
        "endogenous_liquidity_feedback_increases_tail_weight",
        "full",
        "fixed_liquidity",
        "excess_kurtosis",
        criterion("minimum_endogenous_tail_excess", 0.0),
    )
    news_threshold = criterion(
        "minimum_news_channel_price_response_ratio", 1.0
    )
    public_news_marginal = paired_ratio_evidence(
        "public_news_marginal_response",
        "full",
        "no_direct_news",
        "mean_absolute_log_price_gap",
        news_threshold,
    )
    public_news_independent = paired_ratio_evidence(
        "public_news_independent_response",
        "no_agent_information",
        "no_information_channels",
        "mean_absolute_log_price_gap",
        news_threshold,
    )
    agent_threshold = criterion(
        "minimum_agent_channel_price_response_ratio", 1.0
    )
    agent_marginal = paired_ratio_evidence(
        "agent_information_marginal_response",
        "full",
        "no_agent_information",
        "mean_absolute_log_price_gap",
        agent_threshold,
    )
    agent_independent = paired_ratio_evidence(
        "agent_information_independent_response",
        "no_direct_news",
        "no_information_channels",
        "mean_absolute_log_price_gap",
        agent_threshold,
    )
    empty_information_pass = paired_ratio_evidence(
        "empty_information_baseline_is_weakest_discovery",
        "full",
        "no_information_channels",
        "mean_absolute_log_price_gap",
        1.0,
    )
    strategy_thresholds = {
        "value_only": criterion("minimum_value_only_volatility", 0.001),
        "trend_only": criterion("minimum_trend_only_volatility", 0.002),
        "noise_only": criterion("minimum_noise_only_volatility", 0.003),
    }
    single_strategy_outcomes = [
        scenario_evidence(
            f"{scenario}_remains_active",
            scenario,
            lambda row, minimum=minimum: float(row["daily_volatility"])
            >= minimum,
            observed=median(scenario, "daily_volatility"),
            threshold=minimum,
        )
        for scenario, minimum in strategy_thresholds.items()
    ]
    single_strategy_seed_pass = all(single_strategy_outcomes)
    ablation_outcomes = [
        scenario_evidence(
            f"{scenario}_avoids_pathological_return_predictability",
            scenario,
            lambda row: abs(float(row["return_acf_1"]))
            <= criterion("maximum_ablation_absolute_return_acf_1", 0.35),
            observed=abs(median(scenario, "return_acf_1")),
            threshold=criterion(
                "maximum_ablation_absolute_return_acf_1", 0.35
            ),
        )
        for scenario in bounded_scenarios
    ]
    ablation_seed_pass = all(ablation_outcomes)
    spread_depth_seed_pass = scenario_evidence(
        "spread_and_depth_are_finite",
        "full",
        lambda row: (
            criterion("minimum_mean_spread_bps", 1.0)
            <= float(row["mean_spread_bps"])
            <= criterion("maximum_mean_spread_bps", 25.0)
            and float(row["mean_depth"]) > 0.0
        ),
        observed=median("full", "mean_spread_bps"),
        threshold={
            "minimum": criterion("minimum_mean_spread_bps", 1.0),
            "maximum": criterion("maximum_mean_spread_bps", 25.0),
        },
    )
    liquidity_stress_seed_pass = paired_difference_evidence(
        "liquidity_stress_increases_clustering",
        "liquidity_stress",
        "full",
        "absolute_return_acf_1",
        0.0,
    )
    checks = {
        "full_runs_without_price_cap": (
            float(full["price_cap_hits"]["maximum"]) == 0.0
        ),
        "cash_conservation": all(
            float(metrics["maximum_cash_relative_error"]["maximum"])
            <= criterion("maximum_conservation_relative_error", 1e-8)
            for metrics in summary.values()
        ),
        "share_conservation": all(
            float(metrics["maximum_share_relative_error"]["maximum"])
            <= criterion("maximum_conservation_relative_error", 1e-8)
            for metrics in summary.values()
        ),
        "full_volatility_in_daily_range": (
            criterion("minimum_daily_volatility", 0.005)
            <= float(full["daily_volatility"]["median"])
            <= criterion("maximum_daily_volatility", 0.04)
            and volatility_seed_pass
        ),
        "full_return_autocorrelation_near_zero_across_lags": (
            abs(float(full["return_acf_1"]["median"]))
            <= criterion("maximum_absolute_return_acf_1", 0.15)
            and float(
                full["maximum_absolute_return_acf_1_20"]["median"]
            )
            <= criterion(
                "maximum_absolute_return_acf_1_20", 0.20
            )
            and autocorrelation_seed_pass
        ),
        "full_volatility_clustering_has_decay": (
            float(full["absolute_return_acf_1"]["median"])
            >= criterion("minimum_absolute_return_acf_1", 0.03)
            and float(full["absolute_return_acf_5"]["median"])
            >= criterion("minimum_absolute_return_acf_5", 0.01)
            and float(full["absolute_return_acf_20"]["median"])
            < float(full["absolute_return_acf_1"]["median"])
            and clustering_seed_pass
        ),
        "full_heavier_than_gaussian": (
            float(full["excess_kurtosis"]["median"])
            >= criterion("minimum_excess_kurtosis", 0.10)
            and tail_seed_pass
        ),
        "endogenous_liquidity_feedback_increases_tail_weight": (
            endogenous_tail_pass
        ),
        "full_volume_volatility_relation_positive": (
            float(full["volume_absolute_return_correlation"]["median"])
            >= criterion(
                "minimum_volume_absolute_return_correlation", 0.10
            )
            and volume_relation_seed_pass
        ),
        "full_price_discovery_bounded": (
            float(full["mean_absolute_log_price_gap"]["median"])
            <= criterion("maximum_mean_absolute_log_price_gap", 0.10)
            and price_discovery_seed_pass
        ),
        "public_news_channel_has_paired_price_response": (
            public_news_marginal and public_news_independent
        ),
        "agent_information_channel_has_paired_price_response": (
            agent_marginal and agent_independent
        ),
        "empty_information_baseline_is_weakest_discovery": (
            empty_information_pass
        ),
        "single_strategy_markets_remain_active": (
            median("value_only", "daily_volatility")
            >= criterion("minimum_value_only_volatility", 0.001)
            and median("trend_only", "daily_volatility")
            >= criterion("minimum_trend_only_volatility", 0.002)
            and median("noise_only", "daily_volatility")
            >= criterion("minimum_noise_only_volatility", 0.003)
            and single_strategy_seed_pass
        ),
        "ablations_avoid_pathological_return_predictability": (
            all(
                abs(median(scenario, "return_acf_1"))
                <= criterion(
                    "maximum_ablation_absolute_return_acf_1", 0.35
                )
                for scenario in bounded_scenarios
            )
            and ablation_seed_pass
        ),
        "fixed_population_has_no_strategy_turnover": (
            median("full", "strategy_turnover") == 0.0
            and median("no_logit", "strategy_turnover") == 0.0
        ),
        "logit_imitation_fires_and_tracks_fitness": (
            median("logit_learning", "strategy_turnover")
            >= criterion("minimum_logit_strategy_turnover", 1.0)
            and float(
                summary["logit_learning"]["learning_fitness_alignment"][
                    "median"
                ]
            )
            >= criterion("minimum_logit_fitness_alignment", 0.99)
            and scenario_evidence(
                "logit_imitation_fires_and_tracks_fitness",
                "logit_learning",
                lambda row: (
                    float(row["strategy_turnover"])
                    >= criterion("minimum_logit_strategy_turnover", 1.0)
                    and float(row["learning_fitness_alignment"])
                    >= criterion("minimum_logit_fitness_alignment", 0.99)
                ),
                observed=median("logit_learning", "strategy_turnover"),
                threshold=criterion("minimum_logit_strategy_turnover", 1.0),
            )
        ),
        "logit_market_remains_active_and_bounded": scenario_evidence(
            "logit_market_remains_active_and_bounded",
            "logit_learning",
            lambda row: (
                criterion("minimum_daily_volatility", 0.005)
                <= float(row["daily_volatility"])
                <= criterion("maximum_logit_daily_volatility", 0.04)
                and float(row["price_cap_hits"]) == 0.0
            ),
            observed=median("logit_learning", "daily_volatility"),
            threshold={
                "minimum": criterion("minimum_daily_volatility", 0.005),
                "maximum": criterion("maximum_logit_daily_volatility", 0.04),
            },
        ),
        "persistent_order_flow_is_observable": (
            float(full["order_flow_imbalance_acf_1"]["median"])
            >= criterion(
                "minimum_order_flow_imbalance_acf_1", 0.05
            )
            and scenario_evidence(
                "persistent_order_flow_is_observable",
                "full",
                lambda row: float(row["order_flow_imbalance_acf_1"])
                >= criterion("minimum_order_flow_imbalance_acf_1", 0.05),
                observed=median("full", "order_flow_imbalance_acf_1"),
                threshold=criterion(
                    "minimum_order_flow_imbalance_acf_1", 0.05
                ),
            )
        ),
        "spread_and_depth_are_finite": (
            criterion("minimum_mean_spread_bps", 1.0)
            <= float(full["mean_spread_bps"]["median"])
            <= criterion("maximum_mean_spread_bps", 25.0)
            and float(full["mean_depth"]["minimum"]) > 0.0
            and spread_depth_seed_pass
        ),
        "liquidity_stress_increases_clustering": (
            median("liquidity_stress", "absolute_return_acf_1")
            > median("full", "absolute_return_acf_1")
            and liquidity_stress_seed_pass
        ),
        "garch_control_is_labelled_exogenous": (
            scenario_configs is not None
            and scenario_configs["full"].fundamental_process != "garch_t"
            and scenario_configs["garch_t_control"].fundamental_process
            == "garch_t"
        ),
    }
    return {
        "checks": checks,
        "evidence": evidence,
        "minimum_seed_pass_rate": minimum_pass_rate,
        "minimum_paired_seed_pass_rate": minimum_paired_pass_rate,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "passed": all(checks.values()),
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, float | int | str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("scenario", "seed", *METRIC_NAMES),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_gate_evidence_csv(
    path: Path,
    evidence: Mapping[str, Mapping[str, object]],
) -> None:
    fieldnames = (
        "check",
        "kind",
        "comparison",
        "observed",
        "threshold",
        "pass_rate",
        "required_pass_rate",
        "failed_seeds",
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in evidence.items():
            if item.get("scenario") is not None:
                comparison = str(item["scenario"])
            elif item.get("baseline_scenario") is not None:
                comparison = (
                    f"{item.get('baseline_scenario')} -> "
                    f"{item.get('ablated_scenario')}"
                )
            else:
                comparison = (
                    f"{item.get('stronger_scenario')} - "
                    f"{item.get('weaker_scenario')}"
                )
            observed = item.get(
                "observed",
                item.get("median_ratio", item.get("median_difference")),
            )
            writer.writerow(
                {
                    "check": name,
                    "kind": item.get("kind"),
                    "comparison": comparison,
                    "observed": json.dumps(observed, sort_keys=True),
                    "threshold": json.dumps(
                        item.get("threshold"), sort_keys=True
                    ),
                    "pass_rate": item.get("pass_rate"),
                    "required_pass_rate": item.get("required_pass_rate"),
                    "failed_seeds": ";".join(
                        str(seed) for seed in item.get("failed_seeds", [])
                    ),
                }
            )


def _write_figure(
    path: Path,
    rows: Sequence[Mapping[str, float | int | str]],
) -> None:
    scenarios = tuple(dict.fromkeys(str(row["scenario"]) for row in rows))
    displayed = (
        "full",
        "no_direct_news",
        "no_agent_information",
        "no_information_channels",
        "value_only",
        "trend_only",
        "noise_only",
        "independent_signals",
        "fixed_participation",
        "fixed_liquidity",
        "liquidity_stress",
        "garch_t_control",
    )
    displayed = tuple(name for name in displayed if name in scenarios)
    panels = (
        ("daily_volatility", "Daily volatility"),
        (
            "maximum_absolute_return_acf_1_20",
            "Maximum |return ACF|, lags 1–20",
        ),
        ("absolute_return_acf_1", "|Return| ACF(1)"),
        ("absolute_return_acf_20", "|Return| ACF(20)"),
        ("excess_kurtosis", "Excess kurtosis"),
        ("mean_spread_bps", "Mean spread (bps)"),
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 9), constrained_layout=True
    )
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        values = [
            [
                float(row[metric])
                for row in rows
                if row["scenario"] == scenario
            ]
            for scenario in displayed
        ]
        axis.boxplot(values, tick_labels=displayed, showfliers=False)
        axis.axhline(0.0, color="#888888", linewidth=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Stage 1 mechanism attribution across frozen seeds",
        fontsize=15,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_markdown(
    path: Path,
    *,
    protocol: MechanismProtocol,
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    gate: Mapping[str, object],
) -> None:
    lines = [
        "# Stage 1 mechanism validation",
        "",
        f"- Protocol: `{protocol.version}`",
        f"- Frozen seeds: `{len(protocol.seeds)}`",
        f"- Decision: `{'PASS' if gate['passed'] else 'FAIL'}`",
        "",
        "## Gate checks",
        "",
    ]
    checks = gate["checks"]
    for name, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Gate evidence",
            "",
            "| Check | Observed | Threshold | Pass rate | Failed seeds |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, item in gate.get("evidence", {}).items():
        observed = item.get(
            "observed",
            item.get("median_ratio", item.get("median_difference")),
        )
        threshold = item.get("threshold")
        pass_rate = float(item.get("pass_rate", 1.0))
        failed = ", ".join(
            str(seed) for seed in item.get("failed_seeds", [])
        ) or "none"
        lines.append(
            f"| `{name}` | `{json.dumps(observed, sort_keys=True)}` "
            f"| `{json.dumps(threshold, sort_keys=True)}` "
            f"| {pass_rate:.1%} | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Scenario medians",
            "",
            "| Scenario | Volatility | Return ACF | |Return| ACF | "
            "Excess kurtosis | Volume-|return| corr | Price gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, metrics in summary.items():
        lines.append(
            f"| {scenario} "
            f"| {metrics['daily_volatility']['median']:.6f} "
            f"| {metrics['return_acf_1']['median']:.6f} "
            f"| {metrics['absolute_return_acf_1']['median']:.6f} "
            f"| {metrics['excess_kurtosis']['median']:.6f} "
            f"| {metrics['volume_absolute_return_correlation']['median']:.6f} "
            f"| {metrics['mean_absolute_log_price_gap']['median']:.6f} |"
        )
    full = summary["full"]
    lines.extend(
        [
            "",
            "## Full-market structural metrics",
            "",
            "| Metric | Median |",
            "|---|---:|",
            f"| Maximum absolute return ACF, lags 1–20 | "
            f"{full['maximum_absolute_return_acf_1_20']['median']:.6f} |",
            f"| Absolute-return ACF(5) | "
            f"{full['absolute_return_acf_5']['median']:.6f} |",
            f"| Absolute-return ACF(20) | "
            f"{full['absolute_return_acf_20']['median']:.6f} |",
            f"| Absolute-return ACF(50) | "
            f"{full['absolute_return_acf_50']['median']:.6f} |",
            f"| Volume ACF(1) | "
            f"{full['volume_acf_1']['median']:.6f} |",
            f"| OFI ACF(1) | "
            f"{full['order_flow_imbalance_acf_1']['median']:.6f} |",
            f"| OFI-return correlation | "
            f"{full['order_flow_imbalance_return_correlation']['median']:.6f} |",
            f"| Mean spread (bps) | "
            f"{full['mean_spread_bps']['median']:.6f} |",
            f"| Three-sigma tail fraction | "
            f"{full['three_sigma_tail_fraction']['median']:.6f} |",
            f"| Return skewness | "
            f"{full['return_skewness']['median']:.6f} |",
            f"| Leverage correlation, r(t) vs |r|(t+1) | "
            f"{full['leverage_correlation']['median']:.6f} |",
            f"| Crash-day fraction, r < -5% | "
            f"{full['crash_day_fraction']['median']:.6f} |",
            f"| Bubble-day fraction, log gap > 10% | "
            f"{full['bubble_day_fraction']['median']:.6f} |",
            f"| Maximum drawdown | "
            f"{full['maximum_drawdown']['median']:.6f} |",
            f"| Final wealth Gini | "
            f"{full['final_wealth_gini']['median']:.6f} |",
        ]
    )
    lines.extend(
        [
            "",
            "The GARCH-t scenario is an explicitly exogenous stress control. "
            "It is not counted as evidence that Agent interaction generated "
            "fat tails or volatility clustering.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_metric_task(
    task: tuple[str, dict[str, object], int],
) -> dict[str, float | int | str]:
    scenario_name, config_payload, seed = task
    seeded_config = replace(
        Stage1Config.from_dict(config_payload), seed=seed
    )
    result = run_stage1(seeded_config)
    return {
        "scenario": scenario_name,
        "seed": seed,
        **result_metrics(result, burn_in_days=seeded_config.burn_in_days),
    }


def _protocol_identity(
    config: Stage1Config,
    protocol: MechanismProtocol,
) -> str:
    package_root = Path(__file__).resolve().parent
    implementation_files = (
        package_root / "config.py",
        package_root / "population.py",
        package_root / "policies.py",
        package_root / "harness.py",
        package_root / "microstructure.py",
        package_root / "mechanism_validation.py",
    )
    payload = {
        "config": config.to_dict(),
        "protocol": {
            "version": protocol.version,
            "stage": protocol.stage,
            "purpose": protocol.purpose,
            "seeds": list(protocol.seeds),
            "excluded_development_seeds": list(
                protocol.excluded_development_seeds
            ),
            "criteria": dict(protocol.criteria),
            "minimum_burn_in_days": protocol.minimum_burn_in_days,
            "minimum_evaluation_days": protocol.minimum_evaluation_days,
        },
        "scenarios": list(SCENARIO_NAMES),
        "implementation_sha256": {
            path.name: sha256_file(path) for path in implementation_files
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_checkpoint(
    path: Path,
    expected_identity: str,
) -> list[dict[str, float | int | str]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != expected_identity:
        raise ValueError("checkpoint identity does not match config and protocol")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("checkpoint rows must be a JSON array")
    return [dict(row) for row in rows]


def _write_checkpoint(
    path: Path,
    identity: str,
    rows: Sequence[Mapping[str, float | int | str]],
) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    payload = {
        "schema_version": "1.0.0",
        "identity": identity,
        "rows": list(rows),
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def run_protocol(
    config: Stage1Config,
    protocol: MechanismProtocol,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    workers: int = 1,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, object],
]:
    if config.burn_in_days < protocol.minimum_burn_in_days:
        raise ValueError(
            "config burn_in_days is below the protocol minimum"
        )
    evaluation_days = config.trading_days - config.burn_in_days
    if evaluation_days < protocol.minimum_evaluation_days:
        raise ValueError(
            "config evaluation period is below the protocol minimum"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    scenario_configs = build_scenarios(config)
    if tuple(scenario_configs) != SCENARIO_NAMES:
        raise ValueError("mechanism scenario set does not match the protocol")
    identity = _protocol_identity(config, protocol)
    if resume and checkpoint_path is None:
        raise ValueError("resume requires a checkpoint path")
    rows = (
        _load_checkpoint(checkpoint_path, identity)
        if resume and checkpoint_path is not None
        else []
    )
    completed = {
        (str(row["scenario"]), int(row["seed"])) for row in rows
    }
    tasks = [
        (scenario_name, scenario_config.to_dict(), seed)
        for scenario_name, scenario_config in scenario_configs.items()
        for seed in protocol.seeds
        if (scenario_name, seed) not in completed
    ]

    def record(row: dict[str, float | int | str]) -> None:
        rows.append(row)
        rows.sort(
            key=lambda item: (
                SCENARIO_NAMES.index(str(item["scenario"])),
                int(item["seed"]),
            )
        )
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, identity, rows)

    if workers == 1:
        for task in tasks:
            record(_run_metric_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_metric_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                record(future.result())

    expected_tasks = len(SCENARIO_NAMES) * len(protocol.seeds)
    if len(rows) != expected_tasks or len(completed | {
        (str(row["scenario"]), int(row["seed"])) for row in rows
    }) != expected_tasks:
        raise RuntimeError("mechanism task matrix is incomplete or duplicated")
    summary = _summarize(rows)
    gate = evaluate_gate(
        summary,
        protocol.criteria,
        rows,
        scenario_configs,
    )
    return rows, summary, gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent simulation processes (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an output directory containing a matching checkpoint",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="required matching SHA-256 manifest for a formal protocol",
    )
    parser.add_argument(
        "--unseal-formal",
        action="store_true",
        help="explicitly authorize execution of reserved formal seeds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError(args.output)
    config = load_stage1_config(args.config)
    protocol = MechanismProtocol.from_json(args.protocol)
    freeze_verification: dict[str, object] | None = None
    if protocol.stage == "formal":
        if not args.unseal_formal:
            raise PermissionError(
                "formal protocol requires explicit --unseal-formal"
            )
        if args.freeze_manifest is None:
            raise ValueError("formal protocol requires --freeze-manifest")
        freeze_verification = verify_freeze_manifest(
            args.freeze_manifest,
            project_root=Path.cwd(),
            formal_config=args.config,
            formal_protocol=args.protocol,
        )
    elif args.unseal_formal or args.freeze_manifest is not None:
        raise ValueError(
            "formal unseal options cannot be used with a development protocol"
        )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoint_path = args.output / "checkpoint.json"
    rows, summary, gate = run_protocol(
        config,
        protocol,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        workers=args.workers,
    )
    report = {
        "schema_version": "1.0.0",
        "protocol": {
            "version": protocol.version,
            "stage": protocol.stage,
            "purpose": protocol.purpose,
            "seeds": list(protocol.seeds),
            "excluded_development_seeds": list(
                protocol.excluded_development_seeds
            ),
            "minimum_burn_in_days": protocol.minimum_burn_in_days,
            "minimum_evaluation_days": (
                protocol.minimum_evaluation_days
            ),
            "criteria": dict(protocol.criteria),
        },
        "config": config.to_dict(),
        "code_revision": resolve_code_revision(),
        "freeze_verification": freeze_verification,
        "summary": summary,
        "gate": gate,
    }
    (args.output / "mechanism_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / "per_seed_metrics.csv", rows)
    _write_gate_evidence_csv(
        args.output / "gate_evidence.csv", gate.get("evidence", {})
    )
    _write_figure(args.output / "mechanism_attribution.png", rows)
    _write_markdown(
        args.output / "mechanism_report.md",
        protocol=protocol,
        summary=summary,
        gate=gate,
    )
    final_pass = bool(
        protocol.stage == "formal"
        and freeze_verification is not None
        and freeze_verification.get("verified")
        and gate["passed"]
    )
    final_decision = {
        "schema_version": "1.0.0",
        "decision": (
            "PASS_STAGE1_FORMAL_ACCEPTANCE"
            if final_pass
            else (
                "FAIL_STAGE1_FORMAL_ACCEPTANCE"
                if protocol.stage == "formal"
                else "DEVELOPMENT_EVIDENCE_ONLY"
            )
        ),
        "protocol": protocol.version,
        "protocol_stage": protocol.stage,
        "stage1_complete": final_pass,
        "may_enter_stage2": final_pass,
        "passed_checks": gate["passed_checks"],
        "total_checks": gate["total_checks"],
        "formal_seeds": (
            len(protocol.seeds) if protocol.stage == "formal" else 0
        ),
        "scenarios_per_seed": len(SCENARIO_NAMES),
        "simulations": len(rows),
        "freeze_verification": freeze_verification,
        "formal_report": str(args.output / "mechanism_report.json"),
    }
    (args.output / "final_decision.json").write_text(
        json.dumps(
            final_decision, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": gate["passed"],
                "passed_checks": gate["passed_checks"],
                "total_checks": gate["total_checks"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
