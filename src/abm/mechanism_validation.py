"""Cross-seed mechanism attribution for the Stage 1 artificial market."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import Stage1Config, load_stage1_config
from .harness import SimulationResult, run_stage1
from .runner import resolve_code_revision

FloatArray = NDArray[np.float64]

METRIC_NAMES = (
    "mean_return",
    "daily_volatility",
    "return_acf_1",
    "absolute_return_acf_1",
    "excess_kurtosis",
    "volume_absolute_return_correlation",
    "mean_absolute_log_price_gap",
    "maximum_drawdown",
    "bubble_day_fraction",
    "crash_day_fraction",
    "strategy_turnover",
    "final_wealth_gini",
    "price_cap_hits",
    "maximum_cash_relative_error",
    "maximum_share_relative_error",
)


@dataclass(frozen=True, slots=True)
class MechanismProtocol:
    version: str
    purpose: str
    seeds: tuple[int, ...]
    excluded_development_seeds: tuple[int, ...]
    criteria: Mapping[str, float]

    @classmethod
    def from_json(cls, path: str | Path) -> "MechanismProtocol":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        return cls(
            version=str(payload["version"]),
            purpose=str(payload["purpose"]),
            seeds=seeds,
            excluded_development_seeds=excluded,
            criteria={
                str(name): float(value)
                for name, value in payload["criteria"].items()
            },
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


def _lag_one_autocorrelation(values: FloatArray) -> float:
    if values.size < 3:
        return 0.0
    return _correlation(values[:-1], values[1:])


def _gini(values: FloatArray) -> float:
    if values.size == 0 or np.any(values < 0):
        raise ValueError("gini values must be nonempty and nonnegative")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        2.0 * np.dot(ranks, ordered) / (ordered.size * total)
        - (ordered.size + 1.0) / ordered.size
    )


def result_metrics(result: SimulationResult) -> dict[str, float]:
    returns = np.diff(np.log(result.prices))
    volatility = float(np.std(returns, ddof=0))
    if volatility > 0:
        standardized = (returns - float(returns.mean())) / volatility
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        excess_kurtosis = 0.0
    running_peak = np.maximum.accumulate(result.prices)
    drawdowns = result.prices / running_peak - 1.0
    log_gap = np.log(result.prices / result.fundamentals)
    initial_counts = result.strategy_counts[0].astype(np.float64)
    final_counts = result.strategy_counts[-1].astype(np.float64)
    final_wealth = (
        result.final_agent_cash
        + result.final_agent_positions * result.prices[-1]
    )
    initial_cash = float(result.total_cash[0])
    initial_shares = float(result.total_shares[0])
    return {
        "mean_return": float(np.mean(returns)),
        "daily_volatility": volatility,
        "return_acf_1": _lag_one_autocorrelation(returns),
        "absolute_return_acf_1": _lag_one_autocorrelation(
            np.abs(returns)
        ),
        "excess_kurtosis": excess_kurtosis,
        "volume_absolute_return_correlation": _correlation(
            result.volumes,
            np.abs(returns),
        ),
        "mean_absolute_log_price_gap": float(np.mean(np.abs(log_gap))),
        "maximum_drawdown": float(-np.min(drawdowns)),
        "bubble_day_fraction": float(np.mean(log_gap > 0.10)),
        "crash_day_fraction": float(np.mean(returns < -0.05)),
        "strategy_turnover": float(
            np.abs(final_counts - initial_counts).sum() / 2.0
        ),
        "final_wealth_gini": _gini(final_wealth),
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
    return {
        "full": config,
        "value_only": replace(
            config,
            strategy_shares={"value": 1.0, "trend": 0.0, "noise": 0.0},
            learning_enabled=False,
        ),
        "trend_only": replace(
            config,
            strategy_shares={"value": 0.0, "trend": 1.0, "noise": 0.0},
            learning_enabled=False,
            # Without any value anchor a pure trend market is structurally
            # explosive. Use a controlled one-quarter order capacity so the
            # direction of persistence remains measurable for 250 days.
            max_order_fraction=min(config.max_order_fraction, 0.01),
        ),
        "noise_only": replace(
            config,
            strategy_shares={"value": 0.0, "trend": 0.0, "noise": 1.0},
            learning_enabled=False,
        ),
        "no_logit": replace(config, learning_enabled=False),
        "independent_signals": replace(
            config,
            common_signal_correlation=0.0,
        ),
        "fixed_participation": replace(
            config,
            activity_volatility_sensitivity=0.0,
        ),
        "liquidity_stress": replace(
            config,
            liquidity_volatility_sensitivity=1.0,
            minimum_liquidity_fraction=0.6,
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
) -> dict[str, object]:
    full = summary["full"]

    def median(scenario: str, metric: str) -> float:
        return float(summary[scenario][metric]["median"])

    checks = {
        "full_runs_without_price_cap": (
            float(full["price_cap_hits"]["maximum"]) == 0.0
        ),
        "cash_conservation": all(
            float(metrics["maximum_cash_relative_error"]["maximum"])
            <= criteria["maximum_conservation_relative_error"]
            for metrics in summary.values()
        ),
        "share_conservation": all(
            float(metrics["maximum_share_relative_error"]["maximum"])
            <= criteria["maximum_conservation_relative_error"]
            for metrics in summary.values()
        ),
        "full_volatility_in_daily_range": (
            criteria["minimum_daily_volatility"]
            <= float(full["daily_volatility"]["median"])
            <= criteria["maximum_daily_volatility"]
        ),
        "full_return_autocorrelation_near_zero": (
            abs(float(full["return_acf_1"]["median"]))
            <= criteria["maximum_absolute_return_acf_1"]
        ),
        "full_volatility_clustering_positive": (
            float(full["absolute_return_acf_1"]["median"])
            >= criteria["minimum_absolute_return_acf_1"]
        ),
        "full_heavier_than_gaussian": (
            float(full["excess_kurtosis"]["median"])
            >= criteria["minimum_excess_kurtosis"]
        ),
        "full_volume_volatility_relation_positive": (
            float(full["volume_absolute_return_correlation"]["median"])
            >= criteria["minimum_volume_absolute_return_correlation"]
        ),
        "full_price_discovery_bounded": (
            float(full["mean_absolute_log_price_gap"]["median"])
            <= criteria["maximum_mean_absolute_log_price_gap"]
        ),
        "value_strategy_anchors_price": (
            median("value_only", "mean_absolute_log_price_gap")
            < median("noise_only", "mean_absolute_log_price_gap")
        ),
        "trend_strategy_creates_persistence": (
            median("trend_only", "return_acf_1")
            > median("noise_only", "return_acf_1")
        ),
        "logit_changes_strategy_ecology": (
            median("full", "strategy_turnover")
            > criteria["minimum_full_strategy_turnover"]
            and median("no_logit", "strategy_turnover") == 0.0
        ),
        "common_beliefs_increase_tail_weight": (
            median("full", "excess_kurtosis")
            > median("independent_signals", "excess_kurtosis")
        ),
        "adaptive_participation_reduces_return_predictability": (
            abs(median("full", "return_acf_1"))
            < abs(median("fixed_participation", "return_acf_1"))
        ),
        "liquidity_stress_increases_clustering": (
            median("liquidity_stress", "absolute_return_acf_1")
            > median("full", "absolute_return_acf_1")
        ),
        "garch_control_is_labelled_exogenous": True,
    }
    return {
        "checks": checks,
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


def _write_figure(
    path: Path,
    rows: Sequence[Mapping[str, float | int | str]],
) -> None:
    scenarios = tuple(dict.fromkeys(str(row["scenario"]) for row in rows))
    displayed = (
        "full",
        "value_only",
        "trend_only",
        "noise_only",
        "independent_signals",
        "fixed_participation",
        "liquidity_stress",
        "garch_t_control",
    )
    displayed = tuple(name for name in displayed if name in scenarios)
    panels = (
        ("daily_volatility", "Daily volatility"),
        ("return_acf_1", "Return ACF(1)"),
        ("absolute_return_acf_1", "|Return| ACF(1)"),
        ("excess_kurtosis", "Excess kurtosis"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
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


def run_protocol(
    config: Stage1Config,
    protocol: MechanismProtocol,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, object],
]:
    rows: list[dict[str, float | int | str]] = []
    for scenario_name, scenario_config in build_scenarios(config).items():
        for seed in protocol.seeds:
            result = run_stage1(replace(scenario_config, seed=seed))
            rows.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    **result_metrics(result),
                }
            )
    summary = _summarize(rows)
    gate = evaluate_gate(summary, protocol.criteria)
    return rows, summary, gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_stage1_config(args.config)
    protocol = MechanismProtocol.from_json(args.protocol)
    rows, summary, gate = run_protocol(config, protocol)
    args.output.mkdir(parents=True)
    report = {
        "schema_version": "1.0.0",
        "protocol": {
            "version": protocol.version,
            "purpose": protocol.purpose,
            "seeds": list(protocol.seeds),
            "excluded_development_seeds": list(
                protocol.excluded_development_seeds
            ),
            "criteria": dict(protocol.criteria),
        },
        "config": config.to_dict(),
        "code_revision": resolve_code_revision(),
        "summary": summary,
        "gate": gate,
    }
    (args.output / "mechanism_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / "per_seed_metrics.csv", rows)
    _write_figure(args.output / "mechanism_attribution.png", rows)
    _write_markdown(
        args.output / "mechanism_report.md",
        protocol=protocol,
        summary=summary,
        gate=gate,
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
