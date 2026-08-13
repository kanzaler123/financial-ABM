"""Frozen multi-seed Stage 1 calibration and validation protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import Stage1Config, load_stage1_config
from .french49 import (
    French49Dataset,
    IndustrySplit,
    benchmark_group,
    parse_french49_zip,
    return_statistics,
)
from .harness import run_stage1

FloatArray = NDArray[np.float64]

CALIBRATION_METRICS = (
    "mean_daily_return",
    "daily_volatility",
    "return_autocorrelation_lag1",
    "absolute_return_autocorrelation_lag1",
    "excess_kurtosis",
    "quantile_01",
    "quantile_99",
)


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    version: str
    trading_days: int
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    sealed_test_seeds: tuple[int, ...]
    common_seed: int
    fundamental_common_correlation: float = 0.0

    def __post_init__(self) -> None:
        if self.trading_days < 250:
            raise ValueError("calibration paths must contain at least 250 days")
        if not self.training_seeds or not self.validation_seeds:
            raise ValueError("training and validation seeds must not be empty")
        all_seeds = (
            self.training_seeds
            + self.validation_seeds
            + self.sealed_test_seeds
        )
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("calibration seed groups must be disjoint")
        if self.common_seed in set(all_seeds):
            raise ValueError("common-factor seed must be reserved")
        if not 0.0 <= self.fundamental_common_correlation <= 1.0:
            raise ValueError(
                "fundamental_common_correlation must be in [0, 1]"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibrationProtocol":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=str(payload["version"]),
            trading_days=int(payload["trading_days"]),
            training_seeds=tuple(int(value) for value in payload["training_seeds"]),
            validation_seeds=tuple(
                int(value) for value in payload["validation_seeds"]
            ),
            sealed_test_seeds=tuple(
                int(value) for value in payload["sealed_test_seeds"]
            ),
            common_seed=int(payload["common_seed"]),
            fundamental_common_correlation=float(
                payload.get("fundamental_common_correlation", 0.0)
            ),
        )


def _metric_summary(
    per_path: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in CALIBRATION_METRICS:
        values = np.array(
            [float(statistics[metric]) for statistics in per_path.values()],
            dtype=np.float64,
        )
        summary[metric] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return summary


def simulation_group(
    returns_by_path: Mapping[str, FloatArray],
) -> dict[str, object]:
    if not returns_by_path:
        raise ValueError("simulation group must not be empty")
    lengths = {values.size for values in returns_by_path.values()}
    if len(lengths) != 1:
        raise ValueError("simulation paths must have equal lengths")
    per_path = {
        name: return_statistics(values)
        for name, values in returns_by_path.items()
    }
    matrix = np.column_stack(tuple(returns_by_path.values()))
    cross_sectional = (
        np.std(matrix, axis=1, ddof=1)
        if matrix.shape[1] > 1
        else np.zeros(matrix.shape[0], dtype=np.float64)
    )
    return {
        "paths": len(per_path),
        "observations_per_path": int(next(iter(lengths))),
        "per_path": per_path,
        "summary": _metric_summary(per_path),
        "mean_daily_cross_sectional_volatility": float(
            np.mean(cross_sectional)
        ),
    }


def compare_groups(
    empirical: Mapping[str, object],
    simulated: Mapping[str, object],
) -> dict[str, object]:
    empirical_summary = empirical["summary"]
    simulated_summary = simulated["summary"]
    metrics: dict[str, object] = {}
    all_inside = True
    for metric in CALIBRATION_METRICS:
        target = empirical_summary[metric]
        simulated_median = float(simulated_summary[metric]["median"])
        inside = float(target["q25"]) <= simulated_median <= float(target["q75"])
        width = max(float(target["q75"]) - float(target["q25"]), 1e-12)
        metrics[metric] = {
            "empirical_q25": float(target["q25"]),
            "empirical_median": float(target["median"]),
            "empirical_q75": float(target["q75"]),
            "simulation_median": simulated_median,
            "inside_empirical_iqr": inside,
            "normalized_median_distance": abs(
                simulated_median - float(target["median"])
            )
            / width,
        }
        all_inside = all_inside and inside
    empirical_cross = float(empirical["mean_daily_cross_sectional_volatility"])
    simulation_cross = float(
        simulated["mean_daily_cross_sectional_volatility"]
    )
    return {
        "metrics": metrics,
        "all_metric_medians_inside_empirical_iqr": all_inside,
        "cross_sectional_volatility": {
            "empirical": empirical_cross,
            "simulation": simulation_cross,
            "relative_error": (
                abs(simulation_cross - empirical_cross) / empirical_cross
            ),
        },
    }


def simulate_seed_group(
    config: Stage1Config,
    *,
    seeds: Sequence[int],
    trading_days: int,
    common_seed: int,
    fundamental_common_correlation: float = 0.0,
) -> dict[str, FloatArray]:
    paths: dict[str, FloatArray] = {}
    common_rng = np.random.default_rng(common_seed)
    common_normal = common_rng.standard_normal(trading_days)
    if config.fundamental_shock_df > 0:
        shared_scale = np.sqrt(
            common_rng.chisquare(
                config.fundamental_shock_df, size=trading_days
            )
            / config.fundamental_shock_df
        )
        variance_normalizer = np.sqrt(
            config.fundamental_shock_df
            / (config.fundamental_shock_df - 2.0)
        )
    else:
        shared_scale = np.ones(trading_days, dtype=np.float64)
        variance_normalizer = 1.0
    if not 0.0 <= fundamental_common_correlation <= 1.0:
        raise ValueError(
            "fundamental_common_correlation must be in [0, 1]"
        )
    correlation = fundamental_common_correlation
    for seed in seeds:
        idiosyncratic_normal = np.random.default_rng(int(seed)).standard_normal(
            trading_days
        )
        correlated_normal = (
            np.sqrt(correlation) * common_normal
            + np.sqrt(1.0 - correlation) * idiosyncratic_normal
        )
        innovations = (
            correlated_normal / shared_scale / variance_normalizer
        )
        result = run_stage1(
            replace(config, seed=int(seed), trading_days=trading_days),
            fundamental_innovations=innovations,
        )
        paths[str(seed)] = result.prices[1:] / result.prices[:-1] - 1.0
    return paths


def build_calibration_report(
    dataset: French49Dataset,
    split: IndustrySplit,
    config: Stage1Config,
    protocol: CalibrationProtocol,
) -> dict[str, object]:
    training_empirical = benchmark_group(dataset, split.train)
    validation_empirical = benchmark_group(dataset, split.validation)
    training_simulation = simulation_group(
        simulate_seed_group(
            config,
            seeds=protocol.training_seeds,
            trading_days=protocol.trading_days,
            common_seed=protocol.common_seed,
            fundamental_common_correlation=(
                protocol.fundamental_common_correlation
            ),
        )
    )
    validation_simulation = simulation_group(
        simulate_seed_group(
            config,
            seeds=protocol.validation_seeds,
            trading_days=protocol.trading_days,
            common_seed=protocol.common_seed,
            fundamental_common_correlation=(
                protocol.fundamental_common_correlation
            ),
        )
    )
    return {
        "schema_version": "0.2.0",
        "protocol_version": protocol.version,
        "period_start": str(dataset.dates[0]),
        "period_end": str(dataset.dates[-1]),
        "trading_days_per_simulation": protocol.trading_days,
        "training": {
            "empirical": training_empirical,
            "simulation": training_simulation,
            "comparison": compare_groups(
                training_empirical, training_simulation
            ),
        },
        "validation": {
            "empirical": validation_empirical,
            "simulation": validation_simulation,
            "comparison": compare_groups(
                validation_empirical, validation_simulation
            ),
        },
        "sealed_test": {
            "industries": list(split.test),
            "reserved_seeds": list(protocol.sealed_test_seeds),
            "common_seed": protocol.common_seed,
            "metrics_read": False,
            "simulations_run": False,
        },
    }


def build_sealed_test_report(
    dataset: French49Dataset,
    split: IndustrySplit,
    config: Stage1Config,
    protocol: CalibrationProtocol,
) -> dict[str, object]:
    empirical = benchmark_group(dataset, split.test)
    simulation = simulation_group(
        simulate_seed_group(
            config,
            seeds=protocol.sealed_test_seeds,
            trading_days=protocol.trading_days,
            common_seed=protocol.common_seed,
            fundamental_common_correlation=(
                protocol.fundamental_common_correlation
            ),
        )
    )
    return {
        "schema_version": "0.2.0",
        "protocol_version": protocol.version,
        "period_start": str(dataset.dates[0]),
        "period_end": str(dataset.dates[-1]),
        "trading_days_per_simulation": protocol.trading_days,
        "sealed_test_unsealed": True,
        "test": {
            "industries": list(split.test),
            "seeds": list(protocol.sealed_test_seeds),
            "empirical": empirical,
            "simulation": simulation,
            "comparison": compare_groups(empirical, simulation),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="run the one-time frozen test group instead of training/validation",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = parse_french49_zip(args.archive).between(args.start, args.end)
    split = IndustrySplit.from_json(args.split)
    config = load_stage1_config(args.config)
    protocol = CalibrationProtocol.from_json(args.protocol)
    if args.unseal_test:
        report = build_sealed_test_report(
            dataset, split, config, protocol
        )
    else:
        report = build_calibration_report(
            dataset, split, config, protocol
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
