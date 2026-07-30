from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.calibration import (
    CalibrationProtocol,
    build_parser,
    compare_groups,
    simulation_group,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_calibration_protocol_has_disjoint_29_10_10_seeds() -> None:
    protocol = CalibrationProtocol.from_json(
        PROJECT_ROOT / "configs" / "stage1_calibration_v1.json"
    )

    assert len(protocol.training_seeds) == 29
    assert len(protocol.validation_seeds) == 10
    assert len(protocol.sealed_test_seeds) == 10
    assert len(
        set(
            protocol.training_seeds
            + protocol.validation_seeds
            + protocol.sealed_test_seeds
        )
    ) == 49
    assert protocol.common_seed not in protocol.training_seeds


def test_simulation_group_and_iqr_comparison_are_deterministic() -> None:
    paths = {
        "a": np.linspace(-0.02, 0.02, 100),
        "b": np.linspace(-0.018, 0.022, 100),
    }

    first = simulation_group(paths)
    second = simulation_group(paths)
    empirical = {
        "summary": {
            metric: {
                "q25": values["q25"] - 1.0,
                "median": values["median"],
                "q75": values["q75"] + 1.0,
            }
            for metric, values in first["summary"].items()
        },
        "mean_daily_cross_sectional_volatility": first[
            "mean_daily_cross_sectional_volatility"
        ],
    }

    assert first == second
    assert compare_groups(
        empirical, first
    )["all_metric_medians_inside_empirical_iqr"]


def test_protocol_rejects_overlapping_seeds() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        CalibrationProtocol(
            version="bad",
            trading_days=250,
            training_seeds=(1,),
            validation_seeds=(1,),
            sealed_test_seeds=(2,),
            common_seed=3,
        )


def test_unseal_test_requires_explicit_cli_flag() -> None:
    parser = build_parser()
    base = [
        "--config",
        "config.json",
        "--protocol",
        "protocol.json",
        "--archive",
        "data.zip",
        "--split",
        "split.json",
        "--output",
        "report.json",
    ]

    assert parser.parse_args(base).unseal_test is False
    assert parser.parse_args([*base, "--unseal-test"]).unseal_test is True
