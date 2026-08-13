"""Tooling tests: calibration, French 49 benchmark, manifests, scenarios."""
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from abm.calibration import (
    CalibrationProtocol,
    build_parser,
    compare_groups,
    simulation_group,
)
from abm.french49 import (
    French49Dataset,
    build_benchmark_report,
    deterministic_split,
    parse_french49_zip,
)
from abm.manifest import build_run_manifest, sha256_file
from abm.scenario import load_synthetic_scenario
from abm.schemas import RunManifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = PROJECT_ROOT / "data" / "synthetic" / "stage0_10_day.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "stage0_10_day.json"


# --- Calibration ----------------------------------------------------------


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


# --- French 49 benchmark --------------------------------------------------


def test_french49_parser_reads_value_weighted_daily_table(tmp_path) -> None:
    industries = [f"Ind{index:02d}" for index in range(49)]
    rows = [
        "Synthetic French fixture",
        "Average Value Weighted Returns -- Daily",
        "," + ",".join(industries),
    ]
    start = date(2020, 1, 1)
    for offset in range(25):
        current = start + timedelta(days=offset)
        values = [
            "-99.99" if offset == 0 and index == 0 else "1.00"
            for index in range(49)
        ]
        rows.append(current.strftime("%Y%m%d") + "," + ",".join(values))
    rows.extend(["", "Average Equal Weighted Returns -- Daily"])
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("49_Industry_Portfolios_Daily.csv", "\n".join(rows))

    dataset = parse_french49_zip(archive_path)

    assert dataset.returns.shape == (25, 49)
    assert np.isnan(dataset.returns[0, 0])
    assert dataset.returns[1, 0] == 0.01
    assert dataset.industries == tuple(industries)


def test_deterministic_split_and_report_keep_test_metrics_sealed() -> None:
    industries = tuple(f"Ind{index:02d}" for index in range(49))
    split = deterministic_split(industries)
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.015, size=(100, 49))
    dataset = French49Dataset(
        dates=np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-04-10"),
            dtype="datetime64[D]",
        )[:100],
        industries=industries,
        returns=returns,
    )
    simulation_returns = rng.normal(0.0004, 0.015, size=100)
    prices = 100 * np.cumprod(np.concatenate(([1.0], 1 + simulation_returns)))

    report = build_benchmark_report(dataset, split, prices)

    assert len(split.train) == 29
    assert len(split.validation) == 10
    assert len(split.test) == 10
    assert report["sealed_test"]["metrics_read"] is False
    assert "per_industry" not in report["sealed_test"]
    assert len(report["training"]["per_industry"]) == 29
    assert len(report["validation"]["per_industry"]) == 10


# --- Manifests ------------------------------------------------------------


def test_manifest_is_deterministic_across_mapping_order(tmp_path) -> None:
    first_data = tmp_path / "first.json"
    second_data = tmp_path / "second.json"
    first_data.write_text('{"value": 1}\n', encoding="utf-8")
    second_data.write_text('{"value": 2}\n', encoding="utf-8")

    first = build_run_manifest(
        config={"market": {"enabled": True}, "seed": 7},
        data_files={"second": second_data, "first": first_data},
        random_seed=7,
        code_revision="test-revision",
    )
    second = build_run_manifest(
        config={"seed": 7, "market": {"enabled": True}},
        data_files={"first": first_data, "second": second_data},
        random_seed=7,
        code_revision="test-revision",
    )

    assert first == second
    assert RunManifest.from_dict(first.to_dict()) == first


def test_manifest_identity_changes_when_data_changes(tmp_path) -> None:
    data_path = tmp_path / "fixture.json"
    data_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    before = build_run_manifest(
        config={"seed": 7},
        data_files={"fixture": data_path},
        random_seed=7,
        code_revision="test-revision",
    )

    data_path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    after = build_run_manifest(
        config={"seed": 7},
        data_files={"fixture": data_path},
        random_seed=7,
        code_revision="test-revision",
    )

    assert before.run_id != after.run_id
    assert before.data_sha256 != after.data_sha256


# --- Synthetic scenarios --------------------------------------------------


def test_stage0_scenario_loads_all_contracts() -> None:
    scenario = load_synthetic_scenario(SCENARIO_PATH)

    assert len(scenario.market_states) == 10
    assert len(scenario.agent_states) == 5
    assert len(scenario.company_states) == 2
    assert len(scenario.messages) == 3
    assert len(scenario.edges) == 4
    assert len(scenario.orders) == 10
    assert len(scenario.factor_events) == 2


def test_stage0_market_dates_are_weekdays() -> None:
    scenario = load_synthetic_scenario(SCENARIO_PATH)

    assert all(state.date.weekday() < 5 for state in scenario.market_states)


def test_frozen_fixture_matches_immutable_data_manifest() -> None:
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][0]

    assert artifact["sha256"] == sha256_file(SCENARIO_PATH)
