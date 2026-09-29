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


def test_stage1_diagnostics_pair_scenarios_without_changing_simulations(tmp_path) -> None:
    from dataclasses import replace

    from abm.config import load_stage1_config
    from abm.harness import run_stage1
    from abm.mechanism_validation import MechanismProtocol
    from abm.stage1_diagnostics import diagnostic_scenarios, run_diagnostics

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5,
        liquidity_scale=500.0, announcements=(),
    )
    protocol = MechanismProtocol("test", "diagnostics", (71,), (), {})
    output = tmp_path / "diagnostics"
    report = run_diagnostics(config, protocol, output, workers=2)

    scenarios = diagnostic_scenarios(config)
    assert list(scenarios) == [
        "legacy", "reference_notional", "depth_notional", "candidate",
        "legacy_fixed_liquidity", "candidate_fixed_liquidity",
    ]
    assert len(report["runs"]) == 6
    assert {run["seed"] for run in report["runs"]} == {71}
    assert report["decision"] == "DEVELOPMENT_DIAGNOSTICS_ONLY"
    assert report["stage1_complete"] is False
    assert report["may_enter_stage2"] is False
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "COMPLETE"
    assert len(checkpoint["completed_runs"]) == 6
    assert checkpoint["unfinished_runs"] == []
    for run in report["runs"]:
        plain = run_stage1(replace(scenarios[run["scenario"]], seed=71))
        assert run["fingerprint"] == plain.fingerprint()
        daily = np.genfromtxt(output / run["daily_file"], delimiter=",", names=True)
        assert len(daily) == 35
        np.testing.assert_array_equal(daily["transient_impact"], plain.transient_impacts)
        reconstructed = (
            daily["public_news_impact"] + daily["permanent_impact"]
            + daily["transient_impact_change"] + daily["price_cap_adjustment"]
        )
        np.testing.assert_allclose(reconstructed, daily["total_log_return"], atol=1e-12)
        evaluation = daily[config.burn_in_days:]
        decomposition = run["variance_decomposition"]
        covariance = np.cov(
            evaluation["public_news_impact"], evaluation["demand_log_return"], ddof=0
        )[0, 1]
        assert decomposition["ddof"] == 0
        assert decomposition["news_demand_covariance"] == pytest.approx(covariance)
        assert decomposition["total_variance"] == pytest.approx(
            decomposition["news_variance"] + decomposition["demand_variance"]
            + 2 * covariance,
            abs=1e-12,
        )
        assert abs(decomposition["reconstruction_error"]) < 1e-12
        assert len(run["cap_events"]) == int(run["metrics"]["price_cap_hits"])


def test_stage1_diagnostics_reject_formal_protocol_and_overwrite(tmp_path) -> None:
    from dataclasses import replace

    from abm.config import load_stage1_config
    from abm.mechanism_validation import MechanismProtocol
    from abm.stage1_diagnostics import run_diagnostics

    config = load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json")
    protocol = MechanismProtocol("test", "diagnostics", (71,), (), {})
    output = tmp_path / "diagnostics"
    with pytest.raises(ValueError, match="development"):
        run_diagnostics(config, replace(protocol, stage="formal"), output)
    assert not output.exists()
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("existing result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_diagnostics(config, protocol, output)
    assert marker.read_text(encoding="utf-8") == "existing result"


def test_stage1_diagnostics_stop_on_price_decomposition_error(tmp_path, monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from abm.config import load_stage1_config
    from abm.harness import MarketHarness
    from abm.mechanism_validation import MechanismProtocol
    from abm import stage1_diagnostics

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=10, burn_in_days=0,
    )

    class InvalidPriceHarness(MarketHarness):
        def step(self, day_index):
            audit = super().step(day_index)
            return replace(audit, public_news_impact=audit.public_news_impact + 0.01)

    monkeypatch.setattr(stage1_diagnostics, "MarketHarness", InvalidPriceHarness)
    (tmp_path / "daily").mkdir()
    with pytest.raises(RuntimeError, match="price decomposition failed on day 1"):
        stage1_diagnostics._run_diagnostic("invalid", config, tmp_path)
    monkeypatch.setattr(stage1_diagnostics, "ProcessPoolExecutor", ThreadPoolExecutor)
    output = tmp_path / "failed"
    protocol = MechanismProtocol("test", "diagnostics", (71,), (), {})
    with pytest.raises(RuntimeError, match="price decomposition failed on day 1"):
        stage1_diagnostics.run_diagnostics(config, protocol, output, workers=1)
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "ENGINEERING_ERROR"
    assert checkpoint["completed_runs"] == []
    assert len(checkpoint["unfinished_runs"]) == checkpoint["expected_runs"] == 6
    assert checkpoint["errors"][0]["type"] == "RuntimeError"


def test_price_gap_reconstruction_keeps_initial_and_burn_in_levels_and_caps() -> None:
    from dataclasses import replace

    from abm.config import load_stage1_config
    from abm.stage1_diagnostics import reconstruct_price_gap

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        trading_days=4, burn_in_days=2, initial_price=100.0,
        initial_fundamental=90.0, announcements=(),
    )
    news = np.array([0.05, -0.03, 0.02, 0.01])
    permanent = np.array([-0.02, 0.04, -0.03, 0.01])
    transient = np.array([0.01, -0.02, 0.015, -0.005])
    cap = np.array([0.0, 0.025, 0.0, -0.01])
    transient_change = np.diff(np.r_[0.0, transient])
    raw = permanent + transient_change
    total = config.public_news_price_pass_through * news + raw + cap
    prices = np.r_[config.initial_price, config.initial_price * np.exp(np.cumsum(total))]
    fundamentals = config.initial_fundamental * np.exp(np.cumsum(news))
    rows = [
        {
            "day": day + 1, "price_before": prices[day],
            "price_after": prices[day + 1],
            "public_news_impact": config.public_news_price_pass_through * news[day],
            "permanent_impact": permanent[day], "transient_impact": transient[day],
            "transient_impact_change": transient_change[day],
            "price_cap_adjustment": cap[day], "price_cap_hit": cap[day] != 0,
            "raw_demand_log_return": raw[day], "demand_log_return": raw[day] + cap[day],
            "total_log_return": total[day],
        }
        for day in range(4)
    ]

    report = reconstruct_price_gap(rows, config)
    expected = np.log(prices[1:] / fundamentals)
    np.testing.assert_allclose(
        [row["signed_log_price_gap"] for row in report["daily"]], expected,
        atol=1e-13,
    )
    assert report["summary"]["g_burn_in"] == pytest.approx(expected[1])
    assert report["summary"]["mean_absolute_log_price_gap"] == pytest.approx(
        np.mean(np.abs(expected[2:]))
    )
    assert report["daily"][-1]["price_cap_adjustment_cumulative"] == pytest.approx(0.015)
    assert max(abs(row["gap_reconstruction_error"]) for row in report["daily"]) < 1e-12
    with pytest.raises(ValueError):
        reconstruct_price_gap(rows, replace(config, public_news_price_pass_through=0.0))
    with pytest.raises((ValueError, RuntimeError)):
        reconstruct_price_gap(rows[:-1], config)
    damaged = [dict(row) for row in rows]
    damaged[2]["price_before"] *= 1.01
    with pytest.raises((ValueError, RuntimeError)):
        reconstruct_price_gap(damaged, config)


def test_price_discovery_probes_preserve_real_policy_population_and_rng() -> None:
    from copy import deepcopy
    from dataclasses import fields, replace

    from abm.config import load_stage1_config
    from abm.harness import MarketHarness
    from abm.policies import MarketObservation
    from abm.stage1_diagnostics import _PriceDiscoveryPolicy

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5, announcements=(),
    )
    plain = MarketHarness(config)
    plain.population.desired_positions += 2.0
    plain.population.base_activity_rates[:] = 1.0 - 1e-12
    observed = deepcopy(plain)
    observed.policy = _PriceDiscoveryPolicy(observed.policy)
    observation = MarketObservation(
        price=100.0, fundamental_value=100.0, price_history=np.array([100.0]),
        public_news_return=0.0, realized_volatility=config.fundamental_volatility,
        realized_volatility_reference=config.fundamental_volatility,
    )
    expected = plain.policy.act(observation, plain.population, plain.noise_rng)
    actual = observed.policy.act(observation, observed.population, observed.noise_rng)

    np.testing.assert_array_equal(actual, expected)
    for field in fields(plain.population):
        np.testing.assert_equal(
            getattr(observed.population, field.name), getattr(plain.population, field.name)
        )
    assert observed.noise_rng.bit_generator.state == plain.noise_rng.bit_generator.state
    assert observed.policy.policy.activity_state == plain.policy.activity_state
    np.testing.assert_array_equal(
        observed.policy.policy.common_signal_state, plain.policy.common_signal_state
    )
    assert observed.policy.policy.diagnostics == plain.policy.diagnostics
    assert len(observed.policy.probes) == 1
    assert observed.policy.probes[0]["baseline_orders_exact"]
    assert observed.policy.probes[0]["fixed_state_local_only"]
    interventions = observed.policy.probes[0]["interventions"]
    baseline_value = interventions["baseline"]["value"]["net_notional"]
    assert baseline_value > 0.0
    assert interventions["cancel_in_band_target_backlog"]["value"]["net_notional"] == 0.0
    assert interventions["cancel_in_band_target_backlog"]["value"]["net_notional_delta_from_baseline"] == -baseline_value
    assert interventions["value_sensitivity_zero"]["value"]["net_notional_delta_from_baseline"] == 0.0


def test_price_discovery_replays_match_sources_and_preserve_missing_runs(tmp_path, monkeypatch) -> None:
    import csv
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict, replace

    from abm import stage1_diagnostics
    from abm.config import load_stage1_config
    from abm.harness import run_stage1
    from abm.mechanism_validation import MechanismProtocol, build_scenarios, result_metrics

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5,
        liquidity_scale=500.0, announcements=(), seed=71,
    )
    protocol = MechanismProtocol("test", "price discovery", (71,), (), {})
    source = tmp_path / "source"
    source.mkdir()
    stage1_diagnostics.run_diagnostics(config, protocol, source / "diagnostics", workers=1)
    development = source / "development"
    development.mkdir()
    (development / "mechanism_report.json").write_text(json.dumps({
        "config": config.to_dict(), "protocol": asdict(protocol),
    }), encoding="utf-8")
    reference = {}
    metric_rows = []
    for name in ("full", "fixed_liquidity", "no_agent_information"):
        result = run_stage1(build_scenarios(config)[name])
        reference[name] = result
        metric_rows.append({"scenario": name, "seed": 71, **result_metrics(result, 5)})
    with (development / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    output = tmp_path / "discovery"
    report = stage1_diagnostics.run_price_discovery(
        config, protocol, output, source=source, workers=1,
    )
    assert len(report["runs"]) == 3
    for run in report["runs"]:
        plain = reference[run["scenario"]]
        assert run["fingerprint"] == plain.fingerprint()
        assert run["metrics"] == result_metrics(plain, 5)
        daily = np.genfromtxt(output / run["daily_file"], delimiter=",", names=True)
        np.testing.assert_allclose(
            daily["signed_log_price_gap"], np.log(plain.prices[1:] / plain.fundamentals[1:]),
            atol=1e-13,
        )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "COMPLETE"
    assert len(checkpoint["completed_runs"]) == 3
    assert checkpoint["unfinished_runs"] == []
    with pytest.raises(FileExistsError):
        stage1_diagnostics.run_price_discovery(config, protocol, output, source=source)
    with pytest.raises(ValueError):
        stage1_diagnostics.run_price_discovery(
            replace(config, value_sensitivity=config.value_sensitivity + 0.1),
            protocol, tmp_path / "incompatible", source=source,
        )

    monkeypatch.setattr(stage1_diagnostics, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(stage1_diagnostics, "as_completed", lambda futures: iter(futures))
    source_result = source / "diagnostics" / "candidate_71.json"
    original_result = source_result.read_text(encoding="utf-8")
    poisoned = json.loads(original_result)
    poisoned["fingerprint"] = "not-the-original-fingerprint"
    source_result.write_text(json.dumps(poisoned), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        stage1_diagnostics.run_price_discovery(
            config, protocol, tmp_path / "fingerprint_mismatch", source=source, workers=1,
        )
    source_result.write_text(original_result, encoding="utf-8")
    source_metrics = development / "per_seed_metrics.csv"
    original_metrics = source_metrics.read_text(encoding="utf-8")
    metric_rows[-1]["mean_return"] += 1.0
    with source_metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with pytest.raises(RuntimeError, match="metrics"):
        stage1_diagnostics.run_price_discovery(
            config, protocol, tmp_path / "metrics_mismatch", source=source, workers=1,
        )
    source_metrics.write_text(original_metrics, encoding="utf-8")
    original_replay = stage1_diagnostics._run_price_discovery

    def fail_replay(scenario, *args, **kwargs):
        if scenario == "fixed_liquidity":
            raise RuntimeError("missing replay fixture")
        return original_replay(scenario, *args, **kwargs)

    monkeypatch.setattr(stage1_diagnostics, "_run_price_discovery", fail_replay)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="missing replay fixture"):
        stage1_diagnostics.run_price_discovery(
            config, protocol, failed, source=source, workers=1,
        )
    checkpoint = json.loads((failed / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "ENGINEERING_ERROR"
    assert checkpoint["completed_runs"] == [{"scenario": "full", "seed": 71}]
    assert {(item["scenario"], item["seed"]) for item in checkpoint["unfinished_runs"]} == {
        ("fixed_liquidity", 71), ("no_agent_information", 71),
    }


@pytest.mark.parametrize("constraint", ["none", "valuation_direction"])
@pytest.mark.parametrize("inventory_control", ["none", "budget_priority"])
def test_value_order_instrumentation_preserves_market_rng_and_snapshots(
    constraint, inventory_control,
) -> None:
    from copy import deepcopy
    from dataclasses import fields, replace

    from abm.config import load_stage1_config
    from abm.harness import MarketHarness

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5,
        liquidity_scale=500.0, announcements=(), value_order_constraint=constraint,
        value_inventory_control=inventory_control,
    )
    plain = MarketHarness(config)
    observed = deepcopy(plain)
    observed.policy.instrument_orders = True
    plain.step(0)
    observed.step(0)
    snapshot = observed.policy.order_diagnostics
    assert snapshot is not None
    saved = {}
    for field in fields(snapshot):
        values = getattr(snapshot, field.name)
        assert values.shape == (config.population_size,)
        assert not values.flags.writeable
        saved[field.name] = values.copy()
        with pytest.raises(ValueError):
            values[0] = 0.0

    assert observed.run().fingerprint() == plain.run().fingerprint()
    assert observed.noise_rng.bit_generator.state == plain.noise_rng.bit_generator.state
    assert observed.learning_rng.bit_generator.state == plain.learning_rng.bit_generator.state
    assert plain.policy.instrument_orders is False
    assert plain.policy.order_diagnostics is None
    for field in fields(plain.population):
        np.testing.assert_equal(
            getattr(observed.population, field.name), getattr(plain.population, field.name)
        )
    for name, values in saved.items():
        np.testing.assert_array_equal(getattr(snapshot, name), values)


@pytest.mark.parametrize(
    "stage,orders,forced,price,expected",
    [
        ("before_constraint", [-3, 0, 4, 0, -9, 5, -20, 30], None, 10,
         (1, 3, 2, 9, 1, 1)),
        ("after_constraint", [0, 0, 0, 0, -9, 0, -20, 30], None, 10,
         (0, 0, 0, 0, 2, 3)),
        ("after_risk", [0, 0, 0, 0, -9, 5, -20, 30], [0, 0, 0, 0, 0, 5, 0, 0], 10,
         (0, 0, 0, 0, 2, 3)),
        ("filled", [0, 0, 0, 0, -7, 4, -15, 25], [0, 0, 0, 0, 0, 4, 0, 0], 12,
         (0, 0, 0, 0, 2, 3)),
    ],
)
def test_value_direction_statistics_separate_forced_covers_and_order_stages(
    stage, orders, forced, price, expected,
) -> None:
    from abm.population import NOISE_STRATEGY, TREND_STRATEGY, VALUE_STRATEGY
    from abm.stage1_diagnostics import _value_direction_fields

    values = _value_direction_fields(
        np.array(orders, dtype=float),
        np.array([1, 1, -1, -1, 0, -1, 1, -1], dtype=float),
        np.array([VALUE_STRATEGY] * 6 + [TREND_STRATEGY, NOISE_STRATEGY]),
        price, stage,
        forced_cover_orders=None if forced is None else np.array(forced, dtype=float),
    )
    sell_count, sell_shares, buy_count, buy_shares, positive_zero, negative_zero = expected
    prefix = f"value_{stage}_"
    for direction, count, shares in (
        ("positive_signal_voluntary_sell", sell_count, sell_shares),
        ("negative_signal_voluntary_buy", buy_count, buy_shares),
    ):
        assert values[f"{prefix}{direction}_count"] == count
        assert values[f"{prefix}{direction}_shares"] == shares
        assert values[f"{prefix}{direction}_notional"] == shares * price
    assert values[f"{prefix}positive_signal_zero_order_count"] == positive_zero
    assert values[f"{prefix}negative_signal_zero_order_count"] == negative_zero


def test_value_direction_replays_normalize_old_manifests_and_preserve_partial_results(
    tmp_path, monkeypatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict, replace

    from abm import stage1_diagnostics
    from abm.config import load_stage1_config
    from abm.harness import run_stage1
    from abm.mechanism_validation import MechanismProtocol, result_metrics

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5,
        liquidity_scale=500.0, announcements=(), seed=71,
        value_order_constraint="valuation_direction",
    )
    baseline = replace(config, value_order_constraint="none")
    protocol = MechanismProtocol("new", "value direction", (71,), (), {})
    original_protocol = replace(protocol, version="old", purpose="price discovery")
    reference = {
        "baseline": run_stage1(baseline), "candidate": run_stage1(config),
    }
    source = tmp_path / "price_discovery"
    source.mkdir()
    old_config = baseline.to_dict()
    del old_config["value_order_constraint"]
    (source / "manifest.json").write_text(json.dumps({
        "config": old_config, "protocol": asdict(original_protocol),
        "scenario_configs": {"full": old_config},
    }), encoding="utf-8")
    (source / "full_71.json").write_text(json.dumps({
        "scenario": "full", "seed": 71,
        "metrics": result_metrics(reference["baseline"], 5),
        "fingerprint": reference["baseline"].fingerprint(),
    }), encoding="utf-8")
    monkeypatch.setattr(stage1_diagnostics, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(stage1_diagnostics, "as_completed", lambda futures: iter(futures))
    output = tmp_path / "direction"
    report = stage1_diagnostics.run_value_direction(
        config, protocol, output, source=source, workers=1,
    )
    assert report["decision"] == "DEVELOPMENT_DIAGNOSTICS_ONLY"
    assert report["stage1_complete"] is False
    assert report["may_enter_stage2"] is False
    assert len(report["runs"]) == 2
    assert len(report["paired_comparisons"]) == 1
    for run in report["runs"]:
        plain = reference[run["scenario"]]
        assert run["fingerprint"] == plain.fingerprint()
        assert run["metrics"] == result_metrics(plain, 5)
        assert run["probes"] == []
        daily = np.genfromtxt(output / run["daily_file"], delimiter=",", names=True)
        assert len(daily) == config.trading_days
        np.testing.assert_allclose(
            daily["signed_log_price_gap"],
            np.log(plain.prices[1:] / plain.fundamentals[1:]), atol=1e-13,
        )
        evaluation = daily[config.burn_in_days:]
        for category in ("direction_counts", "direction_flows"):
            for name, total in run[category].items():
                assert total == pytest.approx(evaluation[name].sum())
        for strategy in ("value", "trend", "noise"):
            assert run["exposure"][f"{strategy}_mean_gross_notional"] == pytest.approx(
                evaluation[f"{strategy}_actual_gross_notional_before"].mean()
            )
            assert run["exposure"][f"{strategy}_final_wealth"] == pytest.approx(
                daily[f"{strategy}_wealth_after"][-1]
            )
        if run["scenario"] == "candidate":
            for stage in ("after_constraint", "after_risk", "filled"):
                for direction in ("positive_signal_voluntary_sell", "negative_signal_voluntary_buy"):
                    assert np.count_nonzero(daily[f"value_{stage}_{direction}_count"]) == 0
    gaps = report["paired_comparisons"][0]["gap_reconstruction"]
    for name, index in (("g_burn_in", config.burn_in_days), ("final_signed_log_price_gap", -1)):
        expected = {
            scenario: np.log(result.prices[index] / result.fundamentals[index])
            for scenario, result in reference.items()
        }
        assert gaps[name]["baseline"] == pytest.approx(expected["baseline"])
        assert gaps[name]["candidate"] == pytest.approx(expected["candidate"])
        assert gaps[name]["delta"] == pytest.approx(expected["candidate"] - expected["baseline"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_configs = {"baseline": baseline.to_dict(), "candidate": config.to_dict()}
    assert manifest["scenario_configs"] == expected_configs
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "COMPLETE"
    assert len(checkpoint["completed_runs"]) == 2
    assert checkpoint["unfinished_runs"] == []
    original_report = (output / "diagnostics.json").read_bytes()
    with pytest.raises(FileExistsError):
        stage1_diagnostics.run_value_direction(config, protocol, output, source=source)
    assert (output / "diagnostics.json").read_bytes() == original_report
    for changed_config, changed_protocol in (
        (replace(config, value_sensitivity=config.value_sensitivity + 0.1), protocol),
        (config, replace(protocol, seeds=(72,))),
        (config, replace(protocol, criteria={"minimum_excess_kurtosis": 0.2})),
    ):
        with pytest.raises(ValueError):
            stage1_diagnostics.run_value_direction(
                changed_config, changed_protocol, tmp_path / "incompatible", source=source,
            )
        assert not (tmp_path / "incompatible").exists()

    source_result = source / "full_71.json"
    original_result = source_result.read_text(encoding="utf-8")
    for field in ("metrics", "fingerprint"):
        poisoned = json.loads(original_result)
        if field == "metrics":
            poisoned[field]["mean_return"] += 1.0
        else:
            poisoned[field] = "not-the-original-fingerprint"
        source_result.write_text(json.dumps(poisoned), encoding="utf-8")
        rejected = tmp_path / f"{field}_mismatch"
        with pytest.raises(RuntimeError, match=field):
            stage1_diagnostics.run_value_direction(
                config, protocol, rejected, source=source, workers=1,
            )
        checkpoint = json.loads((rejected / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["status"] == "ENGINEERING_ERROR"
        assert checkpoint["completed_runs"] == []
        assert len(checkpoint["unfinished_runs"]) == 2
        source_result.write_text(original_result, encoding="utf-8")

    original_replay = stage1_diagnostics._run_value_direction

    def fail_candidate(scenario, *args, **kwargs):
        if scenario == "candidate":
            raise RuntimeError("missing candidate fixture")
        return original_replay(scenario, *args, **kwargs)

    monkeypatch.setattr(stage1_diagnostics, "_run_value_direction", fail_candidate)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="missing candidate fixture"):
        stage1_diagnostics.run_value_direction(
            config, protocol, failed, source=source, workers=1,
        )
    checkpoint = json.loads((failed / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "ENGINEERING_ERROR"
    assert checkpoint["completed_runs"] == [{"scenario": "baseline", "seed": 71}]
    assert checkpoint["unfinished_runs"] == [{"scenario": "candidate", "seed": 71}]
    assert (failed / "baseline_71.json").is_file()
    assert not (failed / "diagnostics.json").exists()


def test_value_budget_replays_both_sources_and_preserves_four_branch_results(
    tmp_path, monkeypatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict, replace

    from abm import stage1_diagnostics
    from abm.config import load_stage1_config
    from abm.harness import MarketHarness, run_stage1
    from abm.mechanism_validation import MechanismProtocol, result_metrics
    from abm.population import VALUE_STRATEGY

    config = replace(
        load_stage1_config(PROJECT_ROOT / "configs" / "stage1_structural_dev_v7.json"),
        population_size=30, trading_days=35, burn_in_days=5,
        liquidity_scale=500.0, announcements=(), seed=71,
        value_order_constraint="valuation_direction", value_inventory_control="budget_priority",
        target_position_fraction=0.01,
    )
    scenarios = {
        "baseline": replace(config, value_order_constraint="none", value_inventory_control="none"),
        "direction_only": replace(config, value_inventory_control="none"),
        "budget_only": replace(config, value_order_constraint="none"),
        "candidate": config,
    }
    protocol = MechanismProtocol("new", "value budget", (71,), (), {})
    original_protocol = replace(protocol, version="old", purpose="value direction")
    references = {name: run_stage1(item) for name, item in scenarios.items()}
    source = tmp_path / "value_direction"
    source.mkdir()
    old_configs = {}
    for old_name, name in (("baseline", "baseline"), ("candidate", "direction_only")):
        old_config = scenarios[name].to_dict()
        del old_config["value_inventory_control"]
        old_configs[old_name] = old_config
        (source / f"{old_name}_71.json").write_text(json.dumps({
            "scenario": old_name, "seed": 71,
            "metrics": result_metrics(references[name], 5),
            "fingerprint": references[name].fingerprint(),
        }), encoding="utf-8")
    (source / "manifest.json").write_text(json.dumps({
        "config": old_configs["candidate"], "protocol": asdict(original_protocol),
        "scenario_configs": old_configs,
    }), encoding="utf-8")
    monkeypatch.setattr(stage1_diagnostics, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(stage1_diagnostics, "as_completed", lambda futures: iter(futures))
    output = tmp_path / "budget"
    report = stage1_diagnostics.run_value_budget(
        config, protocol, output, source=source, workers=1,
    )
    assert report["decision"] == "DEVELOPMENT_DIAGNOSTICS_ONLY"
    assert report["stage1_complete"] is False
    assert report["may_enter_stage2"] is False
    assert [run["scenario"] for run in report["runs"]] == list(scenarios)
    for run in report["runs"]:
        plain = references[run["scenario"]]
        assert run["fingerprint"] == plain.fingerprint()
        assert run["metrics"] == result_metrics(plain, 5)
        assert run["probes"] == []
        if run["scenario"] in ("baseline", "direction_only"):
            assert run["source_verification"]["all_metrics_exact"] is True
            assert run["source_verification"]["fingerprint_exact"] is True
        daily = np.genfromtxt(output / run["daily_file"], delimiter=",", names=True)
        np.testing.assert_allclose(
            daily["signed_log_price_gap"],
            np.log(plain.prices[1:] / plain.fundamentals[1:]), atol=1e-13,
        )
        if run["scenario"] == "candidate":
            trace = MarketHarness(config)
            trace.policy.instrument_orders = True
            for index, row in enumerate(daily):
                before_positions = trace.population.positions.copy()
                audit = trace.step(index)
                snapshot = trace.policy.order_diagnostics
                value = trace.population.strategies == VALUE_STRATEGY
                orders = {
                    "before_constraint": snapshot.before_constraint,
                    "after_direction": snapshot.after_constraint,
                    "after_budget": snapshot.after_budget,
                    "after_risk": snapshot.after_risk,
                    "filled": trace.population.positions - before_positions,
                }
                for stage, quantities in orders.items():
                    selected = quantities[value]
                    valuation = audit.execution_price if stage == "filled" else audit.price_before
                    for side, shares in (
                        ("buy", np.maximum(selected, 0).sum()),
                        ("sell", -np.minimum(selected, 0).sum()),
                        ("net", selected.sum()),
                    ):
                        assert row[f"value_{stage}_{side}_shares"] == pytest.approx(shares)
                        assert row[f"value_{stage}_{side}_notional"] == pytest.approx(shares * valuation)
                closing_limits = np.minimum(
                    config.target_position_fraction * trace.population.reference_wealth
                    / audit.mid_price_after, config.position_cap,
                )
                for stage, positions, limits, valuation in (
                    ("opening", before_positions, snapshot.budget_limit, audit.price_before),
                    ("after_budget", before_positions + snapshot.after_budget,
                     snapshot.budget_limit, audit.price_before),
                    ("after_fill_decision_limit", trace.population.positions,
                     snapshot.budget_limit, audit.price_before),
                    ("closing", trace.population.positions, closing_limits, audit.mid_price_after),
                ):
                    for side, excess in (
                        ("long", np.maximum(positions[value] - limits[value], 0)),
                        ("short", np.maximum(-limits[value] - positions[value], 0)),
                    ):
                        prefix = f"value_{stage}_{side}_excess_"
                        assert row[prefix + "count"] == np.count_nonzero(excess > 1e-9)
                        assert row[prefix + "shares"] == pytest.approx(excess.sum())
                        assert row[prefix + "notional"] == pytest.approx(excess.sum() * valuation)
                assert row["value_policy_forced_cover_shares"] == pytest.approx(
                    snapshot.forced_cover_orders[value].sum()
                )
                additional_covers = np.maximum(orders["filled"] - np.maximum(snapshot.after_risk, 0), 0)
                assert row["value_settlement_additional_cover_shares"] == pytest.approx(
                    additional_covers[value].sum()
                )
            assert trace.run().fingerprint() == plain.fingerprint()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scenario_configs"] == {
        name: item.to_dict() for name, item in scenarios.items()
    }
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "COMPLETE"
    assert checkpoint["expected_runs"] == len(checkpoint["completed_runs"]) == 4
    assert checkpoint["unfinished_runs"] == []
    assert report["paired_seed_count"] == 1
    assert report["paired_comparison_count"] == 3
    for pair in report["paired_comparisons"]:
        reference_name = pair["comparison"].removeprefix("candidate_minus_")
        assert reference_name in ("direction_only", "budget_only", "baseline")
        expected_reference = result_metrics(references[reference_name], 5)
        expected_candidate = result_metrics(references["candidate"], 5)
        for name, values in pair["metrics"].items():
            assert values["reference"] == expected_reference[name]
            assert values["candidate"] == expected_candidate[name]
            assert values["delta"] == pytest.approx(expected_candidate[name] - expected_reference[name])
    original_report = (output / "diagnostics.json").read_bytes()
    with pytest.raises(FileExistsError):
        stage1_diagnostics.run_value_budget(config, protocol, output, source=source)
    assert (output / "diagnostics.json").read_bytes() == original_report
    for changed_config, changed_protocol in (
        (replace(config, max_order_fraction=config.max_order_fraction + 0.01), protocol),
        (config, replace(protocol, seeds=(72,))),
        (config, replace(protocol, minimum_burn_in_days=1)),
    ):
        with pytest.raises(ValueError):
            stage1_diagnostics.run_value_budget(
                changed_config, changed_protocol, tmp_path / "incompatible", source=source,
            )
        assert not (tmp_path / "incompatible").exists()

    for old_name, field in (("baseline", "metrics"), ("candidate", "fingerprint")):
        source_result = source / f"{old_name}_71.json"
        original_result = source_result.read_text(encoding="utf-8")
        poisoned = json.loads(original_result)
        if field == "metrics":
            poisoned[field]["mean_return"] += 1.0
        else:
            poisoned[field] = "not-the-original-fingerprint"
        source_result.write_text(json.dumps(poisoned), encoding="utf-8")
        with pytest.raises(RuntimeError, match=field):
            stage1_diagnostics.run_value_budget(
                config, protocol, tmp_path / f"{old_name}_mismatch", source=source, workers=1,
            )
        source_result.write_text(original_result, encoding="utf-8")

    source_manifest = source / "manifest.json"
    original_manifest = source_manifest.read_text(encoding="utf-8")
    poisoned_manifest = json.loads(original_manifest)
    poisoned_manifest["scenario_configs"]["baseline"]["max_order_fraction"] += 0.01
    source_manifest.write_text(json.dumps(poisoned_manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        stage1_diagnostics.run_value_budget(
            config, protocol, tmp_path / "baseline_config_mismatch", source=source,
        )
    assert not (tmp_path / "baseline_config_mismatch").exists()
    source_manifest.write_text(original_manifest, encoding="utf-8")

    original_replay = stage1_diagnostics._run_value_budget

    def fail_budget_only(scenario, *args, **kwargs):
        if scenario == "budget_only":
            raise RuntimeError("missing budget fixture")
        return original_replay(scenario, *args, **kwargs)

    monkeypatch.setattr(stage1_diagnostics, "_run_value_budget", fail_budget_only)
    failed = tmp_path / "failed_budget"
    with pytest.raises(RuntimeError, match="missing budget fixture"):
        stage1_diagnostics.run_value_budget(
            config, protocol, failed, source=source, workers=1,
        )
    checkpoint = json.loads((failed / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "ENGINEERING_ERROR"
    assert checkpoint["completed_runs"] == [
        {"scenario": "baseline", "seed": 71}, {"scenario": "direction_only", "seed": 71},
    ]
    assert checkpoint["unfinished_runs"] == [
        {"scenario": "budget_only", "seed": 71}, {"scenario": "candidate", "seed": 71},
    ]
    assert (failed / "baseline_71.json").is_file()
    assert (failed / "direction_only_71.json").is_file()
    assert not (failed / "diagnostics.json").exists()


@pytest.mark.parametrize(
    "stage,positions,limit,price,long_excess,short_excess",
    [
        ("opening", [12, -15, 5, -2, 20, -30], 10, 10, 2, 5),
        ("after_budget", [10, -11, 5, -2, 20, -30], 10, 10, 0, 1),
        ("after_fill_decision_limit", [11, -12, 5, -2, 20, -30], 10, 10, 1, 2),
        ("closing", [11, -12, 5, -2, 20, -30], 5, 20, 6, 7),
        ("opening", [1, -2, 0, 0, 20, -30], 0, 10, 1, 2),
    ],
)
def test_value_budget_excess_keeps_both_sides_and_valuation_times(
    stage, positions, limit, price, long_excess, short_excess,
) -> None:
    from abm.population import NOISE_STRATEGY, TREND_STRATEGY, VALUE_STRATEGY
    from abm.stage1_diagnostics import _inventory_excess_fields

    values = _inventory_excess_fields(
        np.array(positions, dtype=float), np.full(6, limit, dtype=float),
        np.array([VALUE_STRATEGY] * 4 + [TREND_STRATEGY, NOISE_STRATEGY]), price, stage,
    )
    for side, shares in (("long", long_excess), ("short", short_excess)):
        prefix = f"value_{stage}_{side}_excess_"
        assert values[prefix + "count"] == int(shares > 0)
        assert values[prefix + "shares"] == shares
        assert values[prefix + "notional"] == shares * price
    assert values[f"value_{stage}_position_gross_notional"] == sum(abs(item) for item in positions[:4]) * price
    assert values[f"value_{stage}_position_net_notional"] == sum(positions[:4]) * price
    assert values[f"value_{stage}_target_limit_notional"] == 4 * limit * price


@pytest.mark.parametrize(
    "stage,orders,margin,price,sell_parts,buy_parts",
    [
        ("after_budget", [-5, -4, 6, 0, -40], [0, 0, 0, 0, 0], 10,
         (9, 7, 0, 2), (6, 3, 0, 3)),
        ("after_risk", [-4, -2, 4, 3, -20], [0, 0, 0, 3, 0], 10,
         (6, 4.4, 0, 1.6), (7, 2, 3, 2)),
        ("filled", [-2, -1, 2, 5, -10], [0, 0, 0, 5, 0], 12,
         (3, 2.2, 0, 0.8), (7, 1, 5, 1)),
    ],
)
def test_value_budget_inverse_orders_separate_budget_margin_and_other(
    stage, orders, margin, price, sell_parts, buy_parts,
) -> None:
    from abm.population import TREND_STRATEGY, VALUE_STRATEGY
    from abm.stage1_diagnostics import _budget_inverse_fields

    values = _budget_inverse_fields(
        np.array([-2, 0, 3, 0, -40], dtype=float),
        np.array([-5, -4, 6, 0, -40], dtype=float), np.array(orders, dtype=float),
        np.array([1, 1, -1, -1, 1], dtype=float),
        np.array([VALUE_STRATEGY] * 4 + [TREND_STRATEGY]), price, stage,
        margin_orders=np.array(margin, dtype=float),
    )
    for direction, expected in (
        ("positive_signal_sell", sell_parts), ("negative_signal_buy", buy_parts),
    ):
        prefix = f"value_{stage}_{direction}_"
        for component, shares in zip(("total", "budget", "margin", "other"), expected):
            assert values[f"{prefix}{component}_shares"] == pytest.approx(shares)
            assert values[f"{prefix}{component}_notional"] == pytest.approx(shares * price)
        assert values[f"{prefix}total_shares"] == pytest.approx(sum(
            values[f"{prefix}{component}_shares"] for component in ("budget", "margin", "other")
        ))
    assert values[f"value_{stage}_positive_signal_sell_total_count"] == 2
    assert values[f"value_{stage}_positive_signal_sell_budget_count"] == 2
    assert values[f"value_{stage}_positive_signal_sell_other_count"] == 1
    assert values[f"value_{stage}_negative_signal_buy_budget_count"] == 1
    assert values[f"value_{stage}_negative_signal_buy_margin_count"] == int(stage != "after_budget")
