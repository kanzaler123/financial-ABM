"""Stage 1 mechanism protocol and freeze-manifest tests."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.config import load_stage1_config
from abm.harness import run_stage1
from abm.mechanism_validation import (
    MechanismProtocol,
    build_scenarios,
    main as mechanism_main,
    result_metrics,
    run_protocol,
)
from abm.stage1_freeze import (
    build_freeze_manifest,
    verify_freeze_manifest,
    write_freeze_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_protocol_separates_development_and_formal_seeds(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        """
        {
          "version": "test",
          "purpose": "test",
          "seeds": [1, 2],
          "excluded_development_seeds": [3, 4],
          "criteria": {"minimum_daily_volatility": 0.001}
        }
        """,
        encoding="utf-8",
    )

    protocol = MechanismProtocol.from_json(path)

    assert protocol.seeds == (1, 2)
    assert protocol.excluded_development_seeds == (3, 4)


def test_seed_groups_and_criteria_are_pre_registered() -> None:
    tuning = MechanismProtocol.from_json(
        PROJECT_ROOT / "configs" / "stage1_mechanism_dev_v8.json"
    )
    holdout = MechanismProtocol.from_json(
        PROJECT_ROOT / "configs" / "stage1_mechanism_holdout_v7.json"
    )
    formal = MechanismProtocol.from_json(
        PROJECT_ROOT / "configs" / "stage1_mechanism_acceptance_v7.json"
    )

    assert tuning.stage == holdout.stage == "development"
    assert formal.stage == "formal"
    assert len(tuning.seeds) == 20
    assert len(holdout.seeds) == 10
    assert len(formal.seeds) == 50
    assert not (set(tuning.seeds) & set(holdout.seeds))
    assert not (set(tuning.seeds) & set(formal.seeds))
    assert not (set(holdout.seeds) & set(formal.seeds))
    assert tuning.criteria == holdout.criteria == formal.criteria
    assert formal.minimum_burn_in_days == 1000
    assert formal.minimum_evaluation_days == 2500


def test_protocol_rejects_unknown_keys_and_criteria(tmp_path) -> None:
    unknown_key = tmp_path / "unknown-key.json"
    unknown_key.write_text(
        """
        {
          "version": "test",
          "purpose": "test",
          "seeds": [1],
          "excluded_development_seeds": [2],
          "criteria": {},
          "unexpected": true
        }
        """,
        encoding="utf-8",
    )
    unknown_criterion = tmp_path / "unknown-criterion.json"
    unknown_criterion.write_text(
        """
        {
          "version": "test",
          "purpose": "test",
          "seeds": [1],
          "excluded_development_seeds": [2],
          "criteria": {"not_a_gate": 1.0}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid mechanism protocol keys"):
        MechanismProtocol.from_json(unknown_key)
    with pytest.raises(ValueError, match="unknown mechanism criteria"):
        MechanismProtocol.from_json(unknown_criterion)


def test_scenarios_make_complex_shocks_explicit_controls() -> None:
    config = load_stage1_config(CONFIG_PATH)
    scenarios = build_scenarios(config)

    assert scenarios["full"].fundamental_process == "gaussian"
    assert scenarios["full"].public_news_price_pass_through == 0.9
    assert scenarios["no_direct_news"].public_news_price_pass_through == 0.0
    assert scenarios["no_agent_information"].value_sensitivity == 0.0
    assert scenarios["no_agent_information"].value_update_rate == 0.0
    assert (
        scenarios["no_information_channels"].public_news_price_pass_through
        == 0.0
    )
    assert scenarios["no_information_channels"].value_sensitivity == 0.0
    assert scenarios["garch_t_control"].fundamental_process == "garch_t"
    assert scenarios["independent_signals"].common_signal_correlation == 0.0
    assert scenarios["fixed_liquidity"].liquidity_volatility_sensitivity == 0.0
    assert scenarios["fixed_liquidity"].minimum_liquidity_fraction == 1.0
    assert not scenarios["no_logit"].learning_enabled
    assert scenarios["logit_learning"].learning_enabled


def test_result_metrics_align_post_burn_in_days() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=80,
        trading_days=12,
        burn_in_days=5,
        announcements=(),
    )
    result = run_stage1(config)
    metrics = result_metrics(result, burn_in_days=config.burn_in_days)
    closing_prices = result.prices[config.burn_in_days + 1 :]
    expected_spread_bps = float(
        np.mean(result.spreads[config.burn_in_days :] / closing_prices) * 1e4
    )
    expected_gap = float(
        np.mean(
            np.abs(
                np.log(
                    closing_prices
                    / result.fundamentals[config.burn_in_days + 1 :]
                )
            )
        )
    )

    assert np.isclose(metrics["mean_spread_bps"], expected_spread_bps)
    assert np.isclose(metrics["mean_absolute_log_price_gap"], expected_gap)


def test_result_metrics_are_finite_and_capture_strategy_turnover() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=120,
        trading_days=60,
        burn_in_days=0,
        announcements=(),
    )
    result = run_stage1(config)
    metrics = result_metrics(result)

    assert set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["price_cap_hits"] == 0.0
    assert metrics["strategy_turnover"] == 0.0
    assert metrics["mean_spread_bps"] > 0.0
    assert metrics["mean_depth"] > 0.0


def test_formal_protocol_requires_unseal_and_freeze_manifest(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=10,
        trading_days=4,
        burn_in_days=0,
        announcements=(),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    protocol_path = tmp_path / "formal.json"
    protocol_path.write_text(
        """
        {
          "version": "formal-test",
          "stage": "formal",
          "purpose": "test",
          "seeds": [1],
          "excluded_development_seeds": [2],
          "criteria": {}
        }
        """,
        encoding="utf-8",
    )
    arguments = [
        "--config", str(config_path),
        "--protocol", str(protocol_path),
        "--output", str(tmp_path / "output"),
    ]

    with pytest.raises(PermissionError, match="unseal"):
        mechanism_main(arguments)
    with pytest.raises(ValueError, match="freeze-manifest"):
        mechanism_main([*arguments, "--unseal-formal"])


def test_run_protocol_checkpoint_resumes_without_duplicate_tasks(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=20,
        trading_days=12,
        burn_in_days=0,
        announcements=(),
    )
    protocol = MechanismProtocol(
        version="checkpoint-test",
        purpose="test",
        seeds=(11,),
        excluded_development_seeds=(12,),
        criteria={
            "minimum_seed_pass_rate": 0.0,
            "minimum_paired_seed_pass_rate": 0.0,
        },
    )
    checkpoint = tmp_path / "checkpoint.json"

    first_rows, first_summary, first_gate = run_protocol(
        config,
        protocol,
        checkpoint_path=checkpoint,
    )
    resumed_rows, resumed_summary, resumed_gate = run_protocol(
        config,
        protocol,
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert checkpoint.is_file()
    assert resumed_rows == first_rows
    assert resumed_summary == first_summary
    assert resumed_gate == first_gate
    assert len(resumed_rows) == len(build_scenarios(config))
    assert len(
        {(row["scenario"], row["seed"]) for row in resumed_rows}
    ) == len(resumed_rows)


def test_strategy_turnover_counts_actual_learning_updates() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=120,
        trading_days=120,
        burn_in_days=0,
        learning_enabled=True,
        announcements=(),
    )
    result = run_stage1(config)
    metrics = result_metrics(result)

    assert metrics["strategy_turnover"] == sum(
        audit.updated_agents for audit in result.learning_audits
    )
    assert metrics["strategy_turnover"] > 0.0


# --- Freeze manifest ------------------------------------------------------


def _freeze_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "src" / "abm").mkdir(parents=True)
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    files = {
        "src/abm/core.py": "VALUE = 1\n",
        "web/src/App.tsx": "export default 1\n",
        "web/src/app.css": "body {}\n",
        "web/package.json": "{}\n",
        "web/package-lock.json": "{}\n",
        "pyproject.toml": "[project]\nname='test'\n",
        "requirements-lock.txt": "pytest==1\n",
        "configs/formal.json": "{}\n",
        "configs/protocol.json": "{}\n",
    }
    for relative, content in files.items():
        (tmp_path / relative).write_text(content, encoding="utf-8")
    return (
        tmp_path / "configs" / "formal.json",
        tmp_path / "configs" / "protocol.json",
        tmp_path / "freeze.json",
    )


def test_freeze_manifest_detects_changed_file(tmp_path) -> None:
    config, protocol, output = _freeze_fixture(tmp_path)
    manifest = build_freeze_manifest(
        tmp_path,
        formal_config=config,
        formal_protocol=protocol,
    )
    write_freeze_manifest(output, manifest)

    verified = verify_freeze_manifest(
        output,
        project_root=tmp_path,
        formal_config=config,
        formal_protocol=protocol,
    )
    assert verified["verified"]

    (tmp_path / "src" / "abm" / "core.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mismatched"):
        verify_freeze_manifest(output, project_root=tmp_path)


def test_freeze_manifest_identity_is_tamper_evident(tmp_path) -> None:
    config, protocol, output = _freeze_fixture(tmp_path)
    manifest = build_freeze_manifest(
        tmp_path,
        formal_config=config,
        formal_protocol=protocol,
    )
    write_freeze_manifest(output, manifest)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["formal_protocol"] = "configs/other.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity hash"):
        verify_freeze_manifest(output, project_root=tmp_path)
