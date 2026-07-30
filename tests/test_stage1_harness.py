import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.config import load_stage1_config
from abm.harness import MarketHarness, run_stage1
from abm.schemas import RunManifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_same_seed_and_config_replay_exactly() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=200,
        trading_days=80,
        announcements=(),
    )

    first = run_stage1(config)
    second = run_stage1(config)

    assert first.fingerprint() == second.fingerprint()
    assert np.array_equal(first.prices, second.prices)
    assert np.array_equal(first.final_agent_cash, second.final_agent_cash)
    assert np.array_equal(first.final_strategies, second.final_strategies)


def test_disabling_learning_recovers_fixed_rule_baseline_exactly() -> None:
    base = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=200,
        trading_days=59,
        announcements=(),
    )

    disabled = run_stage1(replace(base, learning_enabled=False))
    not_yet_updated = run_stage1(replace(base, learning_enabled=True))

    assert disabled.fingerprint() == not_yet_updated.fingerprint()
    assert np.array_equal(disabled.prices, not_yet_updated.prices)
    assert np.array_equal(
        disabled.final_strategies, not_yet_updated.final_strategies
    )


def test_duplicate_or_out_of_order_settlement_is_rejected() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=20,
        trading_days=2,
        announcements=(),
    )
    harness = MarketHarness(config)
    harness.step(0)

    with pytest.raises(RuntimeError, match="settled twice"):
        harness.step(0)


def test_logit_updates_are_traceable_to_strategy_fitness() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=300,
        trading_days=120,
        announcements=(),
    )
    result = run_stage1(config)

    assert [audit.day for audit in result.learning_audits] == [60, 120]
    for audit in result.learning_audits:
        assert sum(audit.counts_before) == config.population_size
        assert sum(audit.counts_after) == config.population_size
        assert np.isclose(sum(audit.choice_probabilities), 1.0)
        best_fitness = int(np.argmax(audit.mean_fitness_by_strategy))
        best_probability = int(np.argmax(audit.choice_probabilities))
        assert best_probability == best_fitness
        assert audit.updated_agents == round(
            config.population_size * config.learning_update_fraction
        )


def test_full_stage1_acceptance_run_meets_invariants() -> None:
    config = load_stage1_config(CONFIG_PATH)
    result = run_stage1(config)

    assert result.prices.shape == (251,)
    assert len(result.daily_audits) == 250
    assert len({audit.settlement_id for audit in result.daily_audits}) == 250
    assert np.all(np.isfinite(result.prices))
    assert np.all(result.prices > 0)
    assert np.all(result.final_agent_cash >= 0)
    assert np.all(
        result.final_agent_positions
        >= -config.max_short_leverage
        * (
            result.final_agent_cash
            + result.final_agent_positions * result.prices[-1]
        )
        / result.prices[-1]
        - 1e-8
    )
    assert np.max(np.abs(result.total_cash - result.total_cash[0])) < 1e-6
    assert np.max(np.abs(result.total_shares - result.total_shares[0])) < 1e-8
    assert result.strategy_counts.shape == (251, 3)
    assert np.all(result.strategy_counts.sum(axis=1) == 1000)
    assert len(result.learning_audits) == 4
    assert all(audit.direct_news_return == 0.0 for audit in result.daily_audits)
    demand_log_returns = (
        config.price_impact
        * result.submitted_net_demand
        / config.liquidity_scale
    )
    assert float(np.max(np.abs(demand_log_returns))) < config.max_log_return
    assert not any(audit.price_cap_hit for audit in result.daily_audits)


def test_public_news_changes_price_only_through_agent_demand_by_default() -> None:
    base = load_stage1_config(CONFIG_PATH)
    config = replace(
        base,
        population_size=200,
        trading_days=60,
        fundamental_drift=0.0,
        fundamental_volatility=0.0,
        announcements=(base.announcements[0],),
    )
    result = run_stage1(config)
    announcement_audit = result.daily_audits[59]

    assert announcement_audit.public_news_return != 0.0
    assert announcement_audit.direct_news_return == 0.0
    assert np.isclose(
        np.log(
            announcement_audit.execution_price
            / announcement_audit.price_before
        ),
        announcement_audit.demand_log_return,
    )


def test_learning_disabled_keeps_fixed_strategy_population_for_250_days() -> None:
    config = replace(load_stage1_config(CONFIG_PATH), learning_enabled=False)
    result = run_stage1(config)

    assert not result.learning_audits
    assert np.all(result.strategy_counts == result.strategy_counts[0])


def test_run_outputs_are_auditable_and_not_overwritten(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=20,
        trading_days=5,
        announcements=(),
    )
    result = run_stage1(config)
    output = tmp_path / "run"

    result.write(output, config)

    assert (output / "state_arrays.npz").is_file()
    manifest = RunManifest.from_dict(
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.random_seed == config.seed
    assert len((output / "daily_audit.jsonl").read_text().splitlines()) == 5
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["fingerprint"] == result.fingerprint()
    with pytest.raises(FileExistsError):
        result.write(output, config)
