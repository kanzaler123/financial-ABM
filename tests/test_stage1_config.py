from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.config import load_stage1_config
from abm.harness import build_fundamental_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_stage1_config_and_external_announcements_load() -> None:
    config = load_stage1_config(CONFIG_PATH)
    path = build_fundamental_path(
        replace(
            config,
            fundamental_drift=0.0,
            fundamental_volatility=0.0,
        )
    )

    assert config.population_size == 1000
    assert path[59] == 100.0
    assert path[60] == 102.0
    assert path[120] == 98.0
    assert path[180] == 101.0


def test_stochastic_fundamental_path_is_seed_reproducible() -> None:
    config = load_stage1_config(CONFIG_PATH)

    first = build_fundamental_path(config)
    second = build_fundamental_path(config)

    assert first.tolist() == second.tolist()
    assert first[1] != config.initial_fundamental
    assert config.fundamental_process == "gaussian"
    assert config.public_news_price_pass_through == 0.0


def test_strategy_shares_must_sum_to_one() -> None:
    config = load_stage1_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="sum to one"):
        replace(
            config,
            strategy_shares={"value": 0.5, "trend": 0.5, "noise": 0.5},
        )


def test_fundamental_volatility_dynamics_must_be_stationary() -> None:
    config = load_stage1_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="must be below one"):
        replace(
            config,
            fundamental_process="garch_t",
            fundamental_shock_df=5.5,
            fundamental_arch=0.2,
            fundamental_garch=0.8,
        )


def test_shock_process_parameters_cannot_be_hidden_in_gaussian_baseline() -> None:
    config = load_stage1_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="requires zero df/ARCH/GARCH"):
        replace(config, fundamental_shock_df=5.5)


def test_garch_t_is_an_explicit_optional_process() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        fundamental_process="garch_t",
        fundamental_shock_df=5.5,
        fundamental_arch=0.11,
        fundamental_garch=0.88,
    )

    first = build_fundamental_path(config)
    second = build_fundamental_path(config)

    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_external_fundamental_innovations_are_reproducible() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        trading_days=5,
        announcements=(),
    )
    innovations = np.linspace(-1.0, 1.0, config.trading_days)

    first = build_fundamental_path(config, innovations=innovations)
    second = build_fundamental_path(config, innovations=innovations)

    assert np.array_equal(first, second)
