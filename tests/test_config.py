"""Config validation and data-contract tests."""
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from abm.config import load_stage1_config
from abm.harness import build_fundamental_path
from abm.scenario import load_synthetic_scenario
from abm.schemas import MarketState, Message, Order

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"
SCENARIO_PATH = PROJECT_ROOT / "data" / "synthetic" / "stage0_10_day.json"


# --- Stage1Config loading and fundamental path ---------------------------


def test_stage1_config_and_external_announcements_load() -> None:
    config = load_stage1_config(CONFIG_PATH)
    path = build_fundamental_path(
        replace(
            config,
            fundamental_drift=0.0,
            fundamental_volatility=0.0,
        )
    )

    assert config.population_size == 10000
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
    assert config.public_news_price_pass_through == 0.9
    assert config.price_formation == "quasi_order_book"
    assert config.burn_in_days == 1000


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
        burn_in_days=0,
        announcements=(),
    )
    innovations = np.linspace(-1.0, 1.0, config.trading_days)

    first = build_fundamental_path(config, innovations=innovations)
    second = build_fundamental_path(config, innovations=innovations)

    assert np.array_equal(first, second)


# --- Stage 0 data contracts ----------------------------------------------


def test_market_state_round_trip() -> None:
    state = MarketState(
        date=date(2026, 1, 5),
        asset_id="ACME",
        price=100.0,
        fundamental_value=101.0,
        volume=0.0,
        market_maker_inventory=10_000.0,
        market_maker_cash=1_000_000.0,
    )

    assert MarketState.from_dict(state.to_dict()) == state


def test_all_stage0_contracts_round_trip() -> None:
    scenario = load_synthetic_scenario(SCENARIO_PATH)
    records = (
        scenario.market_states[0],
        scenario.agent_states[0],
        scenario.company_states[0],
        scenario.messages[0],
        scenario.edges[0],
        scenario.orders[0],
        scenario.factor_events[0],
    )

    for record in records:
        assert type(record).from_dict(record.to_dict()) == record


def test_future_information_cannot_arrive_before_event() -> None:
    with pytest.raises(ValueError, match="available_at"):
        Message(
            message_id="m-1",
            source_id="company",
            target_scope=("public",),
            event_time=datetime(2026, 1, 6, tzinfo=timezone.utc),
            available_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            content_type="announcement",
            payload={},
            credibility=1.0,
        )


@pytest.mark.parametrize(
    ("side", "quantity"),
    [("hold", 1.0), ("buy", 0.0), ("sell", -1.0), ("invalid", 1.0)],
)
def test_invalid_order_quantities_are_rejected(
    side: str, quantity: float
) -> None:
    with pytest.raises(ValueError):
        Order(
            agent_id="trader-001",
            asset_id="ACME",
            side=side,
            quantity=quantity,
            submitted_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            reason_code="test",
        )
