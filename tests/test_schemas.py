from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from abm.scenario import load_synthetic_scenario
from abm.schemas import MarketState, Message, Order

SCENARIO_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "stage0_10_day.json"
)


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
def test_invalid_order_quantities_are_rejected(side: str, quantity: float) -> None:
    with pytest.raises(ValueError):
        Order(
            agent_id="trader-001",
            asset_id="ACME",
            side=side,
            quantity=quantity,
            submitted_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            reason_code="test",
        )
