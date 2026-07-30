"""Loading and validation for the frozen Stage 0 synthetic scenario."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

from .schemas import (
    AgentState,
    CompanyState,
    Edge,
    FactorEvent,
    JsonValue,
    MarketState,
    Message,
    Order,
)


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    schema_version: str
    scenario_id: str
    start_date: date
    end_date: date
    market_states: tuple[MarketState, ...]
    agent_states: tuple[AgentState, ...]
    company_states: tuple[CompanyState, ...]
    messages: tuple[Message, ...]
    edges: tuple[Edge, ...]
    orders: tuple[Order, ...]
    factor_events: tuple[FactorEvent, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, JsonValue]) -> "SyntheticScenario":
        required = {
            "schema_version",
            "scenario_id",
            "start_date",
            "end_date",
            "market_states",
            "agent_states",
            "company_states",
            "messages",
            "edges",
            "orders",
            "factor_events",
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing or unknown:
            raise ValueError(
                f"invalid scenario keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )

        def records(name: str) -> list[dict[str, JsonValue]]:
            value = data[name]
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise TypeError(f"{name} must be a JSON array of objects")
            return value

        scenario = cls(
            schema_version=str(data["schema_version"]),
            scenario_id=str(data["scenario_id"]),
            start_date=date.fromisoformat(str(data["start_date"])),
            end_date=date.fromisoformat(str(data["end_date"])),
            market_states=tuple(
                MarketState.from_dict(item) for item in records("market_states")
            ),
            agent_states=tuple(
                AgentState.from_dict(item) for item in records("agent_states")
            ),
            company_states=tuple(
                CompanyState.from_dict(item) for item in records("company_states")
            ),
            messages=tuple(Message.from_dict(item) for item in records("messages")),
            edges=tuple(Edge.from_dict(item) for item in records("edges")),
            orders=tuple(Order.from_dict(item) for item in records("orders")),
            factor_events=tuple(
                FactorEvent.from_dict(item) for item in records("factor_events")
            ),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("scenario end_date must not precede start_date")
        market_dates = [state.date for state in self.market_states]
        if len(market_dates) != 10 or len(set(market_dates)) != 10:
            raise ValueError("Stage 0 scenario must contain ten distinct market dates")
        if market_dates != sorted(market_dates):
            raise ValueError("market_states must be ordered by date")
        if market_dates[0] != self.start_date or market_dates[-1] != self.end_date:
            raise ValueError("scenario bounds must match the market-state dates")
        agent_ids = {state.agent_id for state in self.agent_states}
        if len(agent_ids) != len(self.agent_states):
            raise ValueError("agent_id values must be unique")
        if any(order.agent_id not in agent_ids for order in self.orders):
            raise ValueError("every order must reference a known agent")


def load_synthetic_scenario(path: str | Path) -> SyntheticScenario:
    with Path(path).open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError("scenario root must be a JSON object")
    return SyntheticScenario.from_dict(payload)

