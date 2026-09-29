"""Daily graph market with twenty-day bandits and sixty-day company learning."""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import Stage1Config
from .harness import MarketHarness, SimulationResult
from .information import InformationGraph
from .manifest import build_run_manifest, canonical_json_bytes
from .policies import RuleBasedPolicy
from .schemas import Edge, Message, RunManifest
from .tabular import (COMPANY_ACTIONS, COMPANY_THRESHOLDS, TRADER_ACTIONS,
                      TRADER_THRESHOLDS, CompanyQLearner, ContextualBandit, ternary_states)
from .telemetry import TelemetryEvent, TelemetrySink

EPOCH = datetime(2026, 1, 5, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Stage2Config(Stage1Config):
    stage2_enabled: bool = True
    bandit_enabled: bool = True
    company_learning_enabled: bool = True
    graph_enabled: bool = True
    dynamic_graph_enabled: bool = False
    graph_fixture: str = ""
    company_count: int = 6
    bandit_epsilon: float = 0.1
    q_epsilon: float = 0.1
    q_learning_rate: float = 0.1
    q_discount: float = 0.9
    graph_adaptation_rate: float = 0.1
    drawdown_penalty: float = 1.0
    real_investment_fraction: float = 0.02
    disclosure_investment_fraction: float = 0.005
    company_operating_yield: float = 0.0008
    company_interest_rate: float = 0.0002
    investment_productivity: float = 0.05
    capital_retention: float = 0.99
    disclosure_signal_gain: float = 10.0
    verification_base_rate: float = 0.03
    verification_gap_sensitivity: float = 0.5
    verification_penalty_rate: float = 0.5

    def __post_init__(self) -> None:
        Stage1Config.__post_init__(self)
        if self.company_count < 1:
            raise ValueError("company_count must be positive")
        for name in ("bandit_epsilon", "q_epsilon", "graph_adaptation_rate", "real_investment_fraction",
                     "disclosure_investment_fraction", "capital_retention", "verification_base_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 0 < self.q_learning_rate <= 1 or not 0 <= self.q_discount < 1:
            raise ValueError("invalid Q-learning step size or discount")
        for name in ("drawdown_penalty", "company_operating_yield", "company_interest_rate",
                     "investment_productivity", "disclosure_signal_gain",
                     "verification_gap_sensitivity", "verification_penalty_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.active and self.learning_enabled:
            raise ValueError("Stage 2 learners require the Stage 1 Logit switch to be off")

    @property
    def active(self) -> bool:
        return self.stage2_enabled and (self.bandit_enabled or self.company_learning_enabled)

    def base_config(self) -> Stage1Config:
        return Stage1Config(**{f.name: getattr(self, f.name) for f in fields(Stage1Config)})


def simulation_config_from_dict(payload: Mapping[str, Any]) -> Stage1Config:
    return (Stage2Config if "stage2_enabled" in payload else Stage1Config).from_dict(payload)


def load_simulation_config(path: str | Path) -> Stage1Config:
    return simulation_config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class TraderObservation:
    price: float
    price_history: np.ndarray
    public_news_return: float
    realized_volatility: float
    realized_volatility_reference: float
    spread: float
    depth: float


class Stage2Policy:
    def __init__(self, harness: "Stage2Harness"):
        self.harness = harness
        self.rules = RuleBasedPolicy(harness.config.base_config())
        self.diagnostics = self.rules.diagnostics
        self.last_observation: TraderObservation | None = None
        self.submitted_orders = np.zeros(harness.config.population_size)

    def act(self, observation, population, rng):
        history = observation.price_history.copy()
        history.flags.writeable = False
        public = TraderObservation(
            observation.price, history, self.harness.macro_news,
            observation.realized_volatility, observation.realized_volatility_reference,
            observation.spread, observation.depth,
        )
        self.last_observation = public
        orders = self.rules.act(
            public, population, rng,
            actions=self.harness.trader_actions if self.harness.config.bandit_enabled else None,
            public_news_returns=self.harness.visible_news,
        )
        self.diagnostics = self.rules.diagnostics
        self.submitted_orders = orders.copy()
        return orders


@dataclass(frozen=True, slots=True)
class Stage2Result(SimulationResult):
    stage2_manifest: RunManifest
    stage2_audit: dict[str, Any]

    def fingerprint(self) -> str:
        digest = hashlib.sha256(SimulationResult.fingerprint(self).encode("ascii"))
        for event in self.telemetry_events:
            digest.update(canonical_json_bytes(event.to_dict()))
        digest.update(canonical_json_bytes(self.stage2_audit))
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        summary = SimulationResult.summary(self)
        summary.update(stage=2, market_fingerprint=SimulationResult.fingerprint(self),
                       trader_reward_windows=self.stage2_audit["trader_reward_windows"],
                       company_reward_windows=self.stage2_audit["company_reward_windows"],
                       llm_api_calls=0, llm_api_tokens=0)
        return summary

    def write(self, output_dir, config, *, code_revision="working-tree") -> Path:
        destination = SimulationResult.write(self, output_dir, config, code_revision=code_revision)
        (destination / "run_manifest.json").write_text(
            json.dumps(self.stage2_manifest.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (destination / "stage2_audit.json").write_text(
            json.dumps(self.stage2_audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        journal = destination.parent / f".{destination.name}.events.jsonl"
        if journal.is_file():
            journal.replace(destination / "events.jsonl")
        return destination


class Stage2Harness(MarketHarness):
    def __init__(self, config: Stage2Config, telemetry_sink: TelemetrySink | None = None,
                 *, event_log: str | Path | None = None, code_revision: str = "working-tree"):
        super().__init__(config.base_config())
        self.config = config
        self.display_sink = telemetry_sink
        self.event_log = Path(event_log) if event_log is not None else None
        if self.event_log is not None:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("x", encoding="utf-8"):
                pass
        self.macro_fundamentals = self.fundamentals.copy()
        seeds = np.random.SeedSequence([config.seed, 2]).spawn(4)
        self.bandit_rng, self.company_rng, self.environment_rng, trait_rng = [np.random.default_rng(s) for s in seeds]
        self.trader_ids = tuple(f"trader-{i}" for i in range(config.population_size))
        self.company_ids = tuple(f"company-{i}" for i in range(config.company_count))
        self.trader_types = self.population.strategies.astype(np.int64, copy=True)
        self.factor_preferences = trait_rng.uniform(0, 1, config.population_size)
        self.trader_types.flags.writeable = self.factor_preferences.flags.writeable = False
        self.bandit = ContextualBandit(epsilon=config.bandit_epsilon)
        self.company_q = CompanyQLearner(epsilon=config.q_epsilon, learning_rate=config.q_learning_rate, discount=config.q_discount)
        self.trader_actions = np.zeros(config.population_size, dtype=np.int64)
        self.trader_states = np.zeros(config.population_size, dtype=np.int64)
        self.trader_probabilities = np.full((config.population_size, 4), 0.25)
        self.trader_features = np.zeros((config.population_size, 4))
        self.wealth_peak = self.population.wealth(config.initial_price)
        self.window_start_wealth = self.wealth_peak.copy()
        self.window_peak = self.wealth_peak.copy()
        self.window_drawdown = np.zeros(config.population_size)
        self.window_fees = np.zeros(config.population_size)
        self.window_returns: list[np.ndarray] = []
        self.trader_reward_windows = 0
        self.company_reward_windows = 0
        self.company_groups = np.arange(config.company_count) % 3
        self.company_assets = 100_000.0 * 2.0 ** self.company_groups
        self.company_weights = self.company_assets / self.company_assets.sum()
        self.company_cash = self.company_assets * 0.3
        self.initial_company_cash = float(self.company_cash.sum())
        self.company_external_cash = 0.0
        self.company_debt = self.company_assets * 0.3
        self.company_performance = np.ones(config.company_count)
        self.company_capital = np.zeros(config.company_count)
        self.company_inflation = np.zeros(config.company_count)
        self.company_reported_log = np.zeros(config.company_count)
        self.company_reputation = np.full(config.company_count, 0.8)
        self.company_actions = np.full(config.company_count, 2, dtype=np.int64)
        self.company_states = np.zeros(config.company_count, dtype=np.int64)
        self.company_probabilities = np.full((config.company_count, 3), 1 / 3)
        self.company_costs = np.zeros((config.company_count, 3))
        self.company_interest = np.zeros(config.company_count)
        self.company_window_value = self.company_assets.copy()
        self.company_start_interest = np.zeros(config.company_count)
        self.graph = self._build_graph()
        initial_graph = {"nodes": list(self.graph.nodes), "edges": [e.to_dict() for e in self.graph.edges]}
        graph_version = hashlib.sha256(canonical_json_bytes(initial_graph)).hexdigest()
        sources = {f"source:{p.name}": p for p in Path(__file__).parent.glob("*.py")}
        if config.graph_fixture:
            sources["graph_fixture"] = Path(config.graph_fixture)
        self.manifest = build_run_manifest(
            config=config.to_dict(), data_files=sources, random_seed=config.seed,
            code_revision=code_revision, schema_version=config.schema_version,
            graph_version=graph_version if config.graph_enabled else None,
            agent_models={"traders": "contextual_bandit_v1" if config.bandit_enabled else "fixed_rules",
                          "companies": "tabular_q_v1" if config.company_learning_enabled else "fixed_maintain"},
        )
        self.run_id = self.manifest.run_id
        self.trade_history: deque[np.ndarray] = deque(maxlen=20)
        self.trade_dates: deque[datetime] = deque(maxlen=20)
        self.visible_news = np.zeros(config.population_size)
        self.macro_news = self.book_news = 0.0
        self.policy = Stage2Policy(self)
        self.last_stage2_frame: dict[str, Any] = {}

    def _build_graph(self) -> InformationGraph:
        nodes = self.company_ids + self.trader_ids
        if self.config.graph_fixture:
            fixture = json.loads(Path(self.config.graph_fixture).read_text(encoding="utf-8"))
            if set(fixture["nodes"]) != set(nodes):
                raise ValueError("graph fixture nodes must match the company and trader population")
            return InformationGraph(tuple(fixture["nodes"]), tuple(Edge.from_dict(e) for e in fixture["edges"]))
        edges = [Edge(self.company_ids[i % len(self.company_ids)], trader, "disclosure", 1.0, 0.9,
                      i % 3, EPOCH.date()) for i, trader in enumerate(self.trader_ids)]
        if len(self.trader_ids) > 1:
            edges.extend(Edge(trader, self.trader_ids[(i + 1) % len(self.trader_ids)], "interaction",
                              0.8, 0.95, 1, EPOCH.date()) for i, trader in enumerate(self.trader_ids))
        return InformationGraph(nodes, tuple(edges))

    def _public_news_return(self, day: int) -> float:
        return self.book_news

    def _trader_context(self) -> tuple[np.ndarray, np.ndarray]:
        price = float(self.prices[self.next_day_index])
        history = self.prices[max(0, self.next_day_index - 20):self.next_day_index + 1]
        momentum = float(np.log(history[-1] / history[0])) / max(1, history.size - 1)
        volatility = float(np.std(np.diff(np.log(history)))) if history.size > 2 else self.config.fundamental_volatility
        wealth = self.population.wealth(price)
        self.wealth_peak = np.maximum(self.wealth_peak, wealth)
        features = np.column_stack((self.population.subjective_values / price - 1,
                                    np.full(self.population.size, momentum), np.full(self.population.size, volatility),
                                    1 - wealth / np.maximum(self.wealth_peak, 1e-12)))
        return features, ternary_states(features, TRADER_THRESHOLDS)

    def _company_context(self) -> tuple[np.ndarray, np.ndarray]:
        gap = np.maximum(np.exp(self.company_reported_log) / self.company_performance - 1, 0)
        risk = np.minimum(1, self.config.verification_base_rate + self.config.verification_gap_sensitivity * gap)
        features = np.column_stack((self.company_debt / (self.company_assets * self.company_performance),
                                    gap, risk, self.company_cash / self.company_assets))
        return features, ternary_states(features, COMPANY_THRESHOLDS)

    def _company_day(self, day: int, now: datetime, price: float) -> list[Message]:
        cfg = self.config
        real_cost = np.zeros(cfg.company_count)
        disclosure_cost = np.zeros(cfg.company_count)
        if (day - 1) % 60 == 0:
            _, self.company_states = self._company_context()
            self.company_window_value = self.company_assets * self.company_performance * price / cfg.initial_price
            self.company_start_interest = cfg.company_interest_rate * (1 + max(0, 1 - price / cfg.initial_price)) * self.company_debt
            self.company_costs.fill(0)
            self.company_interest.fill(0)
            if cfg.company_learning_enabled:
                self.company_probabilities = self.company_q.probabilities(self.company_groups, self.company_states)
                self.company_actions = self.company_q.choose(self.company_groups, self.company_states, self.company_rng)
            else:
                self.company_actions.fill(2)
                self.company_probabilities[:] = (0, 0, 1)
            real_cost = np.where(self.company_actions == 0, np.minimum(self.company_cash, cfg.real_investment_fraction * self.company_assets), 0)
            disclosure_cost = np.where(self.company_actions == 1, np.minimum(self.company_cash, cfg.disclosure_investment_fraction * self.company_assets), 0)
            self.company_cash -= real_cost + disclosure_cost
            self.company_external_cash += float((real_cost + disclosure_cost).sum())
            self.company_capital += real_cost / self.company_assets
            self.company_inflation += cfg.disclosure_signal_gain * disclosure_cost / self.company_assets
        self.company_performance *= np.exp(cfg.investment_productivity * self.company_capital
                                          + 0.0005 * self.environment_rng.standard_normal(cfg.company_count))
        self.company_capital *= cfg.capital_retention
        revenue = cfg.company_operating_yield * self.company_assets * self.company_performance
        interest = cfg.company_interest_rate * (1 + max(0, 1 - price / cfg.initial_price)) * self.company_debt
        interest = np.minimum(interest, self.company_cash + revenue)
        self.company_cash += revenue - interest
        self.company_external_cash += float((interest - revenue).sum())
        self.company_interest += interest
        gap = np.maximum(np.exp(self.company_reported_log) / self.company_performance - 1, 0)
        risk = np.minimum(1, cfg.verification_base_rate + cfg.verification_gap_sensitivity * gap)
        verified = self.environment_rng.random(cfg.company_count) < risk
        penalty = np.where(verified, np.minimum(self.company_cash, cfg.verification_penalty_rate * gap * self.company_assets), 0)
        self.company_cash -= penalty
        self.company_external_cash += float(penalty.sum())
        self.company_inflation[verified] = 0
        self.company_reputation[verified] = np.clip(0.9 * self.company_reputation[verified] + 0.1 * (1 - np.minimum(gap[verified], 1)), 0.1, 1)
        self.company_costs += np.column_stack((real_cost, disclosure_cost, penalty))
        messages = []
        for index in range(cfg.company_count):
            if (day - 1) % 60 != 0 and not verified[index]:
                continue
            disclosed_log = float(np.log(self.company_performance[index]) + self.company_inflation[index])
            delta = disclosed_log - self.company_reported_log[index]
            self.company_reported_log[index] = disclosed_log
            messages.append(Message(
                f"{self.company_ids[index]}-{day}", self.company_ids[index], ("public",), now, now,
                "verification" if verified[index] else "disclosure",
                {"log_change": float(delta * self.company_weights[index]), "verified": bool(verified[index])},
                float(self.company_reputation[index]),
            ))
        return messages

    def step(self, day_index: int):
        if day_index != self.next_day_index or day_index >= self.config.trading_days:
            raise RuntimeError("Stage 2 days must be settled exactly once in sequence")
        day, now = day_index + 1, EPOCH + timedelta(days=day_index)
        price_before = float(self.prices[day_index])
        wealth_before = self.population.wealth(price_before)
        self.trader_features, current_states = self._trader_context()
        if (day - 1) % 20 == 0:
            self.trader_states = current_states.copy()
            self.window_start_wealth = wealth_before.copy()
            self.window_peak = wealth_before.copy()
            self.window_drawdown.fill(0)
            self.window_fees.fill(0)
            self.window_returns = []
            if self.config.bandit_enabled:
                self.trader_probabilities = self.bandit.probabilities(self.trader_types, self.trader_states)
                self.trader_actions = self.bandit.choose(self.trader_types, self.trader_states, self.bandit_rng)
        messages = self._company_day(day, now, price_before)
        self.macro_news = float(np.log(float(self.macro_fundamentals[day]) / float(self.macro_fundamentals[day - 1])))
        self.book_news = self.macro_news + sum(float(m.payload["log_change"]) for m in messages)
        self.fundamentals[day] = self.macro_fundamentals[day] * float(np.exp(np.dot(self.company_weights, np.log(self.company_performance))))
        self.visible_news = np.full(self.population.size, self.macro_news)
        if self.config.graph_enabled:
            for message in messages:
                self.graph.publish(message)
            arrivals = self.graph.advance(now)
            for delivery in arrivals:
                if delivery.target_id.startswith("trader-"):
                    index = int(delivery.target_id.removeprefix("trader-"))
                    self.visible_news[index] += float(delivery.payload["log_change"]) * delivery.credibility
            arrival_records = [delivery.to_dict() for delivery in arrivals]
        else:
            self.graph.advance(now)
            for message in messages:
                self.visible_news += float(message.payload["log_change"]) * message.credibility
            arrival_records = []
        self.visible_news.flags.writeable = False
        audit = super().step(day_index)
        wealth = self.population.wealth(audit.mid_price_after)
        fees = self.last_settlement.transaction_costs
        self.window_fees += fees
        self.window_returns.append((wealth + fees - wealth_before) / np.where(wealth_before > 1e-12, wealth_before, 1))
        self.window_peak = np.maximum(self.window_peak, wealth)
        self.window_drawdown = np.maximum(self.window_drawdown, 1 - wealth / np.maximum(self.window_peak, 1e-12))
        self.wealth_peak = np.maximum(self.wealth_peak, wealth)
        trader_rewards = None
        company_rewards = None
        if day % 20 == 0 and self.config.bandit_enabled:
            denominator = np.where(self.window_start_wealth > 1e-12, self.window_start_wealth, 1)
            gross = (wealth + self.window_fees - self.window_start_wealth) / denominator
            cost = self.window_fees / denominator
            volatility_penalty = self.config.risk_penalty * self.population.risk_aversion * np.std(self.window_returns, axis=0)
            drawdown_penalty = self.config.drawdown_penalty * self.window_drawdown
            rewards = gross - cost - volatility_penalty - drawdown_penalty
            self.bandit.update(np.arange(self.population.size), self.trader_types, self.trader_states, self.trader_actions, rewards)
            trader_rewards = {"gross_return": gross.tolist(), "transaction_cost": (-cost).tolist(),
                              "volatility_penalty": (-volatility_penalty).tolist(), "drawdown_penalty": (-drawdown_penalty).tolist(),
                              "reward": rewards.tolist(), "start_day": day - 19, "end_day": day}
            self.trader_reward_windows += 1
        if day % 60 == 0 and self.config.company_learning_enabled:
            value = self.company_assets * self.company_performance * audit.mid_price_after / self.config.initial_price
            components = np.column_stack(((value - self.company_window_value) / self.company_assets,
                                          (self.company_start_interest * 60 - self.company_interest) / self.company_assets,
                                          -self.company_costs / self.company_assets[:, None]))
            _, next_states = self._company_context()
            self.company_q.update(np.arange(self.config.company_count), self.company_groups, self.company_states,
                                  self.company_actions, components.sum(axis=1), next_states)
            company_rewards = {"names": ["company_value_change", "financing_cost_change", "real_investment_cost",
                                         "disclosure_cost", "verification_penalty"], "components": components.tolist(),
                               "reward": components.sum(axis=1).tolist(), "start_day": day - 59, "end_day": day}
            self.company_reward_windows += 1
        self.trade_history.append(self.last_settlement.executed_orders.copy())
        self.trade_dates.append(now)
        graph_updates = ()
        if self.config.graph_enabled and self.config.dynamic_graph_enabled and day % 20 == 0:
            graph_updates = self.graph.adapt_from_trades(self.trader_ids, np.array(self.trade_history),
                                                        tuple(self.trade_dates), now + timedelta(days=1),
                                                        self.config.graph_adaptation_rate)
        company_cash_error = float(self.company_cash.sum()) + self.company_external_cash - self.initial_company_cash
        if abs(company_cash_error) > max(1e-6, self.initial_company_cash * 1e-10) or np.any(self.company_cash < -1e-8):
            raise RuntimeError("company cash ledger failed")
        company_features, company_current_states = self._company_context()
        frame = {
            "day": day, "trader_ids": list(self.trader_ids), "company_ids": list(self.company_ids),
            "trader_types": self.trader_types.tolist(), "factor_preferences": self.factor_preferences.tolist(),
            "risk_aversion": self.population.risk_aversion.tolist(),
            "trader_features": self.trader_features.tolist(), "trader_states": self.trader_states.tolist(),
            "trader_actions": self.trader_actions.tolist() if self.config.bandit_enabled else [],
            "trader_probabilities": self.trader_probabilities.tolist() if self.config.bandit_enabled else [],
            "subjective_values": self.population.subjective_values.tolist(), "visible_news": self.visible_news.tolist(),
            "wealth": wealth.tolist(), "positions": self.population.positions.tolist(),
            "submitted_orders": self.policy.submitted_orders.tolist(), "executed_orders": self.last_settlement.executed_orders.tolist(),
            "transaction_costs": fees.tolist(), "trader_rewards": trader_rewards,
            "company_groups": self.company_groups.tolist(), "company_features": company_features.tolist(),
            "company_states": self.company_states.tolist(), "company_current_states": company_current_states.tolist(),
            "company_actions": self.company_actions.tolist(), "company_cash": self.company_cash.tolist(),
            "company_true_performance": self.company_performance.tolist(), "company_reported_log": self.company_reported_log.tolist(),
            "company_probabilities": self.company_probabilities.tolist(), "company_rewards": company_rewards,
            "bandit_values": self.bandit.values.tolist(), "q_values": self.company_q.values.tolist(),
            "nodes": list(self.graph.nodes) if self.config.graph_enabled else [],
            "edges": [e.to_dict() for e in self.graph.active_edges(now)] if self.config.graph_enabled else [],
            "messages": [m.to_dict() for m in messages], "arrivals": arrival_records, "graph_updates": list(graph_updates),
            "company_cash_error": company_cash_error,
        }
        self.last_stage2_frame = frame
        original = self.telemetry_events[-1]
        payload = dict(original.payload)
        payload["stage2_json"] = json.dumps(frame, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["wealth_median"] = float(np.median(wealth))
        payload["company_cash_error"] = company_cash_error
        for index, name in enumerate(TRADER_ACTIONS):
            payload[f"bandit_{name}_share"] = float(np.mean(self.trader_actions == index)) if self.config.bandit_enabled else 0.0
        event = replace(original, payload=payload)
        self.telemetry_events[-1] = event
        if self.event_log is not None:
            with self.event_log.open("a", encoding="utf-8") as log:
                log.write(json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
        if self.display_sink is not None:
            try:
                self.display_sink.publish(event)
            except (BrokenPipeError, EOFError, OSError, ValueError):
                self.telemetry_publish_failures += 1
        return audit

    def run(self) -> Stage2Result:
        market = super().run()
        audit = {"trader_reward_windows": self.trader_reward_windows, "company_reward_windows": self.company_reward_windows,
                 "pending_trader_days": self.config.trading_days % 20, "pending_company_days": self.config.trading_days % 60,
                 "trader_actions": list(TRADER_ACTIONS), "company_actions": list(COMPANY_ACTIONS),
                 "trader_thresholds": TRADER_THRESHOLDS, "company_thresholds": COMPANY_THRESHOLDS,
                 "bandit_values": self.bandit.values.tolist(), "bandit_counts": self.bandit.counts.tolist(),
                 "q_values": self.company_q.values.tolist(), "q_counts": self.company_q.counts.tolist(),
                 "factor_preferences": self.factor_preferences.tolist(), "trader_types": self.trader_types.tolist(),
                 "company_size_groups": self.company_groups.tolist(), "company_external_cash": self.company_external_cash,
                 "initial_company_cash": self.initial_company_cash, "final_company_cash": self.company_cash.tolist(),
                 "llm_api_calls": 0, "llm_api_tokens": 0}
        return Stage2Result(**{f.name: getattr(market, f.name) for f in fields(SimulationResult)},
                            stage2_manifest=self.manifest, stage2_audit=audit)


def create_market_harness(config: Stage1Config, telemetry_sink: TelemetrySink | None = None,
                          *, event_log: str | Path | None = None, code_revision: str = "working-tree") -> MarketHarness:
    if isinstance(config, Stage2Config):
        if config.active:
            return Stage2Harness(config, telemetry_sink, event_log=event_log, code_revision=code_revision)
        config = config.base_config()
    return MarketHarness(config, telemetry_sink=telemetry_sink)
