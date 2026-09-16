"""Deterministic daily harness for the Stage 1 artificial market."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import STRATEGY_NAMES, Stage1Config
from .learning import LearningAudit, LogitImitator, PerformanceWindow
from .manifest import build_run_manifest, canonical_json_bytes
from .microstructure import QuasiOrderBook
from .policies import MarketObservation, RuleBasedPolicy
from .population import TraderPopulation
from .settlement import SettlementEngine
from .telemetry import TelemetryEvent, TelemetrySink, write_telemetry

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DailyAudit:
    day: int
    settlement_id: str
    price_before: float
    mid_price_after: float
    execution_price: float
    bid_price: float
    ask_price: float
    spread: float
    depth: float
    order_flow_imbalance: float
    order_flow_surprise: float
    fundamental_value: float
    public_news_return: float
    public_news_impact: float
    value_information_updates: int
    mean_subjective_value: float
    mean_absolute_pending_information: float
    demand_log_return: float
    permanent_impact: float
    transient_impact: float
    transient_impact_change: float
    price_cap_hit: bool
    effective_liquidity: float
    submitted_order_count: int
    submitted_net_demand: float
    executed_net_demand: float
    volume: float
    transaction_cost: float
    total_cash: float
    total_shares: float
    cash_error: float
    share_error: float
    minimum_agent_cash: float
    minimum_agent_position: float
    market_maker_cash: float
    market_maker_inventory: float
    strategy_counts: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            "settlement_id": self.settlement_id,
            "price_before": self.price_before,
            "mid_price_after": self.mid_price_after,
            "execution_price": self.execution_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "spread": self.spread,
            "depth": self.depth,
            "order_flow_imbalance": self.order_flow_imbalance,
            "order_flow_surprise": self.order_flow_surprise,
            "fundamental_value": self.fundamental_value,
            "public_news_return": self.public_news_return,
            "public_news_impact": self.public_news_impact,
            "value_information_updates": self.value_information_updates,
            "mean_subjective_value": self.mean_subjective_value,
            "mean_absolute_pending_information": (
                self.mean_absolute_pending_information
            ),
            "demand_log_return": self.demand_log_return,
            "permanent_impact": self.permanent_impact,
            "transient_impact": self.transient_impact,
            "transient_impact_change": self.transient_impact_change,
            "price_cap_hit": self.price_cap_hit,
            "effective_liquidity": self.effective_liquidity,
            "submitted_order_count": self.submitted_order_count,
            "submitted_net_demand": self.submitted_net_demand,
            "executed_net_demand": self.executed_net_demand,
            "volume": self.volume,
            "transaction_cost": self.transaction_cost,
            "total_cash": self.total_cash,
            "total_shares": self.total_shares,
            "cash_error": self.cash_error,
            "share_error": self.share_error,
            "minimum_agent_cash": self.minimum_agent_cash,
            "minimum_agent_position": self.minimum_agent_position,
            "market_maker_cash": self.market_maker_cash,
            "market_maker_inventory": self.market_maker_inventory,
            "strategy_counts": dict(
                zip(STRATEGY_NAMES, self.strategy_counts, strict=True)
            ),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    burn_in_days: int
    price_formation: str
    prices: FloatArray
    execution_prices: FloatArray
    spreads: FloatArray
    depths: FloatArray
    order_flow_imbalance: FloatArray
    order_flow_surprise: FloatArray
    public_news_impacts: FloatArray
    permanent_impacts: FloatArray
    transient_impacts: FloatArray
    fundamentals: FloatArray
    volumes: FloatArray
    submitted_net_demand: FloatArray
    executed_net_demand: FloatArray
    total_cash: FloatArray
    total_shares: FloatArray
    strategy_counts: IntArray
    daily_audits: tuple[DailyAudit, ...]
    learning_audits: tuple[LearningAudit, ...]
    telemetry_events: tuple[TelemetryEvent, ...]
    final_agent_cash: FloatArray
    final_agent_positions: FloatArray
    final_strategies: NDArray[np.int8]
    final_market_maker_cash: float
    final_market_maker_inventory: float

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"{self.burn_in_days}:{self.price_formation}".encode("utf-8")
        )
        for array in (
            self.prices,
            self.execution_prices,
            self.spreads,
            self.depths,
            self.order_flow_imbalance,
            self.order_flow_surprise,
            self.public_news_impacts,
            self.permanent_impacts,
            self.transient_impacts,
            self.fundamentals,
            self.volumes,
            self.submitted_net_demand,
            self.executed_net_demand,
            self.total_cash,
            self.total_shares,
            self.strategy_counts,
            self.final_agent_cash,
            self.final_agent_positions,
            self.final_strategies,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(
            json.dumps(
                [audit.to_dict() for audit in self.daily_audits],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(
                [audit.to_dict() for audit in self.learning_audits],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        minimum_agent_cash = min(
            audit.minimum_agent_cash for audit in self.daily_audits
        )
        minimum_agent_position = min(
            audit.minimum_agent_position for audit in self.daily_audits
        )
        return {
            "trading_days": len(self.daily_audits),
            "burn_in_days": self.burn_in_days,
            "evaluation_days": (
                len(self.daily_audits) - self.burn_in_days
            ),
            "price_formation": self.price_formation,
            "population_size": int(self.final_agent_cash.size),
            "initial_price": float(self.prices[0]),
            "final_price": float(self.prices[-1]),
            "maximum_absolute_cash_error": float(np.max(np.abs(self.total_cash - self.total_cash[0]))),
            "maximum_absolute_share_error": float(
                np.max(np.abs(self.total_shares - self.total_shares[0]))
            ),
            "minimum_agent_cash": minimum_agent_cash,
            "minimum_agent_position": minimum_agent_position,
            "learning_updates": len(self.learning_audits),
            "price_cap_hits": sum(
                int(audit.price_cap_hit) for audit in self.daily_audits
            ),
            "telemetry_events": len(self.telemetry_events),
            "final_strategy_counts": dict(
                zip(
                    STRATEGY_NAMES,
                    (int(value) for value in self.strategy_counts[-1]),
                    strict=True,
                )
            ),
            "fingerprint": self.fingerprint(),
        }

    def write(
        self,
        output_dir: str | Path,
        config: Stage1Config,
        *,
        code_revision: str = "working-tree",
    ) -> Path:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        manifest = build_run_manifest(
            config=config.to_dict(),
            data_files={},
            random_seed=config.seed,
            code_revision=code_revision,
            schema_version=config.schema_version,
        )
        np.savez_compressed(
            destination / "state_arrays.npz",
            prices=self.prices,
            execution_prices=self.execution_prices,
            spreads=self.spreads,
            depths=self.depths,
            order_flow_imbalance=self.order_flow_imbalance,
            order_flow_surprise=self.order_flow_surprise,
            public_news_impacts=self.public_news_impacts,
            permanent_impacts=self.permanent_impacts,
            transient_impacts=self.transient_impacts,
            fundamentals=self.fundamentals,
            volumes=self.volumes,
            submitted_net_demand=self.submitted_net_demand,
            executed_net_demand=self.executed_net_demand,
            total_cash=self.total_cash,
            total_shares=self.total_shares,
            strategy_counts=self.strategy_counts,
            final_agent_cash=self.final_agent_cash,
            final_agent_positions=self.final_agent_positions,
            final_strategies=self.final_strategies,
        )
        (destination / "config.json").write_text(
            json.dumps(
                config.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "summary.json").write_text(
            json.dumps(
                self.summary(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "run_manifest.json").write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with (destination / "daily_audit.jsonl").open("w", encoding="utf-8") as log:
            for audit in self.daily_audits:
                log.write(
                    json.dumps(
                        audit.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        (destination / "learning_audit.json").write_text(
            json.dumps(
                [audit.to_dict() for audit in self.learning_audits],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        write_telemetry(self.telemetry_events, destination)
        from .visualization import create_stage1_visualizations

        create_stage1_visualizations(destination)
        return destination


def build_fundamental_path(
    config: Stage1Config,
    rng: np.random.Generator | None = None,
    innovations: FloatArray | None = None,
) -> FloatArray:
    """Apply external announcements before agents act on their effective day."""
    if innovations is not None and innovations.shape != (config.trading_days,):
        raise ValueError(
            "fundamental innovations must contain one value per trading day"
        )
    if innovations is not None and not np.all(np.isfinite(innovations)):
        raise ValueError("fundamental innovations must be finite")
    path = np.full(
        config.trading_days + 1,
        config.initial_fundamental,
        dtype=np.float64,
    )
    announcements = {
        announcement.day: announcement.fundamental_delta
        for announcement in config.announcements
    }
    current = config.initial_fundamental
    unconditional_variance = config.fundamental_volatility**2
    conditional_variance = unconditional_variance
    variance_intercept = unconditional_variance * (
        1.0 - config.fundamental_arch - config.fundamental_garch
    )
    previous_shock = 0.0
    if rng is None:
        rng = np.random.default_rng(config.seed)
    for day in range(1, config.trading_days + 1):
        if config.fundamental_volatility > 0:
            if config.fundamental_process == "garch_t":
                conditional_variance = (
                    variance_intercept
                    + config.fundamental_arch * previous_shock**2
                    + config.fundamental_garch * conditional_variance
                )
            else:
                conditional_variance = unconditional_variance
            if innovations is not None:
                innovation = float(innovations[day - 1])
            elif config.fundamental_process in ("student_t", "garch_t"):
                innovation = float(
                    rng.standard_t(config.fundamental_shock_df)
                    / np.sqrt(
                        config.fundamental_shock_df
                        / (config.fundamental_shock_df - 2.0)
                    )
                )
            else:
                innovation = float(rng.standard_normal())
            shock = float(np.sqrt(conditional_variance) * innovation)
            current *= float(
                np.exp(
                    config.fundamental_drift
                    - 0.5 * conditional_variance
                    + shock
                )
            )
            previous_shock = shock
        else:
            # Zero volatility removes randomness, not the configured drift.
            current *= float(np.exp(config.fundamental_drift))
        current += announcements.get(day, 0.0)
        if current <= 0 or not np.isfinite(current):
            raise ValueError(f"announcement on day {day} creates an invalid fundamental")
        path[day] = current
    return path


class MarketHarness:
    """Owns the market clock and is the only component allowed to mutate state."""

    def __init__(
        self,
        config: Stage1Config,
        telemetry_sink: TelemetrySink | None = None,
        fundamental_innovations: FloatArray | None = None,
    ):
        self.config = config
        self.telemetry_sink = telemetry_sink
        self.telemetry_publish_failures = 0
        self.run_id = hashlib.sha256(
            canonical_json_bytes(config.to_dict())
        ).hexdigest()
        seed_sequence = np.random.SeedSequence(config.seed)
        (
            population_seed,
            noise_seed,
            learning_seed,
            fundamental_seed,
        ) = seed_sequence.spawn(4)
        self.population = TraderPopulation.initialize(
            config, np.random.default_rng(population_seed)
        )
        self.noise_rng = np.random.default_rng(noise_seed)
        self.learning_rng = np.random.default_rng(learning_seed)
        self.policy = RuleBasedPolicy(config)
        self.order_book = QuasiOrderBook.initialize(config)
        self.settlement = SettlementEngine(config.transaction_cost_rate)
        self.performance = PerformanceWindow.create(config.population_size)
        self.imitator = LogitImitator(
            temperature=config.learning_temperature,
            risk_penalty=config.risk_penalty,
            update_fraction=config.learning_update_fraction,
        )
        self.market_maker_cash = config.market_maker_cash
        self.market_maker_inventory = config.market_maker_inventory
        self.initial_total_cash = (
            float(self.population.cash.sum()) + self.market_maker_cash
        )
        self.initial_total_shares = (
            float(self.population.positions.sum()) + self.market_maker_inventory
        )
        self.fundamentals = build_fundamental_path(
            config,
            np.random.default_rng(fundamental_seed),
            innovations=fundamental_innovations,
        )
        self.prices = np.empty(config.trading_days + 1, dtype=np.float64)
        self.prices[0] = config.initial_price
        self.volatility_ewma = config.fundamental_volatility**2
        self.volatility_reference_ewma = config.fundamental_volatility**2
        self.execution_prices = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.spreads = np.empty(config.trading_days, dtype=np.float64)
        self.depths = np.empty(config.trading_days, dtype=np.float64)
        self.order_flow_imbalance = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.order_flow_surprise = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.public_news_impacts = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.permanent_impacts = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.transient_impacts = np.empty(
            config.trading_days, dtype=np.float64
        )
        self.volumes = np.empty(config.trading_days, dtype=np.float64)
        self.submitted_net_demand = np.empty(config.trading_days, dtype=np.float64)
        self.executed_net_demand = np.empty(config.trading_days, dtype=np.float64)
        self.total_cash = np.empty(config.trading_days + 1, dtype=np.float64)
        self.total_shares = np.empty(config.trading_days + 1, dtype=np.float64)
        self.strategy_count_history = np.empty(
            (config.trading_days + 1, len(STRATEGY_NAMES)), dtype=np.int64
        )
        self.total_cash[0] = self.initial_total_cash
        self.total_shares[0] = self.initial_total_shares
        self.strategy_count_history[0] = self.population.counts()
        self.daily_audits: list[DailyAudit] = []
        self.learning_audits: list[LearningAudit] = []
        self.telemetry_events: list[TelemetryEvent] = []
        self.next_day_index = 0
        self._settlement_ids: set[str] = set()

    def step(self, day_index: int) -> DailyAudit:
        if day_index != self.next_day_index:
            raise RuntimeError(
                f"expected day_index {self.next_day_index}, got {day_index}; "
                "a day cannot be skipped or settled twice"
            )
        if day_index >= self.config.trading_days:
            raise RuntimeError("simulation is already complete")

        day = day_index + 1
        price_before = float(self.prices[day_index])
        fundamental_value = float(self.fundamentals[day])
        fundamental_before = float(self.fundamentals[day - 1])
        public_news_return = float(
            np.log(fundamental_value / fundamental_before)
        )
        public_news_impact = (
            self.config.public_news_price_pass_through * public_news_return
        )
        performance_wealth_before = self.population.wealth(price_before)
        trade_wealth_before = performance_wealth_before
        if day_index > 0:
            previous_log_return = float(
                np.log(
                    self.prices[day_index] / self.prices[day_index - 1]
                )
            )
            self.volatility_ewma = (
                self.config.volatility_ewma_decay * self.volatility_ewma
                + (1.0 - self.config.volatility_ewma_decay)
                * previous_log_return**2
            )
            self.volatility_reference_ewma = (
                self.config.volatility_reference_decay
                * self.volatility_reference_ewma
                + (1.0 - self.config.volatility_reference_decay)
                * previous_log_return**2
            )
        realized_volatility = float(np.sqrt(self.volatility_ewma))
        realized_volatility_reference = float(
            np.sqrt(self.volatility_reference_ewma)
        )
        observation = MarketObservation(
            price=price_before,
            fundamental_value=fundamental_value,
            price_history=self.prices[: day_index + 1].copy(),
            public_news_return=public_news_return,
            realized_volatility=realized_volatility,
            realized_volatility_reference=realized_volatility_reference,
            spread=(
                float(self.spreads[day_index - 1])
                if day_index > 0
                else 0.0
            ),
            depth=self.order_book.depth,
        )
        submitted = self.policy.act(
            observation, self.population, self.noise_rng
        )
        submitted_net = float(submitted.sum())
        quote = self.order_book.quote(
            submitted_orders=submitted,
            public_news_impact=public_news_impact,
            realized_volatility=realized_volatility,
            realized_volatility_reference=realized_volatility_reference,
        )
        execution_price = quote.execution_price

        # The margin floor is computed against a worst-case price that also
        # covers plausible news moves, so a forced cover stays affordable
        # even when a news jump pushes execution above the price-cap bound.
        margin_floor_price = price_before * float(
            np.exp(
                self.config.max_log_return
                + 4.0 * self.config.fundamental_volatility
            )
        )
        settlement = self.settlement.settle(
            population=self.population,
            submitted_orders=submitted,
            execution_price=execution_price,
            market_maker_cash=self.market_maker_cash,
            market_maker_inventory=self.market_maker_inventory,
            minimum_positions=np.maximum(
                -self.config.max_short_leverage
                * trade_wealth_before
                / margin_floor_price,
                -self.config.position_cap,
            ),
        )
        self.market_maker_cash = settlement.market_maker_cash
        self.market_maker_inventory = settlement.market_maker_inventory
        wealth_after = self.population.wealth(quote.mid_price_after)
        self.performance.record(
            wealth_before=performance_wealth_before,
            wealth_after=wealth_after,
            transaction_costs=settlement.transaction_costs,
        )

        self.prices[day] = quote.mid_price_after
        self.execution_prices[day_index] = execution_price
        self.spreads[day_index] = quote.spread
        self.depths[day_index] = quote.depth
        self.order_flow_imbalance[day_index] = (
            quote.order_flow_imbalance
        )
        self.order_flow_surprise[day_index] = quote.order_flow_surprise
        self.public_news_impacts[day_index] = (
            quote.public_news_impact
        )
        self.permanent_impacts[day_index] = quote.permanent_impact
        self.transient_impacts[day_index] = quote.transient_impact
        self.volumes[day_index] = float(np.abs(settlement.executed_orders).sum())
        self.submitted_net_demand[day_index] = submitted_net
        self.executed_net_demand[day_index] = float(
            settlement.executed_orders.sum()
        )
        total_cash = float(self.population.cash.sum()) + self.market_maker_cash
        total_shares = (
            float(self.population.positions.sum()) + self.market_maker_inventory
        )
        self.total_cash[day] = total_cash
        self.total_shares[day] = total_shares

        self.population.validate(
            allow_short=self.config.max_short_leverage > 0,
            price=quote.mid_price_after,
        )
        if self.market_maker_cash < -1e-7:
            raise RuntimeError("market maker has negative cash")
        cash_error = total_cash - self.initial_total_cash
        share_error = total_shares - self.initial_total_shares
        cash_tolerance = max(1e-6, abs(self.initial_total_cash) * 1e-12)
        share_tolerance = max(1e-8, abs(self.initial_total_shares) * 1e-8)
        if abs(cash_error) > cash_tolerance:
            raise RuntimeError(f"cash conservation failed on day {day}: {cash_error}")
        if abs(share_error) > share_tolerance:
            raise RuntimeError(f"share conservation failed on day {day}: {share_error}")

        settlement_id = hashlib.sha256(
            day.to_bytes(8, "little")
            + np.ascontiguousarray(settlement.executed_orders).tobytes()
            + np.float64(execution_price).tobytes()
        ).hexdigest()
        if settlement_id in self._settlement_ids:
            raise RuntimeError(f"duplicate settlement detected on day {day}")
        self._settlement_ids.add(settlement_id)

        audit = DailyAudit(
            day=day,
            settlement_id=settlement_id,
            price_before=price_before,
            mid_price_after=quote.mid_price_after,
            execution_price=execution_price,
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            spread=quote.spread,
            depth=quote.depth,
            order_flow_imbalance=quote.order_flow_imbalance,
            order_flow_surprise=quote.order_flow_surprise,
            fundamental_value=fundamental_value,
            public_news_return=public_news_return,
            public_news_impact=quote.public_news_impact,
            value_information_updates=(
                self.policy.diagnostics.value_information_updates
            ),
            mean_subjective_value=(
                self.policy.diagnostics.mean_subjective_value
            ),
            mean_absolute_pending_information=(
                self.policy.diagnostics.mean_absolute_pending_information
            ),
            demand_log_return=quote.demand_log_return,
            permanent_impact=quote.permanent_impact,
            transient_impact=quote.transient_impact,
            transient_impact_change=quote.transient_impact_change,
            price_cap_hit=quote.price_cap_hit,
            effective_liquidity=quote.depth,
            submitted_order_count=int(np.count_nonzero(submitted)),
            submitted_net_demand=submitted_net,
            executed_net_demand=float(settlement.executed_orders.sum()),
            volume=float(np.abs(settlement.executed_orders).sum()),
            transaction_cost=float(settlement.transaction_costs.sum()),
            total_cash=total_cash,
            total_shares=total_shares,
            cash_error=cash_error,
            share_error=share_error,
            minimum_agent_cash=float(self.population.cash.min()),
            minimum_agent_position=float(self.population.positions.min()),
            market_maker_cash=self.market_maker_cash,
            market_maker_inventory=self.market_maker_inventory,
            strategy_counts=tuple(int(value) for value in self.population.counts()),
        )
        self.daily_audits.append(audit)

        if self.config.learning_enabled and day % self.config.learning_interval == 0:
            self.learning_audits.append(
                self.imitator.update(
                    day=day,
                    population=self.population,
                    performance=self.performance,
                    rng=self.learning_rng,
                )
            )
        elif day % self.config.learning_interval == 0:
            self.performance.reset()

        self.strategy_count_history[day] = self.population.counts()
        daily_return = quote.mid_price_after / price_before - 1.0
        return_start = max(1, day - 19)
        recent_returns = (
            self.prices[return_start : day + 1]
            / self.prices[return_start - 1 : day]
            - 1.0
        )
        counts_after_learning = self.strategy_count_history[day]
        telemetry_event = TelemetryEvent(
            run_id=self.run_id,
            sim_time=day,
            sequence_no=day,
            event_type="daily_market",
            payload={
                "price": execution_price,
                "mid_price": quote.mid_price_after,
                "execution_price": execution_price,
                "bid_price": quote.bid_price,
                "ask_price": quote.ask_price,
                "spread": quote.spread,
                "depth": quote.depth,
                "order_flow_imbalance": quote.order_flow_imbalance,
                "order_flow_surprise": quote.order_flow_surprise,
                "return": daily_return,
                "volatility_20d": float(np.std(recent_returns, ddof=0)),
                "fundamental_value": fundamental_value,
                "public_news_return": public_news_return,
                "public_news_impact": public_news_impact,
                "value_information_updates": (
                    self.policy.diagnostics.value_information_updates
                ),
                "mean_subjective_value": (
                    self.policy.diagnostics.mean_subjective_value
                ),
                "mean_absolute_pending_information": (
                    self.policy.diagnostics.mean_absolute_pending_information
                ),
                "demand_log_return": quote.demand_log_return,
                "permanent_impact": quote.permanent_impact,
                "transient_impact": quote.transient_impact,
                "transient_impact_change": quote.transient_impact_change,
                "price_cap_hit": quote.price_cap_hit,
                "effective_liquidity": quote.depth,
                "submitted_order_count": int(np.count_nonzero(submitted)),
                "volume": audit.volume,
                "value_share": float(
                    counts_after_learning[0] / self.config.population_size
                ),
                "trend_share": float(
                    counts_after_learning[1] / self.config.population_size
                ),
                "noise_share": float(
                    counts_after_learning[2] / self.config.population_size
                ),
                "cash_relative_error": abs(cash_error) / self.initial_total_cash,
                "share_relative_error": abs(share_error)
                / self.initial_total_shares,
            },
        )
        self.telemetry_events.append(telemetry_event)
        if self.telemetry_sink is not None:
            try:
                self.telemetry_sink.publish(telemetry_event)
            except Exception:
                # The visualizer is an optional observer. A detached or faulty
                # observer must never alter or stop the market simulation.
                self.telemetry_publish_failures += 1
        self.next_day_index += 1
        return audit

    def run(self) -> SimulationResult:
        while self.next_day_index < self.config.trading_days:
            self.step(self.next_day_index)
        return SimulationResult(
            burn_in_days=self.config.burn_in_days,
            price_formation=self.config.price_formation,
            prices=self.prices.copy(),
            execution_prices=self.execution_prices.copy(),
            spreads=self.spreads.copy(),
            depths=self.depths.copy(),
            order_flow_imbalance=self.order_flow_imbalance.copy(),
            order_flow_surprise=self.order_flow_surprise.copy(),
            public_news_impacts=(
                self.public_news_impacts.copy()
            ),
            permanent_impacts=self.permanent_impacts.copy(),
            transient_impacts=self.transient_impacts.copy(),
            fundamentals=self.fundamentals.copy(),
            volumes=self.volumes.copy(),
            submitted_net_demand=self.submitted_net_demand.copy(),
            executed_net_demand=self.executed_net_demand.copy(),
            total_cash=self.total_cash.copy(),
            total_shares=self.total_shares.copy(),
            strategy_counts=self.strategy_count_history.copy(),
            daily_audits=tuple(self.daily_audits),
            learning_audits=tuple(self.learning_audits),
            telemetry_events=tuple(self.telemetry_events),
            final_agent_cash=self.population.cash.copy(),
            final_agent_positions=self.population.positions.copy(),
            final_strategies=self.population.strategies.copy(),
            final_market_maker_cash=self.market_maker_cash,
            final_market_maker_inventory=self.market_maker_inventory,
        )


def run_stage1(
    config: Stage1Config,
    telemetry_sink: TelemetrySink | None = None,
    fundamental_innovations: FloatArray | None = None,
) -> SimulationResult:
    return MarketHarness(
        config,
        telemetry_sink=telemetry_sink,
        fundamental_innovations=fundamental_innovations,
    ).run()
