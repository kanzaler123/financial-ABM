"""Sixty-day Logit imitation based on auditable strategy fitness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import STRATEGY_NAMES
from .population import STRATEGY_CODES, TraderPopulation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LearningAudit:
    day: int
    mean_fitness_by_strategy: tuple[float, float, float]
    choice_probabilities: tuple[float, float, float]
    counts_before: tuple[int, int, int]
    counts_after: tuple[int, int, int]
    updated_agents: int

    def to_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            # -inf is an internal softmax mask, not a JSON number.  An absent
            # cohort has no observed fitness; null must not be confused with 0.
            "mean_fitness_by_strategy": {
                name: float(value) if np.isfinite(value) else None
                for name, value in zip(
                    STRATEGY_NAMES, self.mean_fitness_by_strategy, strict=True
                )
            },
            "choice_probabilities": dict(
                zip(STRATEGY_NAMES, self.choice_probabilities, strict=True)
            ),
            "counts_before": dict(
                zip(STRATEGY_NAMES, self.counts_before, strict=True)
            ),
            "counts_after": dict(zip(STRATEGY_NAMES, self.counts_after, strict=True)),
            "updated_agents": self.updated_agents,
        }


@dataclass(slots=True)
class PerformanceWindow:
    gross_return_sum: FloatArray
    gross_return_square_sum: FloatArray
    cost_rate_sum: FloatArray
    observations: int = 0

    @classmethod
    def create(cls, population_size: int) -> "PerformanceWindow":
        return cls(
            gross_return_sum=np.zeros(population_size, dtype=np.float64),
            gross_return_square_sum=np.zeros(population_size, dtype=np.float64),
            cost_rate_sum=np.zeros(population_size, dtype=np.float64),
        )

    def record(
        self,
        *,
        wealth_before: FloatArray,
        wealth_after: FloatArray,
        transaction_costs: FloatArray,
    ) -> None:
        denominator = np.where(wealth_before > 1e-12, wealth_before, 1.0)
        gross_return = (
            wealth_after + transaction_costs - wealth_before
        ) / denominator
        cost_rate = transaction_costs / denominator
        self.gross_return_sum += gross_return
        self.gross_return_square_sum += gross_return * gross_return
        self.cost_rate_sum += cost_rate
        self.observations += 1

    def fitness(self, risk_aversion: FloatArray, risk_penalty: float) -> FloatArray:
        if self.observations <= 0:
            raise RuntimeError("cannot compute fitness for an empty window")
        mean = self.gross_return_sum / self.observations
        variance = np.maximum(
            self.gross_return_square_sum / self.observations - mean * mean,
            0.0,
        )
        mean_cost_rate = self.cost_rate_sum / self.observations
        return mean - risk_penalty * risk_aversion * np.sqrt(variance) - mean_cost_rate

    def reset(self) -> None:
        self.gross_return_sum.fill(0.0)
        self.gross_return_square_sum.fill(0.0)
        self.cost_rate_sum.fill(0.0)
        self.observations = 0


@dataclass(frozen=True, slots=True)
class LogitImitator:
    temperature: float
    risk_penalty: float
    update_fraction: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.risk_penalty) or self.risk_penalty < 0:
            raise ValueError("risk_penalty must be finite and nonnegative")
        if not 0 < self.update_fraction <= 1:
            raise ValueError("update_fraction must be in (0, 1]")

    def update(
        self,
        *,
        day: int,
        population: TraderPopulation,
        performance: PerformanceWindow,
        rng: np.random.Generator,
    ) -> LearningAudit:
        individual_fitness = performance.fitness(
            population.risk_aversion, self.risk_penalty
        )
        if not np.all(np.isfinite(individual_fitness)):
            raise ValueError("individual fitness must be finite")
        counts_before_array = population.counts()
        mean_fitness = np.full(len(STRATEGY_CODES), -np.inf, dtype=np.float64)
        for strategy in STRATEGY_CODES:
            mask = population.strategies == strategy
            if np.any(mask):
                mean_fitness[strategy] = float(individual_fitness[mask].mean())

        finite = np.isfinite(mean_fitness)
        if not np.any(finite):
            raise ValueError("learning requires at least one populated strategy")
        # Center BEFORE division: a tiny temperature must not turn the best
        # score into +inf and accidentally remove it from the choice set.
        with np.errstate(over="ignore", under="ignore"):
            logits = (
                mean_fitness[finite] - float(mean_fitness[finite].max())
            ) / self.temperature
            weights = np.zeros_like(mean_fitness)
            weights[finite] = np.exp(logits)
        probabilities = weights / weights.sum()
        update_count = max(
            1, int(round(population.size * self.update_fraction))
        )
        update_indices = rng.choice(
            population.size,
            size=update_count,
            replace=False,
        )
        population.strategies[update_indices] = rng.choice(
            STRATEGY_CODES,
            size=update_count,
            p=probabilities,
        )
        counts_after_array = population.counts()
        performance.reset()
        return LearningAudit(
            day=day,
            mean_fitness_by_strategy=tuple(float(value) for value in mean_fitness),
            choice_probabilities=tuple(float(value) for value in probabilities),
            counts_before=tuple(int(value) for value in counts_before_array),
            counts_after=tuple(int(value) for value in counts_after_array),
            updated_agents=update_count,
        )
