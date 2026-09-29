"""Shared-type contextual bandits and tabular Q-learning."""

from __future__ import annotations

import numpy as np

TRADER_ACTIONS = ("value", "trend", "contrarian", "reduce")
COMPANY_ACTIONS = ("real_investment", "disclosure_investment", "maintain")
TRADER_THRESHOLDS = ((-0.01, 0.01), (-0.005, 0.005), (0.01, 0.02), (0.02, 0.10))
COMPANY_THRESHOLDS = ((0.2, 0.6), (0.02, 0.10), (0.05, 0.20), (0.1, 0.4))


def ternary_states(features: np.ndarray, thresholds: tuple) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 4 or not np.all(np.isfinite(features)):
        raise ValueError("contexts must contain four finite features")
    bins = np.column_stack([np.digitize(features[:, column], boundaries)
                            for column, boundaries in enumerate(thresholds)])
    return (bins @ np.array([27, 9, 3, 1], dtype=np.int64)).astype(np.int64)


def epsilon_probabilities(values: np.ndarray, epsilon: float) -> np.ndarray:
    best = values == values.max(axis=1, keepdims=True)
    return epsilon / values.shape[1] + (1.0 - epsilon) * best / best.sum(axis=1, keepdims=True)


class ContextualBandit:
    def __init__(self, groups: int = 3, epsilon: float = 0.1):
        if groups < 1 or not 0 <= epsilon <= 1:
            raise ValueError("invalid bandit group count or exploration rate")
        self.epsilon = epsilon
        self.values = np.zeros((groups, 81, 4), dtype=np.float64)
        self.counts = np.zeros((groups, 81, 4), dtype=np.int64)

    def probabilities(self, groups: np.ndarray, states: np.ndarray) -> np.ndarray:
        return epsilon_probabilities(self.values[groups, states], self.epsilon)

    def choose(self, groups: np.ndarray, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probabilities = self.probabilities(groups, states)
        return np.minimum((rng.random(groups.size)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1), 3)

    def update(self, ids: np.ndarray, groups: np.ndarray, states: np.ndarray,
               actions: np.ndarray, rewards: np.ndarray) -> None:
        for index in np.argsort(ids, kind="stable"):
            key = (groups[index], states[index], actions[index])
            self.counts[key] += 1
            self.values[key] += (rewards[index] - self.values[key]) / self.counts[key]


class CompanyQLearner:
    def __init__(self, groups: int = 3, epsilon: float = 0.1,
                 learning_rate: float = 0.1, discount: float = 0.9):
        if groups < 1 or not 0 <= epsilon <= 1 or not 0 < learning_rate <= 1 or not 0 <= discount < 1:
            raise ValueError("invalid Q-learning parameters")
        self.epsilon, self.learning_rate, self.discount = epsilon, learning_rate, discount
        self.values = np.zeros((groups, 81, 3), dtype=np.float64)
        self.counts = np.zeros((groups, 81, 3), dtype=np.int64)

    def probabilities(self, groups: np.ndarray, states: np.ndarray) -> np.ndarray:
        return epsilon_probabilities(self.values[groups, states], self.epsilon)

    def choose(self, groups: np.ndarray, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probabilities = self.probabilities(groups, states)
        return np.minimum((rng.random(groups.size)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1), 2)

    def update(self, ids: np.ndarray, groups: np.ndarray, states: np.ndarray,
               actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray) -> None:
        next_values = self.values[groups, next_states].max(axis=1).copy()
        for index in np.argsort(ids, kind="stable"):
            key = (groups[index], states[index], actions[index])
            target = rewards[index] + self.discount * next_values[index]
            self.values[key] += self.learning_rate * (target - self.values[key])
            self.counts[key] += 1
