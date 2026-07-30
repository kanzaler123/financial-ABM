from dataclasses import replace
from pathlib import Path

import numpy as np

from abm.config import load_stage1_config
from abm.policies import MarketObservation, RuleBasedPolicy
from abm.population import (
    NOISE_STRATEGY,
    TREND_STRATEGY,
    VALUE_STRATEGY,
    TraderPopulation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_three_rule_policies_create_distinct_order_distributions() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=300,
        strategy_shares={"value": 1 / 3, "trend": 1 / 3, "noise": 1 / 3},
    )
    seed_sequence = np.random.SeedSequence(config.seed)
    population_seed, policy_seed = seed_sequence.spawn(2)
    population = TraderPopulation.initialize(
        config, np.random.default_rng(population_seed)
    )
    observation = MarketObservation(
        price=100.0,
        fundamental_value=105.0,
        price_history=np.array([95.0, 100.0]),
    )

    orders = RuleBasedPolicy(config).act(
        observation,
        population,
        np.random.default_rng(policy_seed),
    )

    value_orders = orders[population.strategies == VALUE_STRATEGY]
    trend_orders = orders[population.strategies == TREND_STRATEGY]
    noise_orders = orders[population.strategies == NOISE_STRATEGY]
    assert np.all(value_orders[value_orders != 0] > 0)
    assert 0 < np.count_nonzero(orders) < config.population_size
    assert trend_orders.mean() > 0
    assert trend_orders.std() > 0
    assert not np.isclose(value_orders.mean(), trend_orders.mean())
    assert noise_orders.std() > 0
    assert np.any(noise_orders < 0) and np.any(noise_orders > 0)
