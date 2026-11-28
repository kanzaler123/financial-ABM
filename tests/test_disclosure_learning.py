import hashlib

import numpy as np
import pytest

from abm.disclosure.learning import (
    batch_update, choose_actions, context_states, discrete_candidates, load_frozen,
    npv_discrete, period_discount, realized_rewards, save_frozen,
)
from abm.tabular import CompanyQLearner


def candidates(**changes):
    arguments = dict(improve_capacity=np.array([.25]), inflate_capacity=np.array([.25]),
                     marginal_cost=np.array([.05]), max_change_per_period=.25,
                     wash_cost_ratio=.2, subsidy=0, real_value_per_sd=np.array([.001]),
                     displaced_real_value_per_sd=np.array([.0001]),
                     wash_value_per_sd=np.array([.0001]), fine_value_per_sd=np.array([.01]))
    return discrete_candidates(**{**arguments, **changes})


def transition(learner, *, reward=0, terminated=False, state=0, next_state=0, next_group=0,
               mask=None):
    batch_update(learner, np.array([0]), np.array([0]), np.array([state]), np.array([0]),
                 np.array([reward]), np.array([next_group]), np.array([next_state]),
                 np.ones((1, 3), dtype=bool) if mask is None else mask,
                 elapsed_days=60, annual_discount_rate=.05, terminated=np.array([terminated]))


def test_common_candidates_allow_negative_npv_washing_to_be_explored():
    offered = candidates()
    assert offered.amounts.tolist() == [[.0025, .0025, 0]]
    assert offered.valid_actions.all()
    assert offered.values[0, 1] < 0
    assert npv_discrete(offered).tolist() == [0]
    learner = CompanyQLearner(epsilon=1)
    actions = choose_actions(learner, np.zeros(1000, dtype=int), np.zeros(1000, dtype=int),
                             np.repeat(offered.valid_actions, 1000, axis=0),
                             np.random.default_rng(13), training=True)
    assert np.bincount(actions, minlength=3).min() > 250


def test_capacity_masks_and_real_improvement_when_report_is_full():
    offered = candidates(improve_capacity=np.array([.001, .25, 0]),
                         inflate_capacity=np.array([.0005, 0, 0]))
    np.testing.assert_array_equal(offered.amounts, [[.001, .0005, 0], [.0025, 0, 0], [0, 0, 0]])
    assert offered.valid_actions.tolist() == [[True, True, True], [True, False, True], [False, False, True]]
    learner = CompanyQLearner(epsilon=1)
    learner.values[:, :, :2] = 1e6
    actions = choose_actions(learner, np.zeros(1000, dtype=int), np.zeros(1000, dtype=int),
                             np.repeat(offered.valid_actions[2:], 1000, axis=0),
                             np.random.default_rng(14), training=True)
    assert np.all(actions == 2)
    real = candidates(inflate_capacity=np.array([0]), real_value_per_sd=np.array([.1]),
                      displaced_real_value_per_sd=np.array([.01]))
    assert npv_discrete(real).tolist() == [0]


def test_frozen_ties_and_unvisited_states_maintain_without_random_draws():
    learner = CompanyQLearner(epsilon=1)
    learner.values[0, 0, 0] = 10
    assert choose_actions(learner, np.array([0]), np.array([0]), np.ones((1, 3), dtype=bool),
                          None, training=False).tolist() == [2]
    learner.counts[0, 0] = 1
    learner.values[0, 0] = 0
    assert choose_actions(learner, np.array([0]), np.array([0]), np.ones((1, 3), dtype=bool),
                          None, training=False).tolist() == [2]


def test_contexts_keep_fixed_units_and_known_next_day_audit_liability():
    arguments = dict(financing_exposure=np.array([0, 1, 3]), prices=np.ones(3),
                     inflation=np.array([0, .1, .3]), marginal_cost=np.array([.01, .05, .2]),
                     real_cost_scale=.05, max_change_per_period=.25,
                     pending_due_days=np.array([-1, 120, 30]), pending_cash_amount=np.zeros(3),
                     decision_days=60)
    states = context_states(**arguments)
    assert states.tolist() == [0, 68, 79]
    cash_pending = context_states(**{**arguments, "pending_cash_amount": np.array([-1, 0, 0])})
    assert cash_pending[0] == 1
    np.testing.assert_array_equal(context_states(**{**arguments, "financing_exposure": arguments["financing_exposure"] * 100,
                                                  "prices": arguments["prices"] * 100}), states)


def test_realized_rewards_assign_audit_fine_to_actual_next_day_once():
    savings = np.full((121, 1), .1)
    costs, fines, subsidies = (np.zeros_like(savings) for _ in range(3))
    costs[1], subsidies[1], fines[61] = 2, 1, 7
    cap = np.array([100.])
    first = realized_rewards(savings, costs, fines, subsidies, cap, start_day=0, end_day=60, annual_discount_rate=.05)
    second = realized_rewards(savings, costs, fines, subsidies, cap, start_day=60, end_day=120, annual_discount_rate=.05)
    whole = realized_rewards(savings, costs, fines, subsidies, cap, start_day=0, end_day=120, annual_discount_rate=.05)
    without_fine = realized_rewards(savings, costs, np.zeros_like(fines), subsidies, cap, start_day=0, end_day=60, annual_discount_rate=.05)
    np.testing.assert_array_equal(first, without_fine)
    np.testing.assert_allclose(first + period_discount(.05, 60) * second, whole, atol=1e-15)
    second_without_fine = realized_rewards(savings, costs, np.zeros_like(fines), subsidies, cap, start_day=60, end_day=120, annual_discount_rate=.05)
    np.testing.assert_allclose(second - second_without_fine, -7 * np.exp(-.05 / 252) / 100)


def test_batch_update_distinguishes_truncation_from_absorption_and_masks_next_group():
    learner = CompanyQLearner(groups=2, learning_rate=1)
    learner.values[1, 1] = [3, 1000, 2]
    mask = np.array([[True, False, True]])
    transition(learner, reward=-1, state=0, next_state=1, next_group=1, mask=mask)
    assert learner.values[0, 0, 0] == pytest.approx(-1 + np.exp(-.05 * 60 / 252) * 3)
    transition(learner, reward=-1, terminated=True, state=2, next_state=1, next_group=1, mask=mask)
    assert learner.values[0, 2, 0] == -1


def test_batch_targets_use_one_snapshot_and_ignore_row_order_or_company_numbering():
    first, second = CompanyQLearner(learning_rate=.5), CompanyQLearner(learning_rate=.5)
    first.values[0, 1, 0] = second.values[0, 1, 0] = 4
    groups = np.zeros(3, dtype=int)
    states = np.ones(3, dtype=int)
    actions = np.zeros(3, dtype=int)
    rewards = np.array([1e20, -1e20, 1.])
    for learner, order, ids in ((first, np.arange(3), np.array([7, 8, 9])),
                                (second, np.array([2, 0, 1]), np.array([900, 20, 30]))):
        batch_update(learner, ids, groups[order], states[order], actions[order], rewards[order],
                     groups[order], states[order], np.ones((3, 3), dtype=bool), elapsed_days=60,
                     annual_discount_rate=.05, terminated=np.ones(3, dtype=bool))
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.counts, second.counts)
    assert first.values[0, 1, 0] == pytest.approx(2 + 1 / 6)
    assert first.counts[0, 1, 0] == 3


def test_next_state_target_is_not_changed_by_an_earlier_key_update():
    learner = CompanyQLearner(learning_rate=1)
    learner.values[0, 0, 0] = 4
    batch_update(learner, np.array([1, 2]), np.array([0, 0]), np.array([1, 0]), np.array([0, 0]),
                 np.array([0., -4.]), np.array([0, 0]), np.array([0, 0]), np.ones((2, 3), dtype=bool),
                 elapsed_days=60, annual_discount_rate=.05, terminated=np.array([False, True]))
    assert learner.values[0, 0, 0] == -4
    assert learner.values[0, 1, 0] == pytest.approx(4 * np.exp(-.05 * 60 / 252))


def train_delayed_environment(delay):
    learner = CompanyQLearner(groups=1, epsilon=.3, learning_rate=.2,
                              discount=float(period_discount(.05, 60)))
    rng = np.random.default_rng(150 + delay)
    valid = np.ones((1, 3), dtype=bool)
    initial_real_reward = None
    for _ in range(700):
        queue = [0] * (delay - 1)
        pending_fine = 0
        for _ in range(10):
            state = sum(flag << index for index, flag in enumerate(queue)) + (pending_fine << len(queue))
            action = int(choose_actions(learner, np.array([0]), np.array([state]), valid, rng, training=True)[0])
            savings = np.zeros((61, 1))
            cost, fine, subsidy = (np.zeros_like(savings) for _ in range(3))
            savings[60] = .18 * queue.pop(0)
            cost[1] = .1 if action == 0 else .02 if action == 1 else 0
            fine[1] = .08 * pending_fine
            reward = realized_rewards(savings, cost, fine, subsidy, np.array([1.]),
                                      start_day=0, end_day=60, annual_discount_rate=.05)
            queue.append(int(action == 0))
            pending_fine = int(action == 1)
            next_state = sum(flag << index for index, flag in enumerate(queue)) + (pending_fine << len(queue))
            if state == 0 and action == 0:
                initial_real_reward = float(reward[0])
            batch_update(learner, np.array([0]), np.array([0]), np.array([state]), np.array([action]),
                         reward, np.array([0]), np.array([next_state]), valid, elapsed_days=60,
                         annual_discount_rate=.05, terminated=np.array([False]))
    return learner, initial_real_reward


@pytest.mark.parametrize("delay", [2, 3])
def test_learning_discovers_profitable_delayed_action_with_overlap_and_old_fine(delay):
    learner, immediate_reward = train_delayed_environment(delay)
    assert immediate_reward < 0
    assert learner.values[0, 0, 0] > learner.values[0, 0, 2]
    assert learner.values[0, 0, 1] < learner.values[0, 0, 2]
    queue, pending_fine, selected = [0] * (delay - 1), 0, []
    for _ in range(100):
        state = sum(flag << index for index, flag in enumerate(queue)) + (pending_fine << len(queue))
        action = int(choose_actions(learner, np.array([0]), np.array([state]), np.ones((1, 3), dtype=bool), None, training=False)[0])
        selected.append(action)
        queue.pop(0)
        queue.append(int(action == 0))
        pending_fine = int(action == 1)
    assert selected.count(0) >= 95


def test_frozen_roundtrip_evaluation_is_readonly_and_checks_context(tmp_path):
    learner = CompanyQLearner(epsilon=1)
    learner.values[0, 0] = [1, 2, 0]
    learner.counts[0, 0] = [2, 3, 4]
    path = tmp_path / "learner.npz"
    metadata = {"training_seeds": [25001], "reward_definition": "realized financing ledger", "candidate_fraction": .01}
    sha = save_frozen(path, learner, metadata)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    frozen, restored = load_frozen(path, metadata, expected_sha256=sha)
    assert restored == metadata
    before = frozen.values.copy(), frozen.counts.copy()
    for _ in range(10):
        assert choose_actions(frozen, np.array([0]), np.array([0]), np.ones((1, 3), dtype=bool), None, training=False).tolist() == [1]
    np.testing.assert_array_equal(frozen.values, before[0])
    np.testing.assert_array_equal(frozen.counts, before[1])
    with pytest.raises(ValueError, match="frozen"):
        transition(frozen)
    with pytest.raises(ValueError, match="writable"):
        choose_actions(frozen, np.array([0]), np.array([0]), np.ones((1, 3), dtype=bool), np.random.default_rng(1), training=True)
    with pytest.raises(ValueError, match="metadata"):
        load_frozen(path, {"candidate_fraction": .5})
    with pytest.raises(ValueError, match="hash"):
        load_frozen(path, expected_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        save_frozen(path, learner, metadata)


@pytest.mark.parametrize("changes", [{"amplitude_fraction": 0}, {"wash_cost_ratio": 0},
                                     {"marginal_cost": np.array([np.nan])}])
def test_invalid_candidate_inputs_fail(changes):
    with pytest.raises(ValueError):
        candidates(**changes)
