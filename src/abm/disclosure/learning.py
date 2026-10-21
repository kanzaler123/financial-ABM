"""Shared discrete choices and ledger-based tabular learning components."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..tabular import CompanyQLearner, ternary_states
from .config import TRADING_DAYS_PER_YEAR

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
IMPROVE, INFLATE, MAINTAIN = 0, 1, 2
CANDIDATE_FRACTION = .01
ACTION_PRIORITY = np.array([MAINTAIN, IMPROVE, INFLATE])


def action_kinds(action_count: int) -> IntArray:
    if not isinstance(action_count, (int, np.integer)) or action_count < 3 or action_count % 2 != 1:
        raise ValueError("action count must be odd and at least three")
    return np.array([IMPROVE, INFLATE, MAINTAIN] + [IMPROVE, INFLATE] * ((action_count - 3) // 2), dtype=np.int64)


def _action_priority(action_count: int) -> IntArray:
    kinds = action_kinds(action_count)
    return np.concatenate(([MAINTAIN], np.flatnonzero(kinds == IMPROVE), np.flatnonzero(kinds == INFLATE)))


def _vectors(*values: Any) -> list[FloatArray]:
    arrays = np.broadcast_arrays(*(np.atleast_1d(np.asarray(value, dtype=np.float64)) for value in values))
    if any(array.ndim != 1 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("inputs must be finite scalar or one-dimensional arrays")
    return arrays


@dataclass(frozen=True, slots=True)
class CandidateSet:
    amounts: FloatArray
    cost_fraction: FloatArray
    subsidy_fraction: FloatArray
    values: FloatArray
    valid_actions: NDArray[np.bool_]
    kinds: IntArray


def discrete_candidates(
    *, improve_capacity: FloatArray, inflate_capacity: FloatArray, marginal_cost: FloatArray,
    max_change_per_period: float, wash_cost_ratio: float | FloatArray, subsidy: float,
    real_value_per_sd: FloatArray, displaced_real_value_per_sd: FloatArray,
    wash_value_per_sd: FloatArray, fine_value_per_sd: FloatArray,
    amplitude_fraction: float = CANDIDATE_FRACTION,
    amplitude_fractions: tuple[float, ...] | None = None,
    fixed_fine_value: FloatArray | float = 0.0,
) -> CandidateSet:
    """Candidate values; ``fixed_fine_value`` is a lump sum on every positive wash amplitude.

    ``wash_cost_ratio`` is one ratio or one per firm.
    """
    fractions = (amplitude_fraction,) if amplitude_fractions is None else amplitude_fractions
    firm_ratios = np.ndim(wash_cost_ratio) > 0
    if firm_ratios:
        ratios = np.asarray(wash_cost_ratio, dtype=np.float64)
        if ratios.ndim != 1 or ratios.size != np.size(marginal_cost) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
            raise ValueError("firm wash cost ratios must be positive, finite and one per firm")
    if (not all(math.isfinite(value) and value > 0 for value in (max_change_per_period, 1.0 if firm_ratios else wash_cost_ratio))
            or not fractions or any(not math.isfinite(value) or not 0 < value <= 1 for value in fractions)
            or any(left >= right for left, right in zip(fractions, fractions[1:]))):
        raise ValueError("candidate scale and cost ratio must be positive; amplitude fraction must lie in (0,1]")
    real_cap, wash_cap, c, s, real, displaced, wash, fine, fixed = _vectors(
        improve_capacity, inflate_capacity, marginal_cost, subsidy, real_value_per_sd,
        displaced_real_value_per_sd, wash_value_per_sd, fine_value_per_sd, fixed_fine_value,
    )
    if np.any(c <= 0) or any(np.any(value < 0) for value in (s, real, displaced, wash, fine, fixed)):
        raise ValueError("cost must be positive and subsidy/benefit/fine coefficients nonnegative")
    kinds = action_kinds(2 * len(fractions) + 1)
    real_columns, wash_columns = np.flatnonzero(kinds == IMPROVE), np.flatnonzero(kinds == INFLATE)
    delta = np.asarray(fractions) * max_change_per_period
    real_amount = np.minimum(delta, np.maximum(real_cap[:, None], 0))
    wash_amount = np.minimum(delta, np.maximum(wash_cap[:, None], 0))
    amounts = np.zeros((c.size, kinds.size))
    costs, subsidies, values = np.zeros_like(amounts), np.zeros_like(amounts), np.zeros_like(amounts)
    amounts[:, real_columns], amounts[:, wash_columns] = real_amount, wash_amount
    costs[:, real_columns] = c[:, None] * real_amount**2
    if firm_ratios:
        costs[:, wash_columns] = ratios[:, None] * c[:, None] * wash_amount**2
    else:
        costs[:, wash_columns] = wash_cost_ratio * c[:, None] * wash_amount**2
    subsidies[:, real_columns] = s[:, None] * real_amount
    report_room = np.maximum(wash_cap[:, None], 0)
    values[:, real_columns] = real[:, None] * np.minimum(real_amount, report_room) + displaced[:, None] * np.maximum(real_amount - report_room, 0)
    values[:, wash_columns] = (wash - fine)[:, None] * wash_amount
    values = values + subsidies - costs
    if np.any(fixed):
        values[:, wash_columns] -= fixed[:, None] * (wash_amount > 0)
    valid = amounts > 0
    valid[:, MAINTAIN] = True
    for columns in (real_columns, wash_columns):
        for previous, current in zip(columns, columns[1:]):
            valid[:, current] &= amounts[:, current] != amounts[:, previous]
    return CandidateSet(amounts, costs, subsidies, values, valid, kinds)


def npv_discrete(candidates: CandidateSet, minimum_npv: float = 1e-9) -> IntArray:
    valid = _mask(candidates.valid_actions, candidates.values.shape[0], candidates.kinds.size)
    priority = _action_priority(candidates.kinds.size)
    values = np.where(valid, candidates.values, -np.inf)
    actions = priority[np.argmax(values[:, priority], axis=1)]
    actions[(values[np.arange(actions.size), actions] <= minimum_npv) & valid[:, MAINTAIN]] = MAINTAIN
    return actions


def context_states(
    *, financing_exposure: FloatArray, prices: FloatArray, inflation: FloatArray,
    marginal_cost: FloatArray, real_cost_scale: float, max_change_per_period: float,
    pending_due_days: FloatArray, pending_cash_amount: FloatArray, decision_days: int,
    own_signal: FloatArray | None = None, improvement_stock: FloatArray | None = None,
) -> IntArray:
    """Four ternary features; ``own_signal`` replaces the exposure share by the firm's signal tercile.

    ``improvement_stock`` (committed real improvement in SD) replaces the cost tier, cut like inflation.
    """
    exposure, prices, inflation, cost, due, cash = _vectors(
        financing_exposure, prices, inflation, marginal_cost, pending_due_days, pending_cash_amount,
    )
    if not math.isfinite(real_cost_scale) or real_cost_scale <= 0 or not math.isfinite(max_change_per_period) or max_change_per_period <= 0 or decision_days <= 0:
        raise ValueError("context reference scales and decision interval must be positive")
    if np.any(exposure < 0) or np.any(prices <= 0) or np.any(inflation < 0) or np.any(cost <= 0) or np.any((due < 0) & (due != -1)):
        raise ValueError("invalid financing, attribute, cost or pending context")
    stage = np.where(due >= 0, 2.0, 0.0)
    stage[((due >= 0) & (due <= decision_days)) | (cash != 0)] = 1.0
    first, cuts = exposure / (exposure + prices), (.25, .50)
    if own_signal is not None:
        (signal,) = _vectors(own_signal)
        if signal.shape != exposure.shape:
            raise ValueError("own signal must match the firm vector")
        # Rank share in (0, 1), ties broken by firm order, cut into thirds.
        first = (np.argsort(np.argsort(signal, kind="stable"), kind="stable") + .5) / signal.size
        cuts = (1 / 3, 2 / 3)
    third, third_cuts = np.log(cost / real_cost_scale), (-.5, .5)
    if improvement_stock is not None:
        (third,) = _vectors(improvement_stock)
        if third.shape != exposure.shape or np.any(third < 0):
            raise ValueError("improvement stock must be nonnegative and match the firm vector")
        third_cuts = (1e-12, max_change_per_period)
    features = np.column_stack((first, inflation, third, stage))
    return ternary_states(features, (cuts, (1e-12, max_change_per_period), third_cuts, (.5, 1.5)))


def period_discount(annual_discount_rate: float, elapsed_days: Any) -> FloatArray:
    days = np.asarray(elapsed_days, dtype=np.float64)
    if not math.isfinite(annual_discount_rate) or annual_discount_rate < 0 or not np.isfinite(days).all() or np.any(days <= 0):
        raise ValueError("discount rate must be nonnegative and elapsed days finite and positive")
    return np.exp(-annual_discount_rate * days / TRADING_DAYS_PER_YEAR)


def realized_rewards(
    financing_savings: FloatArray, resource_cost: FloatArray, fines: FloatArray, subsidies: FloatArray,
    initial_cap: FloatArray, *, start_day: int, end_day: int, annual_discount_rate: float,
) -> FloatArray:
    arrays = [np.asarray(value, dtype=np.float64) for value in (financing_savings, resource_cost, fines, subsidies)]
    if arrays[0].ndim != 2 or any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError("daily ledger arrays must have matching finite day-by-firm shapes")
    cap = np.asarray(initial_cap, dtype=np.float64)
    if cap.shape != (arrays[0].shape[1],) or not np.isfinite(cap).all() or np.any(cap <= 0) or any(np.any(value < 0) for value in arrays[1:]):
        raise ValueError("initial capital must be positive; costs, fines and subsidies nonnegative")
    if type(start_day) is not int or type(end_day) is not int or not 0 <= start_day < end_day < arrays[0].shape[0]:
        raise ValueError("reward window must be a nonempty available (start,end] interval")
    net = arrays[0][start_day + 1:end_day + 1] - arrays[1][start_day + 1:end_day + 1] - arrays[2][start_day + 1:end_day + 1] + arrays[3][start_day + 1:end_day + 1]
    return period_discount(annual_discount_rate, np.arange(1, end_day - start_day + 1)) @ net / cap


def _indices(learner: CompanyQLearner, groups: Any, states: Any) -> tuple[IntArray, IntArray]:
    groups, states = np.asarray(groups), np.asarray(states)
    if groups.ndim != 1 or states.shape != groups.shape or not np.issubdtype(groups.dtype, np.integer) or not np.issubdtype(states.dtype, np.integer):
        raise ValueError("group and state indices must be matching integer vectors")
    if np.any(groups < 0) or np.any(groups >= learner.values.shape[0]) or np.any(states < 0) or np.any(states >= 81):
        raise ValueError("group or state out of range")
    return groups, states


def _mask(valid_actions: Any, size: int, action_count: int) -> NDArray[np.bool_]:
    mask = np.asarray(valid_actions)
    if mask.dtype != np.bool_ or mask.shape != (size, action_count) or not np.all(mask.any(axis=1)):
        raise ValueError("legal mask must be boolean N-by-actions with at least one legal action per row")
    return mask


def _kind_probabilities(mask: NDArray[np.bool_], kinds: IntArray) -> FloatArray:
    counts = np.column_stack([mask[:, kinds == kind].sum(axis=1) for kind in (IMPROVE, INFLATE, MAINTAIN)])
    denominator = (counts > 0).sum(axis=1)[:, None] * counts[:, kinds]
    return np.divide(mask, denominator, out=np.zeros(mask.shape), where=denominator > 0)


def choose_actions(
    learner: CompanyQLearner, groups: IntArray, states: IntArray, valid_actions: NDArray[np.bool_],
    rng: np.random.Generator | None, *, training: bool, coverage_target: int = 0,
    forced_choices: NDArray[np.bool_] | None = None,
) -> IntArray:
    if type(coverage_target) is not int or coverage_target < 0:
        raise ValueError("coverage_target must be a nonnegative integer")
    groups, states = _indices(learner, groups, states)
    if forced_choices is not None:
        if (forced_choices.dtype != np.bool_ or forced_choices.shape != groups.shape
                or not forced_choices.flags.writeable):
            raise ValueError("forced choices must be a writable boolean vector matching groups")
        forced_choices[:] = False
    action_count = learner.values.shape[2]
    kinds = action_kinds(action_count)
    valid = _mask(valid_actions, groups.size, action_count)
    values = np.where(valid, learner.values[groups, states], -np.inf)
    if not training:
        priority = _action_priority(action_count)
        actions = priority[np.argmax(values[:, priority], axis=1)]
        unvisited = learner.counts[groups, states].sum(axis=1) == 0
        fallback = priority[np.argmax(valid[:, priority], axis=1)]
        actions[unvisited] = fallback[unvisited]
        return actions
    if rng is None or not learner.values.flags.writeable or not learner.counts.flags.writeable:
        raise ValueError("training requires its own RNG and a writable learner")
    best = valid & (values == values.max(axis=1, keepdims=True))
    if action_count == 3:
        probabilities = learner.epsilon * valid / valid.sum(axis=1, keepdims=True) + (1 - learner.epsilon) * best / best.sum(axis=1, keepdims=True)
    else:
        probabilities = learner.epsilon * _kind_probabilities(valid, kinds) + (1 - learner.epsilon) * _kind_probabilities(best, kinds)
    last_legal = np.where(valid, np.arange(action_count), -1).max(axis=1)
    draws = rng.random(groups.size)
    actions = np.minimum((draws[:, None] >= probabilities.cumsum(axis=1)).sum(axis=1), last_legal)
    if coverage_target:
        reservations = np.zeros_like(learner.counts)
        for row, (group, state) in enumerate(zip(groups, states)):
            counts = learner.counts[group, state] + reservations[group, state]
            deficient = valid[row] & (counts < coverage_target)
            if deficient.any():
                deficient &= counts == counts[deficient].min()
                probability = _kind_probabilities(deficient[None, :], kinds)[0]
                actions[row] = min(int((draws[row] >= probability.cumsum()).sum()),
                                   int(np.flatnonzero(deficient)[-1]))
                if forced_choices is not None:
                    forced_choices[row] = True
            reservations[group, state, actions[row]] += 1
    return actions


def batch_update(
    learner: CompanyQLearner, ids: IntArray, groups: IntArray, states: IntArray, actions: IntArray,
    rewards: FloatArray, next_groups: IntArray, next_states: IntArray, next_valid_actions: NDArray[np.bool_],
    *, elapsed_days: Any, annual_discount_rate: float, terminated: NDArray[np.bool_],
) -> None:
    if not learner.values.flags.writeable or not learner.counts.flags.writeable:
        raise ValueError("frozen learner cannot update")
    groups, states = _indices(learner, groups, states)
    next_groups, next_states = _indices(learner, next_groups, next_states)
    ids, actions, rewards, terminated = np.asarray(ids), np.asarray(actions), np.asarray(rewards), np.asarray(terminated)
    if any(value.shape != groups.shape for value in (ids, actions, rewards, terminated, next_groups)) or not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(actions.dtype, np.integer) or terminated.dtype != np.bool_:
        raise ValueError("transition fields must be matching vectors with integer IDs/actions and boolean termination")
    action_count = learner.values.shape[2]
    if np.unique(ids).size != ids.size or np.any(actions < 0) or np.any(actions >= action_count) or not np.isfinite(rewards).all():
        raise ValueError("transition IDs must be unique, actions valid and rewards finite")
    valid = _mask(next_valid_actions, groups.size, action_count)
    discount = np.broadcast_to(period_discount(annual_discount_rate, elapsed_days), groups.shape)
    old = learner.values.copy()
    future = np.where(valid, old[next_groups, next_states], -np.inf).max(axis=1)
    targets = rewards + discount * np.where(terminated, 0.0, future)
    keys, inverse = np.unique(np.column_stack((groups, states, actions)), axis=0, return_inverse=True)
    power = learner.learning_rate_power
    for index, key in enumerate(keys):
        member = inverse == index
        count = int(member.sum())
        mean = math.fsum(np.sort(targets[member])) / count
        address = tuple(key)
        # A batch of k same-key samples after n updates moves by min(1, k (n + k) ** -power); power 1 is the running mean.
        step = learner.learning_rate if not power else min(1.0, count * float(learner.counts[address] + count) ** -power)
        learner.values[address] = old[address] + step * (mean - old[address])
        learner.counts[address] += count


def table_fingerprint(learner: CompanyQLearner, *, prefix_groups: int | None = None) -> str:
    digest = hashlib.sha256()
    for value in (learner.values[:prefix_groups], learner.counts[:prefix_groups]):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_frozen(path: str | Path, learner: CompanyQLearner, metadata: Mapping[str, Any]) -> str:
    if learner.values.ndim != 3 or learner.values.shape[1] != 81 or learner.values.shape[2] < 3 or learner.values.shape[2] % 2 != 1 or learner.counts.shape != learner.values.shape or not np.issubdtype(learner.counts.dtype, np.integer) or not np.isfinite(learner.values).all() or np.any(learner.counts < 0):
        raise ValueError("cannot freeze an invalid Q table")
    fields = {"format_version": 1 if learner.values.shape[2] == 3 else 2, "metadata": dict(metadata),
              "learning_rate": learner.learning_rate, "discount": learner.discount}
    if learner.learning_rate_power:
        fields["learning_rate_power"] = learner.learning_rate_power
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, values=learner.values, counts=learner.counts, metadata=np.array(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(
    path: str | Path, expected_metadata: Mapping[str, Any] | None = None, *, expected_sha256: str | None = None,
) -> tuple[CompanyQLearner, dict[str, Any]]:
    if expected_sha256 is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("frozen model hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        values, counts = arrays["values"].copy(), arrays["counts"].copy()
        payload = json.loads(str(arrays["metadata"].item()))
    if (values.ndim != 3 or values.shape[1] != 81 or values.shape[2] < 3 or values.shape[2] % 2 != 1
            or payload["format_version"] != (1 if values.shape[2] == 3 else 2)
            or counts.shape != values.shape or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or not np.isfinite(values).all()):
        raise ValueError("invalid frozen Q table")
    metadata = payload["metadata"]
    if expected_metadata is not None and any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("frozen model metadata mismatch")
    learner = CompanyQLearner(groups=values.shape[0], epsilon=0, learning_rate=payload["learning_rate"], discount=payload["discount"], actions=values.shape[2])
    learner.learning_rate_power = float(payload.get("learning_rate_power", 0.0))
    learner.values, learner.counts = values, counts
    learner.values.flags.writeable = learner.counts.flags.writeable = False
    return learner, metadata
