"""Bounded-rationality firm policies overcoming the raw-account SNR barrier (Track 4).

Provides two model-free / boundedly rational firm decision policies that operate on variance-reduced
attribute difference rewards (Agogino & Tumer 2008; Sutton & Barto 2018), stripping exogenous fundamental
value noise while preserving real implementation delays (120d vs 20d), stochastic regulatory audits,
realized fines and stochastic pre-publication assurance checks:

1. ``DifferenceRewardQLearner`` (``firm_policy="diff_q"``):
   Online non-stationary tabular learner over a compact ``(size_tier, open_case)`` state space
   (6 states total), updating action values from realized difference flows plus empirically estimated
   survival/fine continuations (zero knowledge of closed-form policy parameters ``tau``, ``q_i`` or
   ``a_k``).

2. ``PeerImitationPolicy`` (``firm_policy="peer_imitation"``):
   Inter-firm social benchmarking symmetric with investors' Schlag (1998) pairwise proportional
   imitation rule: reviewing firms sample a same-tier peer and copy its disclosure strategy (and
   assurance provider under firm choice) with probability proportional to realized disclosure net payoff
   advantage, plus small behavioral exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import TRADING_DAYS_PER_YEAR
from .learning import IMPROVE, INFLATE, MAINTAIN, CandidateSet

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class _ExperienceTicket:
    firm: int
    tier: int
    state: int
    action: int
    kind: int
    amount: float
    due_day: int
    cost_net: float
    opened_case: bool
    provider: int
    age: int = 0
    acc_fin: float = 0.0
    fine_paid: float = 0.0
    landed_flow: float = 0.0
    alive: bool = True
    settled: bool = False


@dataclass(slots=True)
class DifferenceRewardTracker:
    """Tracks realized variance-reduced attribute difference returns and empirical hazard rates."""

    n_firms: int
    n_tiers: int
    period_days: int
    horizon_periods: int
    discount_rate: float
    ema_alpha: float = 0.15
    n_providers: int = 4
    tickets: list[_ExperienceTicket] = field(default_factory=list)
    # Empirical per-tier audit hazard (Beta(0.5, 49.5) weak prior -> 0.01) & fine-when-caught EMA.
    audit_events: FloatArray = field(init=False)
    audit_catches: FloatArray = field(init=False)
    emp_audit_hazard: FloatArray = field(init=False)
    emp_fine_on_catch: FloatArray = field(init=False)
    emp_fine_seen: NDArray[np.bool_] = field(init=False)
    # Empirical per-provider correction rate (weak prior -> 0.0) updated only from realized checks.
    emp_provider_catch: FloatArray = field(init=False)
    provider_checks: FloatArray = field(init=False)
    # Per-firm trailing realized attribute net performance (used by peer imitation & diagnostics).
    firm_perf: FloatArray = field(init=False)
    # Diagnostic noise tracking: realized reward variance across settled tickets.
    settled_returns: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.audit_events = np.full(self.n_tiers, 20.0, dtype=np.float64)
        self.audit_catches = np.full(self.n_tiers, 0.2, dtype=np.float64)
        self.emp_audit_hazard = np.full(self.n_tiers, 0.01, dtype=np.float64)
        self.emp_fine_on_catch = np.zeros(self.n_tiers, dtype=np.float64)
        self.emp_fine_seen = np.zeros(self.n_tiers, dtype=bool)
        self.emp_provider_catch = np.zeros(self.n_providers, dtype=np.float64)
        self.provider_checks = np.zeros(self.n_providers, dtype=np.float64)
        self.firm_perf = np.zeros(self.n_firms, dtype=np.float64)

    @property
    def gamma(self) -> float:
        return float(np.exp(-self.discount_rate * self.period_days / TRADING_DAYS_PER_YEAR))

    def observe_assurance_period(
        self,
        checked_inflated: NDArray[np.bool_],
        corrected: NDArray[np.bool_],
        client_of: IntArray,
    ) -> None:
        """Update empirical provider detection rates and terminate corrected inflation tickets."""
        for k in range(self.n_providers):
            mask = checked_inflated & (client_of == k)
            n_obs = int(mask.sum())
            if n_obs > 0:
                rate = float(corrected[mask].sum()) / n_obs
                alpha = max(self.ema_alpha, 1.0 / (self.provider_checks[k] + 1.0))
                self.emp_provider_catch[k] = (1.0 - alpha) * self.emp_provider_catch[k] + alpha * rate
                self.provider_checks[k] += 1.0
        if corrected.any():
            corr_firms = set(np.flatnonzero(corrected).tolist())
            for tk in self.tickets:
                if tk.alive and tk.kind == INFLATE and tk.firm in corr_firms:
                    tk.alive = False

    def step_and_settle(
        self,
        *,
        day: int,
        size_class: IntArray,
        selected: NDArray[np.bool_],
        caught: NDArray[np.bool_],
        had_inflation_before_audit: NDArray[np.bool_],
        penalty_fraction: FloatArray,
        benefit: FloatArray,
        transparency: float,
        fixed_fine_basis: str,
        real_persistence: FloatArray | None,
        client_of: IntArray | None,
    ) -> list[tuple[int, int, int, int, float]]:
        """Advance all open tickets by one period; return ``(firm, tier, state, action, R)`` updates."""
        gamma = self.gamma
        dt_years = self.period_days / TRADING_DAYS_PER_YEAR

        # 1. Update empirical audit hazard & fine-when-caught per size tier (model-free).
        for tier in range(self.n_tiers):
            tier_mask = size_class == tier
            n_tier = int(tier_mask.sum())
            if n_tier > 0:
                obs_q = float(selected[tier_mask].sum()) / n_tier
                self.emp_audit_hazard[tier] = (
                    (1.0 - self.ema_alpha) * self.emp_audit_hazard[tier] + self.ema_alpha * obs_q
                )
            caught_tier = tier_mask & caught
            if caught_tier.any():
                mean_pen = float(penalty_fraction[caught_tier].mean())
                if not self.emp_fine_seen[tier]:
                    self.emp_fine_on_catch[tier] = mean_pen
                    self.emp_fine_seen[tier] = True
                else:
                    self.emp_fine_on_catch[tier] = (
                        (1.0 - self.ema_alpha) * self.emp_fine_on_catch[tier] + self.ema_alpha * mean_pen
                    )

        # 2. Advance active tickets and emit settled difference returns.
        updates: list[tuple[int, int, int, int, float]] = []
        survivors: list[_ExperienceTicket] = []
        for tk in self.tickets:
            tk.age += 1
            disc = gamma ** tk.age

            # Check realized audit detection on inflate tickets.
            if tk.alive and tk.kind == INFLATE and caught[tk.firm]:
                if tk.opened_case or fixed_fine_basis == "report":
                    tk.fine_paid += disc * float(penalty_fraction[tk.firm])
                tk.alive = False

            # Accumulate realized attribute difference financing flow once action has landed.
            if tk.alive and day >= tk.due_day:
                if tk.kind == IMPROVE:
                    elapsed = max(0, day - tk.due_day)
                    if real_persistence is not None and elapsed < real_persistence.size:
                        surv_share = float(real_persistence[elapsed])
                    else:
                        surv_share = 1.0
                    delta_s = tk.amount * surv_share
                else:
                    delta_s = (1.0 - transparency) * tk.amount
                flow = dt_years * float(benefit[tk.firm]) * delta_s
                tk.acc_fin += disc * flow
                tk.landed_flow = flow

            # Settle ticket when terminated early OR after landing (age == 3, 180d -> both 20d & 120d landed).
            if not tk.alive:
                ret = tk.acc_fin - tk.cost_net - tk.fine_paid
                updates.append((tk.firm, tk.tier, tk.state, tk.action, ret))
                self.settled_returns.append(ret)
                self._update_firm_perf(tk.firm, ret)
                tk.settled = True
            elif tk.age >= 3 and not tk.settled:
                # Empirical continuation over remaining contract horizon (H_p - age periods)
                # using ONLY observed landed_flow, empirical tier audit hazard, empirical provider
                # correction rate and empirical fine-when-caught (zero closed-form policy reads).
                rem = max(0, self.horizon_periods - tk.age)
                if tk.kind == IMPROVE:
                    if real_persistence is not None:
                        cont_fin = 0.0
                        for step in range(1, rem + 1):
                            d_off = min(
                                max(0, day + step * self.period_days - tk.due_day),
                                real_persistence.size - 1,
                            )
                            cont_fin += (gamma ** (tk.age + step)) * dt_years * float(benefit[tk.firm]) * tk.amount * float(real_persistence[d_off])
                    else:
                        cont_fin = tk.landed_flow * (gamma ** (tk.age + 1)) * (1.0 - gamma**rem) / max(1e-12, 1.0 - gamma)
                    ret = tk.acc_fin + cont_fin - tk.cost_net
                else:
                    q_hat = float(np.clip(self.emp_audit_hazard[tk.tier], 0.0, 0.99))
                    a_hat = 0.0
                    if client_of is not None and client_of[tk.firm] >= 0:
                        a_hat = float(np.clip(self.emp_provider_catch[int(client_of[tk.firm])], 0.0, 0.99))
                    surv_step = (1.0 - q_hat) * (1.0 - a_hat)
                    eff_disc = gamma * surv_step
                    if rem > 0 and tk.landed_flow > 0.0:
                        cont_fin = tk.landed_flow * (gamma ** (tk.age + 1)) * surv_step * (1.0 - eff_disc**rem) / max(1e-12, 1.0 - eff_disc)
                    else:
                        cont_fin = 0.0
                    # Expected remaining fine from empirical audit hazard & empirical fine-on-catch
                    exp_fine = 0.0
                    if (tk.opened_case or fixed_fine_basis == "report") and q_hat > 0.0 and self.emp_fine_seen[tk.tier]:
                        denom = max(1e-12, 1.0 - eff_disc)
                        # Probability audit catches before assurance clears on future steps
                        exp_fine = (
                            float(self.emp_fine_on_catch[tk.tier])
                            * (gamma ** (tk.age + 1))
                            * (1.0 - a_hat)
                            * q_hat
                            / denom
                        )
                    ret = tk.acc_fin + cont_fin - tk.cost_net - tk.fine_paid - exp_fine
                updates.append((tk.firm, tk.tier, tk.state, tk.action, ret))
                self.settled_returns.append(ret)
                self._update_firm_perf(tk.firm, ret)
                tk.settled = True
            else:
                survivors.append(tk)
        self.tickets = survivors
        return updates

    def record_decisions(
        self,
        *,
        day: int,
        size_class: IntArray,
        open_case: NDArray[np.bool_],
        actions: IntArray,
        candidates: CandidateSet,
        real_due: int,
        wash_due: int,
        client_of: IntArray | None,
        provider_quality: FloatArray | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        """Record new period actions; MAINTAIN actions settle immediately at 0.0."""
        immediate: list[tuple[int, int, int, int, float]] = []
        obs_q = self.emp_provider_catch if provider_quality is None else provider_quality
        for i in range(self.n_firms):
            act = int(actions[i])
            kind = int(candidates.kinds[act])
            tier = int(size_class[i])
            prov = -1 if client_of is None else int(client_of[i])
            strict = int(prov >= 0 and float(obs_q[prov]) >= 0.08)
            st = int(open_case[i]) + 2 * strict
            if kind == MAINTAIN:
                immediate.append((i, tier, st, act, 0.0))
                self._update_firm_perf(i, 0.0)
                continue
            due = real_due if kind == IMPROVE else wash_due
            cost_net = float(candidates.cost_fraction[i, act] - candidates.subsidy_fraction[i, act])
            self.tickets.append(
                _ExperienceTicket(
                    firm=i,
                    tier=tier,
                    state=st,
                    action=act,
                    kind=kind,
                    amount=float(candidates.amounts[i, act]),
                    due_day=due,
                    cost_net=cost_net,
                    opened_case=(not bool(open_case[i]) and kind == INFLATE),
                    provider=prov,
                )
            )
        return immediate

    def _update_firm_perf(self, firm: int, ret: float) -> None:
        self.firm_perf[firm] = (1.0 - self.ema_alpha) * self.firm_perf[firm] + self.ema_alpha * ret


class DifferenceRewardQLearner:
    """Online non-stationary tabular learner over (size_tier, (open_case, provider_strict), action)."""

    def __init__(
        self,
        *,
        n_tiers: int,
        n_actions: int,
        alpha: float = 0.15,
        epsilon: float = 0.10,
        n_providers: int = 4,
    ) -> None:
        self.n_tiers = n_tiers
        self.n_states = 4  # 0: clean/lax, 1: open/lax, 2: clean/strict, 3: open/strict
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_providers = n_providers
        self.values = np.zeros((n_tiers, self.n_states, n_actions), dtype=np.float64)
        self.counts = np.zeros((n_tiers, self.n_states, n_actions), dtype=np.int64)
        self.reserved = np.zeros((n_tiers, self.n_states, n_actions), dtype=np.int64)

    def update(self, updates: list[tuple[int, int, int, int, float]]) -> None:
        for _, tier, state, action, ret in updates:
            c = self.counts[tier, state, action]
            # Fast initial convergence (1/(c+1)) transitioning to recency-weighted EMA alpha
            step = max(self.alpha, 1.0 / (c + 1.0))
            self.values[tier, state, action] += step * (ret - self.values[tier, state, action])
            self.counts[tier, state, action] = c + 1

    def state_indices(
        self,
        open_case: NDArray[np.bool_],
        client_of: IntArray | None,
        emp_provider_catch: FloatArray,
    ) -> IntArray:
        states = open_case.astype(np.int64)
        if client_of is not None:
            strict = np.where(client_of >= 0, emp_provider_catch[np.maximum(client_of, 0)] >= 0.08, False)
            states = states + 2 * strict.astype(np.int64)
        return states

    def choose_actions(
        self,
        size_class: IntArray,
        open_case: NDArray[np.bool_],
        valid_actions: NDArray[np.bool_],
        rng: np.random.Generator,
        *,
        client_of: IntArray | None = None,
        emp_provider_catch: FloatArray | None = None,
        provider_quality: FloatArray | None = None,
    ) -> IntArray:
        n = size_class.size
        obs_q = provider_quality if provider_quality is not None else emp_provider_catch
        if client_of is not None and obs_q is not None:
            states = self.state_indices(open_case, client_of, obs_q)
        else:
            states = open_case.astype(np.int64)
        q_vals = np.where(valid_actions, self.values[size_class, states], -np.inf)
        best = np.argmax(q_vals, axis=1)
        draws = rng.random(n)
        actions = best.copy()
        for i in range(n):
            t, s = int(size_class[i]), int(states[i])
            seen = self.counts[t, s] + self.reserved[t, s]
            unvisited = valid_actions[i] & (seen == 0)
            if unvisited.any():
                act = int(np.flatnonzero(unvisited)[0])
                actions[i] = act
                self.reserved[t, s, act] += 1
            elif draws[i] < self.epsilon:
                legal = np.flatnonzero(valid_actions[i])
                actions[i] = int(rng.choice(legal))
        return actions

    def choose_providers(
        self,
        *,
        size_class: IntArray,
        open_case: NDArray[np.bool_],
        mandated: NDArray[np.bool_],
        current_providers: IntArray,
        emp_provider_catch: FloatArray,
        candidates_kinds: IntArray,
        rng: np.random.Generator,
        provider_quality: FloatArray | None = None,
    ) -> IntArray:
        """Boundedly rational provider choice: firms wanting to inflate at lax providers pick lowest stringency."""
        chosen = current_providers.copy()
        obs_q = emp_provider_catch if provider_quality is None else provider_quality
        inflate_cols = np.flatnonzero(candidates_kinds == INFLATE)
        non_inflate_cols = np.flatnonzero(candidates_kinds != INFLATE)
        min_q = float(np.min(obs_q))
        laxest_pool = np.flatnonzero(obs_q <= min_q + 1e-12)
        for i in np.flatnonzero(mandated):
            # Evaluate firm's learned preference for inflation under a lax provider (state 0 or 1)
            lax_state = int(open_case[i])
            q_lax = self.values[size_class[i], lax_state]
            wants_inflate = float(q_lax[inflate_cols].max()) > float(q_lax[non_inflate_cols].max())
            if wants_inflate:
                if rng.random() < self.epsilon:
                    chosen[i] = int(rng.integers(0, obs_q.size))
                elif current_providers[i] < 0 or float(obs_q[current_providers[i]]) > min_q + 1e-12:
                    chosen[i] = int(laxest_pool[0]) if laxest_pool.size == 1 else int(rng.choice(laxest_pool))
        return chosen


class PeerImitationPolicy:
    """Inter-firm pairwise proportional imitation (Schlag 1998) over realized peer-cohort returns."""

    def __init__(
        self,
        *,
        n_firms: int,
        n_actions: int,
        intensity: float = 100.0,
        update_fraction: float = 0.25,
        exploration: float = 0.05,
        alpha: float = 0.15,
        rng: np.random.Generator,
    ) -> None:
        self.n_firms = n_firms
        self.n_tiers = 3
        self.n_states = 4  # 0: clean/lax, 1: open/lax, 2: clean/strict, 3: open/strict
        self.n_actions = n_actions
        self.intensity = intensity
        self.update_fraction = update_fraction
        self.exploration = exploration
        self.alpha = alpha
        # Uniform initial behavioral diversity across actions in each observable condition
        self.strategies = rng.integers(0, n_actions, size=(n_firms, self.n_states), dtype=np.int64)
        self.cohort_perf = np.zeros((self.n_tiers, self.n_states, n_actions), dtype=np.float64)
        self.cohort_counts = np.zeros((self.n_tiers, self.n_states, n_actions), dtype=np.int64)

    def update(self, updates: list[tuple[int, int, int, int, float]]) -> None:
        """Update peer-cohort realized attribute performance from settled difference returns."""
        for _, tier, state, action, ret in updates:
            c = self.cohort_counts[tier, state, action]
            step = max(self.alpha, 1.0 / (c + 1.0))
            self.cohort_perf[tier, state, action] += step * (ret - self.cohort_perf[tier, state, action])
            self.cohort_counts[tier, state, action] = c + 1

    def state_indices(
        self,
        open_case: NDArray[np.bool_],
        client_of: IntArray | None,
        obs_quality: FloatArray | None,
    ) -> IntArray:
        states = open_case.astype(np.int64)
        if client_of is not None and obs_quality is not None:
            strict = np.where(client_of >= 0, obs_quality[np.maximum(client_of, 0)] >= 0.08, False)
            states = states + 2 * strict.astype(np.int64)
        return states

    def choose_actions(
        self,
        *,
        size_class: IntArray,
        open_case: NDArray[np.bool_],
        valid_actions: NDArray[np.bool_],
        firm_perf: FloatArray,
        rng: np.random.Generator,
        client_of: IntArray | None = None,
        emp_provider_catch: FloatArray | None = None,
        provider_quality: FloatArray | None = None,
    ) -> IntArray:
        n = self.n_firms
        obs_q = provider_quality if provider_quality is not None else emp_provider_catch
        states = self.state_indices(open_case, client_of, obs_q)
        n_review = max(1, int(round(self.update_fraction * n)))
        reviewers = rng.choice(n, size=n_review, replace=False)

        # Within-(tier, state) strategy-performance dispersion normalizes imitation probability
        state_scale = np.full((self.n_tiers, self.n_states), 1e-6, dtype=np.float64)
        for t in range(self.n_tiers):
            for s in range(self.n_states):
                seen = self.cohort_counts[t, s] > 0
                if seen.sum() > 1:
                    vals = self.cohort_perf[t, s, seen]
                    span = float(np.max(vals) - np.min(vals))
                    state_scale[t, s] = max(span, 1e-6)

        for i in reviewers:
            t, s = int(size_class[i]), int(states[i])
            if rng.random() < self.exploration:
                legal = np.flatnonzero(valid_actions[i])
                self.strategies[i, s] = int(rng.choice(legal))
                continue
            # Sample a random same-tier peer (Schlag 1998 pairwise proportional imitation)
            pool = np.flatnonzero(size_class == t)
            if pool.size <= 1:
                continue
            peer = int(rng.choice(pool))
            if peer == i:
                continue
            own_act = int(self.strategies[i, s])
            peer_act = int(self.strategies[peer, s])
            if peer_act == own_act or not valid_actions[i, peer_act]:
                continue
            if self.cohort_counts[t, s, peer_act] > 0 and self.cohort_counts[t, s, own_act] > 0:
                adv = float(self.cohort_perf[t, s, peer_act] - self.cohort_perf[t, s, own_act])
                scale = state_scale[t, s]
            else:
                adv = float(firm_perf[peer] - firm_perf[i])
                scale = max(float(np.std(firm_perf[pool])), 1e-6)
            if adv > 0.0:
                prob = min(1.0, (self.intensity / 100.0) * (adv / scale))
                if rng.random() < prob:
                    self.strategies[i, s] = peer_act

        # Enforce legality on every firm
        actions = self.strategies[np.arange(n), states].copy()
        for i in range(n):
            if not valid_actions[i, actions[i]]:
                s = int(states[i])
                actions[i] = MAINTAIN if valid_actions[i, MAINTAIN] else int(np.flatnonzero(valid_actions[i])[0])
                self.strategies[i, s] = actions[i]
        return actions

    def choose_providers(
        self,
        *,
        size_class: IntArray,
        mandated: NDArray[np.bool_],
        current_providers: IntArray,
        firm_perf: FloatArray,
        emp_provider_catch: FloatArray,
        candidates_kinds: IntArray,
        rng: np.random.Generator,
        open_case: NDArray[np.bool_] | None = None,
        provider_quality: FloatArray | None = None,
    ) -> IntArray:
        """Inflating peers switch toward the lowest-stringency provider under firm choice."""
        chosen = current_providers.copy()
        obs_q = emp_provider_catch if provider_quality is None else provider_quality
        min_q = float(np.min(obs_q))
        laxest_pool = np.flatnonzero(obs_q <= min_q + 1e-12)
        for i in np.flatnonzero(mandated):
            if rng.random() < self.exploration:
                chosen[i] = int(rng.integers(0, obs_q.size))
                continue
            lax_state = 0 if open_case is None else int(open_case[i])
            if candidates_kinds[self.strategies[i, lax_state]] == INFLATE:
                if current_providers[i] < 0 or float(obs_q[current_providers[i]]) > min_q + 1e-12:
                    chosen[i] = int(laxest_pool[0]) if laxest_pool.size == 1 else int(rng.choice(laxest_pool))
        return chosen
