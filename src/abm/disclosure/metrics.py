"""Descriptive disclosure-game metrics over a fixed state window.

Decision commitments in (start_day, end_day] are valued at decision-day
prices. They affect intrinsic value on the following day. Delayed actions
committed before the window can still affect its observed attribute changes.
"""

from __future__ import annotations

import json

import numpy as np

from .config import TRADING_DAYS_PER_YEAR, MarketConfig
from .domain import DomainSpec
from .firms import LARGE, SIZE_CLASS_NAMES, SMALL
from .game import IMPROVE, INFLATE, MAINTAIN, DisclosureResult, expected_natural_path


def compute_metrics(
    result: DisclosureResult,
    config: MarketConfig,
    domain: DomainSpec,
    *,
    start_day: int | None = None,
) -> dict:
    """Measure state changes and committed costs, without causal attribution.

    Raw changes retain the domain's direction. Efficiency uses the total
    gain above a bounded natural-decay baseline, with raw gains signed so
    improvement is positive, divided by total real resource expenditure.
    Fines and subsidies are reported separately as fiscal transfers.
    With ``natural_reversion`` the no-action baseline is the expected path
    (bounded decay, then reversion to the cross-sectional mean).
    Full-period changes use day zero. Terminal summaries average the last
    at most 252 state days relative to day zero; shorter histories are flagged.
    """
    true = np.asarray(result.true_goodness)
    if true.ndim != 2 or min(true.shape) == 0:
        raise ValueError("true_goodness must be a nonempty day-by-firm array")
    days, firms = true.shape
    end = days - 1
    start = config.burn_in_days if start_day is None else start_day
    if type(start) is not int or not 0 <= start <= end:
        raise ValueError("start_day must be an integer between zero and the final day")
    if firms != config.n_firms:
        raise ValueError("result firm count does not match config.n_firms")

    prices = np.asarray(result.market.prices)
    inflation = np.asarray(result.inflation)
    impact = np.asarray(result.impact_share)
    impact_wealth = np.asarray(result.impact_wealth_share)
    size_class = np.asarray(result.market.size_class)
    slope = np.asarray(result.pricing_slope)
    if prices.shape != true.shape or inflation.shape != true.shape:
        raise ValueError("prices and inflation must match the attribute history")
    if impact.shape != (days,) or impact_wealth.shape != (days,) or slope.shape != (days,):
        raise ValueError("investor shares and pricing_slope must have one value per state day")
    if size_class.shape != (firms,) or not np.isin(size_class, [0, 1, 2]).all():
        raise ValueError("size_class must identify large, mid or small for each firm")
    for name, values in (
        ("true_goodness", true), ("prices", prices), ("inflation", inflation),
        ("impact_share", impact), ("impact_wealth_share", impact_wealth),
    ):
        if not np.isfinite(values[start:]).all():
            raise ValueError(f"{name} contains non-finite observations in the metric window")
    if (prices[start:] <= 0).any() or (inflation[start:] < 0).any():
        raise ValueError("prices must be positive and inflation nonnegative in the metric window")
    terminal_start = max(0, days - TRADING_DAYS_PER_YEAR)
    if not np.isfinite(true[0]).all() or not np.isfinite(true[terminal_start:]).all():
        raise ValueError("true_goodness contains non-finite initial or terminal observations")
    if not np.isfinite(inflation[terminal_start:]).all() or (inflation[terminal_start:] < 0).any():
        raise ValueError("inflation contains non-finite or negative terminal observations")
    if not np.isfinite(prices[0]).all() or (prices[0] <= 0).any():
        raise ValueError("initial prices must be finite and positive")

    actions = np.asarray(result.actions)
    if actions.ndim != 2 or actions.shape[1] != firms:
        raise ValueError("actions must be a decision-by-firm array")
    decision_days = np.arange(actions.shape[0]) * config.firm_decision_interval
    if actions.shape[0] <= end // config.firm_decision_interval:
        raise ValueError("decision history does not cover the final day")
    selected = (decision_days > start) & (decision_days <= end)
    carry_in = (decision_days == start) & (decision_days > 0)
    checked = selected | carry_in
    fractions = {}
    for name in ("cost_fraction", "penalty_fraction", "subsidy_fraction"):
        values = np.asarray(getattr(result, name))
        if values.shape != actions.shape:
            raise ValueError(f"{name} must match the action history")
        if not np.isfinite(values[checked]).all() or (values[checked] < 0).any():
            raise ValueError(f"{name} contains non-finite or negative decision observations")
        fractions[name] = values
    if not np.isin(actions[checked], [IMPROVE, INFLATE, MAINTAIN]).all():
        raise ValueError("unknown firm action in the metric window")
    detections = np.asarray(result.detections)
    if detections.shape != actions.shape:
        raise ValueError("detections must match the action history")
    if ((actions[checked] == MAINTAIN) & (fractions["cost_fraction"][checked] != 0)).any():
        raise ValueError("maintenance actions must have zero resource cost")

    weights_start = prices[start] / prices[start].sum()
    weights_end = prices[end] / prices[end].sum()

    def summarize(change):
        raw = domain.direction * domain.initial_sd * change
        return {
            "equal_weight_sd": float(change.mean()),
            "equal_weight_raw": float(raw.mean()),
            "fixed_start_cap_weight_sd": float(weights_start @ change),
            "fixed_start_cap_weight_raw": float(weights_start @ raw),
            "end_cap_weight_sd": float(weights_end @ change),
            "end_cap_weight_raw": float(weights_end @ raw),
        }

    change = true[end] - true[start]
    # Terminal-year level against the no-action path from day zero: bounded natural decay only.
    terminal_days = np.arange(terminal_start, days)
    terminal_baseline = np.maximum(
        true[0][None, :] - domain.natural_decay * terminal_days[:, None] / TRADING_DAYS_PER_YEAR,
        domain.goodness_bounds[0],
    )
    baseline = np.maximum(
        true[start] - domain.natural_decay * (end - start) / TRADING_DAYS_PER_YEAR,
        domain.goodness_bounds[0],
    )
    if domain.natural_reversion:
        # Reverting values: the expected no-action path from the same cross-section.
        anchor = getattr(result, "reversion_anchor", None)
        terminal_baseline = expected_natural_path(domain, true[0], terminal_days,
                                                  anchor=None if anchor is None else anchor[0])
        baseline = expected_natural_path(domain, true[start], np.array([end - start]),
                                         anchor=None if anchor is None else anchor[start])[0]
    gain = true[end] - baseline
    size = {}
    for index, name in enumerate(SIZE_CLASS_NAMES):
        members = size_class == index
        count = int(members.sum())
        size[name] = {
            "count": count,
            "change_sd": float(change[members].mean()) if count else None,
            "change_raw": float(domain.direction * domain.initial_sd * change[members].mean()) if count else None,
            "decay_adjusted_gain_sd": float(gain[members].mean()) if count else None,
        }
    has_extremes = size[SIZE_CLASS_NAMES[LARGE]]["count"] and size[SIZE_CLASS_NAMES[SMALL]]["count"]
    gap = size["large"]["change_sd"] - size["small"]["change_sd"] if has_extremes else None
    size["large_minus_small_sd"] = gap
    size["large_minus_small_raw"] = domain.direction * domain.initial_sd * gap if gap is not None else None

    raw_gap = np.abs(domain.to_raw(true[start:] + inflation[start:]) - domain.to_raw(true[start:]))
    money = fractions["cost_fraction"][selected] * prices[decision_days[selected]]
    resource_cost = float(money.sum())
    initial_market_cap = float(prices[0].sum())
    fines = float((fractions["penalty_fraction"][selected] * prices[decision_days[selected]]).sum())
    subsidies = float((fractions["subsidy_fraction"][selected] * prices[decision_days[selected]]).sum())
    final_decision = decision_days[selected] == end
    selected_slope = slope[decision_days[selected]]
    if np.isinf(selected_slope).any():
        raise ValueError("pricing_slope contains infinite decision observations")
    finite_slope = selected_slope[np.isfinite(selected_slope)]
    missing_slope = int(selected_slope.size - finite_slope.size)
    window_actions = actions[selected]
    washing = window_actions == INFLATE

    metrics = {
        "window": {
            "start_day": start,
            "end_day": end,
            "elapsed_days": end - start,
            "state_observations": end - start + 1,
            "decision_observations": int(selected.sum()),
            "decision_days": decision_days[selected].tolist(),
            "cost_basis": "Commitments in (start_day, end_day] at decision-day prices; applied the following day.",
            "interpretation": "Descriptive changes include delayed pre-window actions, not causal returns on in-window decisions.",
        },
        "true_change": summarize(change),
        "full_period_change": {
            "equal_weight_sd": float((true[end] - true[0]).mean()),
            "equal_weight_raw": float(domain.direction * domain.initial_sd * (true[end] - true[0]).mean()),
        },
        "terminal_year": {
            "start_day": terminal_start,
            "observations": days - terminal_start,
            "complete_year": days - terminal_start == TRADING_DAYS_PER_YEAR,
            "change_from_initial_sd": float((true[terminal_start:] - true[0]).mean()),
            "decay_adjusted_from_initial_sd": float((true[terminal_start:] - terminal_baseline).mean()),
            "inflation_mean_sd": float(inflation[terminal_start:].mean()),
        },
        "decay_adjusted_gain": summarize(gain),
        "size": size,
        "inflation": {
            "initial_mean_sd": float(inflation[start].mean()),
            "final_mean_sd": float(inflation[end].mean()),
            "mean_sd": float(inflation[start:].mean()),
            "mean_absolute_raw_gap": float(raw_gap.mean()),
            "final_absolute_raw_gap": float(raw_gap[-1].mean()),
            "mean_range_fraction": float(raw_gap.mean() / (domain.raw_upper - domain.raw_lower)),
        },
        "actions": {
            "scope": "Firm decisions in (start_day, end_day]; detections are that day's audits that found inflation.",
            "decisions": int(window_actions.size),
            "improve": int((window_actions == IMPROVE).sum()),
            "wash": int(washing.sum()),
            "maintain": int((window_actions == MAINTAIN).sum()),
            "detections": int(detections[selected].sum()),
            "firms_washing": int(washing.any(axis=0).sum()),
            "wash_share_by_size": {
                name: float(washing[:, size_class == index].mean()) if washing.size and (size_class == index).any() else None
                for index, name in enumerate(SIZE_CLASS_NAMES)
            },
        },
        "impact": {
            "initial_share": float(impact[start]),
            "final_share": float(impact[end]),
            "initial_wealth_share": float(impact_wealth[start]),
            "final_wealth_share": float(impact_wealth[end]),
        },
        "pricing_slope": {
            "mean": float(finite_slope.mean()) if finite_slope.size else None,
            "observations": int(finite_slope.size),
        },
        "costs": {
            "initial_market_cap": initial_market_cap,
            "resource_cost": resource_cost,
            "real_improvement_cost": float(money[actions[selected] == IMPROVE].sum()),
            "washing_cost": float(money[actions[selected] == INFLATE].sum()),
            "fine_transfer": fines,
            "subsidy_transfer": subsidies,
            "private_net_outflow": resource_cost + fines - subsidies,
            "resource_cost_fraction_initial_cap": resource_cost / initial_market_cap,
            "private_net_outflow_fraction_initial_cap": (resource_cost + fines - subsidies) / initial_market_cap,
            "fine_transfer_fraction_initial_cap": fines / initial_market_cap,
            "subsidy_transfer_fraction_initial_cap": subsidies / initial_market_cap,
            "carry_in_resource_cost": float((fractions["cost_fraction"][carry_in] * prices[decision_days[carry_in]]).sum()),
            "unapplied_final_day_resource_cost": float(money[final_decision].sum()),
        },
        "efficiency": {
            "decay_adjusted_sd_per_resource_cost": float(gain.sum() / resource_cost) if resource_cost > 0 else None,
            "beneficial_raw_gain_per_resource_cost": float(domain.initial_sd * gain.sum() / resource_cost) if resource_cost > 0 else None,
            "undefined_reason": "zero_resource_cost" if resource_cost == 0 else None,
        },
        "diagnostics": {
            "status": "zero_elapsed_days" if end == start else (
                "zero_decision_observations" if not selected.any() else (
                    "missing_pricing_slope" if missing_slope else "ok"
                )
            ),
            "missing_size_classes": [name for name in SIZE_CLASS_NAMES if size[name]["count"] == 0],
            "pricing_slope_missing_observations": missing_slope,
            "finite_metrics": True,
        },
    }
    if domain.observation_link != "linear":
        # Raw-unit changes through the nonlinear observation link, per firm.
        true_raw = domain.to_raw(true[end]) - domain.to_raw(true[start])
        gain_raw = domain.to_raw(true[end]) - domain.to_raw(baseline)
        for key, raw in (("true_change", true_raw), ("decay_adjusted_gain", gain_raw)):
            metrics[key].update(equal_weight_raw=float(raw.mean()), fixed_start_cap_weight_raw=float(weights_start @ raw),
                                end_cap_weight_raw=float(weights_end @ raw))
        metrics["full_period_change"]["equal_weight_raw"] = float((domain.to_raw(true[end]) - domain.to_raw(true[0])).mean())
        for index, name in enumerate(SIZE_CLASS_NAMES):
            members = size_class == index
            size[name]["change_raw"] = float(true_raw[members].mean()) if members.any() else None
        if has_extremes:
            size["large_minus_small_raw"] = size["large"]["change_raw"] - size["small"]["change_raw"]
        if resource_cost > 0:
            metrics["efficiency"]["beneficial_raw_gain_per_resource_cost"] = float(domain.direction * gain_raw.sum() / resource_cost)
        metrics["raw_scale"] = ("empirical_quantile link: raw changes are differences of linked raw scores per firm; "
                                "SD metrics are in latent goodness units.")
    financing_fields = (result.financing_rate, result.financing_cost, result.financing_value_adjustment)
    if all(value is None for value in financing_fields):
        metrics["financing"] = {"status": "not_recorded"}
    else:
        if any(value is None or value.shape != true.shape or not np.isfinite(value).all()
               for value in financing_fields):
            raise ValueError("financing histories must be finite day-by-firm arrays")
        rate, fee, adjustment = financing_fields
        exposure = config.financing_exposure_fraction * prices[0]
        baseline_fee = exposure * rate[0] / TRADING_DAYS_PER_YEAR
        ledger_error = max(
            float(np.abs(fee[1:] - rate[1:] * exposure / TRADING_DAYS_PER_YEAR).max(initial=0)),
            float(np.abs(adjustment[1:] - (baseline_fee - fee[1:])).max(initial=0)),
        )
        metrics["financing"] = {
            "status": "modeled_accrual",
            "interpretation": "Accrued financing fees, not payments or cash-solvency evidence; value adjustment is relative to the initial contract.",
            "fixed_exposure": float(exposure.sum()),
            "accrued_cost": float(fee[start + 1:].sum()),
            "baseline_accrued_cost": float(baseline_fee.sum() * (end - start)),
            "value_adjustment": float(adjustment[start + 1:].sum()),
            "accrued_cost_fraction_initial_cap": float(fee[start + 1:].sum() / initial_market_cap),
            "mean_contract_rate": float(rate[start + 1:].mean()) if end > start else None,
            "maximum_ledger_error": ledger_error,
        }
    try:
        json.dumps(metrics, allow_nan=False)
    except ValueError as error:
        raise ValueError("metric computation produced a non-finite value") from error
    return metrics
