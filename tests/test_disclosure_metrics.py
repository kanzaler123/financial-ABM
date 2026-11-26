import json
from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import IMPROVE, INFLATE, MAINTAIN, DisclosureResult
from abm.disclosure.harness import CrossSectionResult
from abm.disclosure.metrics import compute_metrics


DOMAIN = replace(load_domain("configs/domains/esg.json"), natural_decay=2.52)
CONFIG = MarketConfig(n_firms=3, trading_days=4, burn_in_days=2, firm_decision_interval=2)


@pytest.fixture
def result():
    prices = np.array([[1., 1., 1.], [2., 1., 1.], [3., 1., 1.], [2., 2., 2.], [1., 1., 3.]])
    market = CrossSectionResult(
        burn_in_days=2,
        prices=prices,
        intrinsic_values=prices.copy(),
        size_class=np.array([0, 1, 2]),
        equity_share=np.ones(5),
        mean_depth_fraction=np.ones(5),
        gross_volume=np.zeros(5),
        price_cap_hits=np.zeros(5, dtype=np.int64),
        total_cash=np.ones(5),
        maximum_share_error=np.zeros(5),
    )
    return DisclosureResult(
        market=market,
        true_goodness=np.array([[0., 0., 0.], [0., 0., 0.], [1., -1., .5], [1.1, -.9, .7], [1.2, -.8, .9]]),
        inflation=np.array([[0., 0., 0.], [0., 0., 0.], [.2, .1, 0.], [.3, .1, 0.], [.4, .2, .1]]),
        impact_share=np.array([.2, .2, .3, .4, .5]),
        impact_wealth_share=np.array([.1, .2, .25, .3, .4]),
        pricing_slope=np.array([np.nan, np.nan, -.01, np.nan, -.03]),
        actions=np.array([[MAINTAIN] * 3, [IMPROVE, INFLATE, MAINTAIN], [IMPROVE, INFLATE, MAINTAIN]]),
        amounts=np.zeros((3, 3)),
        detections=np.zeros((3, 3), dtype=np.int64),
        cost_fraction=np.array([[0., 0., 0.], [.4, .5, 0.], [.1, .2, 0.]]),
        penalty_fraction=np.array([[0., 0., 0.], [.1, 0., 0.], [.05, 0., 0.]]),
        subsidy_fraction=np.array([[0., 0., 0.], [.05, 0., 0.], [.02, 0., 0.]]),
    )


def test_change_weights_are_fixed_at_window_start_and_end_weights_are_separate(result):
    metrics = compute_metrics(result, CONFIG, DOMAIN)
    change = metrics["true_change"]
    assert change["equal_weight_sd"] == pytest.approx(.8 / 3)
    assert change["equal_weight_raw"] == pytest.approx(4.)
    assert change["fixed_start_cap_weight_sd"] == pytest.approx(.24)
    assert change["fixed_start_cap_weight_raw"] == pytest.approx(3.6)
    assert change["end_cap_weight_sd"] == pytest.approx(.32)
    assert change["end_cap_weight_raw"] == pytest.approx(4.8)
    assert metrics["size"]["large"]["change_sd"] == pytest.approx(.2)
    assert metrics["size"]["small"]["change_sd"] == pytest.approx(.4)
    assert metrics["size"]["large_minus_small_sd"] == pytest.approx(-.2)


def test_full_period_and_short_terminal_summary_use_day_zero_not_burn_in(result):
    metrics = compute_metrics(result, CONFIG, DOMAIN)
    assert metrics["full_period_change"]["equal_weight_sd"] == pytest.approx(1.3 / 3)
    assert metrics["full_period_change"]["equal_weight_raw"] == pytest.approx(6.5)
    assert metrics["terminal_year"] == {
        "start_day": 0, "observations": 5, "complete_year": False,
        "change_from_initial_sd": pytest.approx(.18),
        "decay_adjusted_from_initial_sd": pytest.approx(.20),
        "inflation_mean_sd": pytest.approx(1.4 / 15),
    }
    zero_window = compute_metrics(result, CONFIG, DOMAIN, start_day=4)
    assert zero_window["full_period_change"] == metrics["full_period_change"]
    assert zero_window["terminal_year"] == metrics["terminal_year"]


def test_terminal_summary_uses_exactly_last_252_state_days(result):
    true = np.full((254, 3), .5)
    true[0], true[1] = 1., 2.
    inflation = np.full((254, 3), .1)
    inflation[:2] = 1.
    decisions = np.full((127, 3), MAINTAIN)
    extended = replace(
        result, market=replace(result.market, prices=np.ones((254, 3))),
        true_goodness=true, inflation=inflation,
        impact_share=np.full(254, .2), impact_wealth_share=np.full(254, .2),
        pricing_slope=np.full(254, np.nan), actions=decisions, detections=np.zeros((127, 3), dtype=np.int64),
        cost_fraction=np.zeros((127, 3)), penalty_fraction=np.zeros((127, 3)),
        subsidy_fraction=np.zeros((127, 3)),
    )
    metrics = compute_metrics(extended, replace(CONFIG, trading_days=253), DOMAIN)
    assert metrics["terminal_year"] == {
        "start_day": 2, "observations": 252, "complete_year": True,
        "change_from_initial_sd": pytest.approx(-.5),
        "decay_adjusted_from_initial_sd": pytest.approx(-.5 + .01 * 127.5),
        "inflation_mean_sd": pytest.approx(.1),
    }


def test_costs_use_decision_prices_exclude_warmup_and_keep_transfers_separate(result):
    metrics = compute_metrics(result, CONFIG, DOMAIN)
    assert metrics["window"]["decision_days"] == [4]
    costs = metrics["costs"]
    assert costs["resource_cost"] == pytest.approx(.3)
    assert costs["real_improvement_cost"] == pytest.approx(.1)
    assert costs["washing_cost"] == pytest.approx(.2)
    assert costs["fine_transfer"] == pytest.approx(.05)
    assert costs["subsidy_transfer"] == pytest.approx(.02)
    assert costs["private_net_outflow"] == pytest.approx(.33)
    assert costs["initial_market_cap"] == pytest.approx(3.)
    assert costs["resource_cost_fraction_initial_cap"] == pytest.approx(.1)
    assert costs["private_net_outflow_fraction_initial_cap"] == pytest.approx(.11)
    assert costs["fine_transfer_fraction_initial_cap"] == pytest.approx(.05 / 3)
    assert costs["subsidy_transfer_fraction_initial_cap"] == pytest.approx(.02 / 3)
    assert costs["carry_in_resource_cost"] == pytest.approx(1.7)
    assert costs["unapplied_final_day_resource_cost"] == pytest.approx(.3)
    assert metrics["efficiency"]["decay_adjusted_sd_per_resource_cost"] == pytest.approx(.86 / .3)
    assert metrics["efficiency"]["beneficial_raw_gain_per_resource_cost"] == pytest.approx(43.)
    all_days = compute_metrics(result, CONFIG, DOMAIN, start_day=0)
    assert all_days["costs"]["resource_cost"] == pytest.approx(2.)
    assert all_days["pricing_slope"]["mean"] == pytest.approx(-.02)
    assert all_days["pricing_slope"]["observations"] == 2


def test_natural_decay_counterfactual_respects_lower_bound(result):
    low = DOMAIN.goodness_bounds[0]
    true = result.true_goodness.copy()
    true[2] = [low + .005, 0., 0.]
    true[3] = [low, 0., .05]
    true[4] = [low, 0., .1]
    metrics = compute_metrics(replace(result, true_goodness=true), CONFIG, DOMAIN)
    assert metrics["decay_adjusted_gain"]["equal_weight_sd"] == pytest.approx(.14 / 3)
    assert metrics["size"]["large"]["decay_adjusted_gain_sd"] == pytest.approx(0.)


def test_terminal_decay_adjustment_starts_at_day_zero_and_respects_lower_bound(result):
    low = DOMAIN.goodness_bounds[0]
    true = result.true_goodness.copy()
    true[0, 0] = low + .01
    metrics = compute_metrics(replace(result, true_goodness=true), CONFIG, DOMAIN)
    # Firm 0's no-action path sits on the bound from day 1; the others decay by .01 a day from zero.
    assert metrics["terminal_year"]["decay_adjusted_from_initial_sd"] == pytest.approx((2.9 - 4 * low) / 15)


def test_lower_is_better_keeps_raw_change_signed_and_efficiency_beneficial(result):
    metrics = compute_metrics(result, CONFIG, replace(DOMAIN, direction=-1))
    assert metrics["true_change"]["equal_weight_raw"] == pytest.approx(-4.)
    assert metrics["full_period_change"]["equal_weight_raw"] == pytest.approx(-6.5)
    assert metrics["size"]["large_minus_small_raw"] == pytest.approx(3.)
    assert metrics["inflation"]["mean_absolute_raw_gap"] == pytest.approx(15 * 1.4 / 9)
    assert metrics["inflation"]["final_absolute_raw_gap"] == pytest.approx(3.5)
    assert metrics["inflation"]["mean_range_fraction"] == pytest.approx(15 * 1.4 / 900)
    assert metrics["efficiency"]["beneficial_raw_gain_per_resource_cost"] == pytest.approx(43.)


@pytest.mark.parametrize("direction", [1, -1])
def test_quantile_link_reports_raw_changes_through_the_link_and_keeps_sd_metrics(result, direction):
    linked = replace(DOMAIN, direction=direction, initial_mean=50.0, initial_sd=10.0,
                     initial_empirical_distribution=((40.0, 1), (60.0, 1)), observation_link="empirical_quantile")
    linear = compute_metrics(result, CONFIG, replace(DOMAIN, direction=direction))
    metrics = compute_metrics(result, CONFIG, linked)
    true = result.true_goodness
    raw = linked.to_raw(true[4]) - linked.to_raw(true[2])
    assert metrics["true_change"]["equal_weight_sd"] == linear["true_change"]["equal_weight_sd"]
    assert metrics["true_change"]["equal_weight_raw"] == pytest.approx(raw.mean())
    assert metrics["true_change"]["fixed_start_cap_weight_raw"] == pytest.approx(np.array([.6, .2, .2]) @ raw)
    assert metrics["full_period_change"]["equal_weight_raw"] == pytest.approx((linked.to_raw(true[4]) - linked.to_raw(true[0])).mean())
    assert [metrics["size"][name]["change_raw"] for name in ("large", "mid", "small")] == pytest.approx(list(raw))
    assert metrics["size"]["large_minus_small_raw"] == pytest.approx(raw[0] - raw[2])
    assert np.sign(metrics["efficiency"]["beneficial_raw_gain_per_resource_cost"]) == 1
    assert "latent" in metrics["raw_scale"] and "raw_scale" not in linear


def test_investor_shares_and_slope_use_requested_window(result):
    metrics = compute_metrics(result, CONFIG, DOMAIN)
    assert metrics["impact"] == {
        "initial_share": .3, "final_share": .5,
        "initial_wealth_share": .25, "final_wealth_share": .4,
    }
    assert metrics["pricing_slope"] == {"mean": -.03, "observations": 1}
    assert metrics["inflation"]["initial_mean_sd"] == pytest.approx(.1)
    assert metrics["inflation"]["final_mean_sd"] == pytest.approx(.7 / 3)


def test_zero_resource_cost_does_not_use_fines_or_subsidies_as_denominator(result):
    metrics = compute_metrics(replace(result, cost_fraction=np.zeros((3, 3))), CONFIG, DOMAIN)
    assert metrics["costs"]["fine_transfer"] > 0
    assert metrics["costs"]["subsidy_transfer"] > 0
    assert metrics["efficiency"] == {
        "decay_adjusted_sd_per_resource_cost": None,
        "beneficial_raw_gain_per_resource_cost": None,
        "undefined_reason": "zero_resource_cost",
    }
    json.dumps(metrics, allow_nan=False)


def test_empty_window_and_missing_size_groups_have_explicit_observation_counts(result):
    market = replace(result.market, size_class=np.zeros(3, dtype=np.int64))
    metrics = compute_metrics(replace(result, market=market), CONFIG, DOMAIN, start_day=4)
    assert metrics["window"]["elapsed_days"] == 0
    assert metrics["window"]["decision_observations"] == 0
    assert metrics["pricing_slope"] == {"mean": None, "observations": 0}
    assert metrics["size"]["small"]["change_sd"] is None
    assert metrics["size"]["large_minus_small_sd"] is None
    assert metrics["diagnostics"]["missing_size_classes"] == ["mid", "small"]
    assert metrics["diagnostics"]["status"] == "zero_elapsed_days"
    json.dumps(metrics, allow_nan=False)


def test_missing_slope_is_flagged_without_nan(result):
    metrics = compute_metrics(replace(result, pricing_slope=np.full(5, np.nan)), CONFIG, DOMAIN)
    assert metrics["pricing_slope"]["mean"] is None
    assert metrics["diagnostics"]["pricing_slope_missing_observations"] == 1
    assert metrics["diagnostics"]["status"] == "missing_pricing_slope"
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("field", ["true_goodness", "inflation", "impact_share", "cost_fraction"])
def test_nonfinite_required_observations_are_rejected(result, field):
    values = getattr(result, field).copy()
    values[-1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(replace(result, **{field: values}), CONFIG, DOMAIN)


@pytest.mark.parametrize("start", [-1, 5, 1.5, True])
def test_invalid_window_is_rejected(result, start):
    with pytest.raises(ValueError, match="start_day"):
        compute_metrics(result, CONFIG, DOMAIN, start_day=start)


def test_assurance_concentration_metrics_handle_quality_ties_and_sorting(result):
    # Case 1: all 4 providers tied at q=0.0 -> raw share is 1.0, excess is exactly 0.0, strict is None
    tied = replace(
        result,
        assurer_assignment=np.array([[-1, -1, -1], [0, 1, 2], [0, 1, 2]], dtype=np.int64),
        assurer_quality=np.zeros((3, 4), dtype=np.float64),
    )
    m_tied = compute_metrics(tied, CONFIG, DOMAIN)["assurance"]
    assert m_tied["assured_wash_decisions"] == 1
    assert m_tied["raw_laxest_wash_share"] == pytest.approx(1.0)
    assert m_tied["chance_laxest_wash_share"] == pytest.approx(1.0)
    assert m_tied["excess_laxest_wash_concentration"] == pytest.approx(0.0)
    assert m_tied["strict_laxest_wash_share"] is None
    assert m_tied["client_minus_unweighted_quality_gap"] == pytest.approx(0.0)

    # Case 2: provider 0 is uniquely laxest (0.0 vs 0.2, 0.2, 0.2); inflating firm 1 switches to provider 0
    sorted_res = replace(
        result,
        assurer_assignment=np.array([[-1, -1, -1], [0, 0, 1], [0, 0, 1]], dtype=np.int64),
        assurer_quality=np.array([[0., 0., 0., 0.], [0.0, 0.2, 0.2, 0.2], [0.0, 0.2, 0.2, 0.2]], dtype=np.float64),
    )
    m_sort = compute_metrics(sorted_res, CONFIG, DOMAIN)["assurance"]
    assert m_sort["raw_laxest_wash_share"] == pytest.approx(1.0)
    assert m_sort["chance_laxest_wash_share"] == pytest.approx(0.25)
    assert m_sort["excess_laxest_wash_concentration"] == pytest.approx(0.75)
    assert m_sort["strict_laxest_wash_share"] == pytest.approx(1.0)
    # 2 of 3 firms at 0.0, 1 at 0.2 -> client_q = 0.2/3 < unweighted_q = 0.6/4 = 0.15
    assert m_sort["client_minus_unweighted_quality_gap"] == pytest.approx(0.2 / 3.0 - 0.15)
