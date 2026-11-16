"""Regression checks for cost accounting, paired checks and zero-parameter controls."""

import json
from dataclasses import replace

import numpy as np
import pytest

from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.financing import financing_discount_factors
from abm.disclosure.experiments import run_experiment
from abm.disclosure.game import ASSURANCE_PROTOCOL, MAINTAIN, DisclosureMarket
from abm.disclosure.metrics import compute_metrics
from abm.disclosure.policy import PolicySchedule
from abm.disclosure.runner import load_result, run_identity, save_result


DOMAIN = replace(load_domain("configs/domains/esg.json"), natural_decay=0.0,
                 baseline_audit_probability=0.0, baseline_penalty=0.0)
CONFIG = MarketConfig(n_investors=20, n_firms=12, trading_days=90, burn_in_days=0,
                      initial_impact_share=0.0, rebalance_speed=0.0,
                      fundamental_drift=0.0, fundamental_volatility=0.0,
                      cash_flow_yield=0.0, noise_flow_scale=0.0,
                      financing_spread=0.0, financing_exposure_fraction=0.0,
                      market_maker_inventory_skew=0.0)


def assured_market(*, scope="all", days=90, fee=.001):
    config = replace(CONFIG, trading_days=days)
    policy = PolicySchedule.constant(DOMAIN).with_intervention(
        1, assurance_scope=scope, assurer_effort_cost=.004,
        assurer_liability=0.0, assurance_fee=fee,
    )
    return DisclosureMarket(config, DOMAIN, policy)


@pytest.mark.parametrize("scope", ["all", "large"])
def test_fee_debit_reconciles_to_private_outflow_without_counting_a_transfer_as_resources(scope):
    market = assured_market(scope=scope)
    result = market.run()
    metrics = compute_metrics(result, market.config, DOMAIN)
    count = CONFIG.n_firms if scope == "all" else int((market.firms.size_class == 0).sum())
    expected_fee = count * .001
    np.testing.assert_array_equal(result.actions, MAINTAIN)
    # With every other value change disabled, this is an independent account check.
    assert (result.market.intrinsic_values[0] - result.market.intrinsic_values[-1]).sum() == pytest.approx(expected_fee)
    assert metrics["costs"]["private_net_outflow"] == pytest.approx(expected_fee)
    assert metrics["costs"]["assurance_fee_transfer"] == pytest.approx(expected_fee)
    assert metrics["costs"]["resource_cost"] == 0.0
    assert metrics["costs"]["assurance_effort_cost"] == 0.0
    assert metrics["efficiency"]["undefined_reason"] == "zero_resource_cost"
    assert np.flatnonzero(result.assurance_fee.sum(axis=1)).tolist() == [80]


def test_provider_effort_is_a_resource_cost_but_not_an_extra_payment_by_the_firm():
    market = assured_market(fee=.002)
    for day in range(60):
        market.step(day)
    # A controlled provider quality: effort is .004 * .5**2 = .001 per check.
    # No firm takes an action, so resource costs and payments are independently known.
    market.assurance.quality[:] = .5
    result = market.run()
    costs = compute_metrics(result, market.config, DOMAIN)["costs"]
    assert costs["firm_resource_cost"] == 0.0
    assert costs["resource_cost"] == pytest.approx(.012)
    assert costs["assurance_effort_cost"] == pytest.approx(.012)
    assert costs["assurance_fee_transfer"] == pytest.approx(.024)
    assert costs["private_net_outflow"] == pytest.approx(.024)
    assert (result.market.intrinsic_values[0] - result.market.intrinsic_values[-1]).sum() == pytest.approx(.024)


@pytest.mark.parametrize("last_day", [79, 80, 81])
def test_check_day_commitments_and_next_day_debits_have_explicit_window_boundaries(last_day):
    market = assured_market(days=last_day)
    result = market.run()
    costs = compute_metrics(result, market.config, DOMAIN)["costs"]
    assert costs["assurance_fee_transfer"] == pytest.approx(.012 if last_day >= 80 else 0.)
    assert costs["unapplied_final_day_assurance_fee"] == pytest.approx(.012 if last_day == 80 else 0.)
    debit = float((result.market.intrinsic_values[0] - result.market.intrinsic_values[-1]).sum())
    assert debit == pytest.approx(.012 if last_day >= 81 else 0.)
    if last_day >= 80:
        after_check = compute_metrics(result, market.config, DOMAIN, start_day=80)["costs"]
        assert after_check["assurance_fee_transfer"] == 0.
        assert after_check["carry_in_assurance_fee"] == pytest.approx(.012)


def test_common_firms_receive_identical_checks_when_mandated_scope_changes():
    all_firms = assured_market(scope="all", days=600)
    large_only = assured_market(scope="large", days=600)
    for market in (all_firms, large_only):
        market._assurance_year(60, market.policy.intervention, lambda a: np.zeros(CONFIG.n_firms))
        market.assurance.quality[:] = .5
        for day in range(80, 600, 60):
            market.attributes.inflation[:] = .1
            market._assurance_checks(day)
    large = all_firms.firms.size_class == 0
    left = all_firms.assurance.history["assurance_corrections"][:, large]
    right = large_only.assurance.history["assurance_corrections"][:, large]
    assert left.any() and (~left[1:9]).any()
    np.testing.assert_array_equal(left, right)


def test_checks_stay_paired_on_the_same_calendar_day_after_different_policy_start_dates():
    earlier = assured_market(days=600)
    later = assured_market(days=600)
    for market, origin in ((earlier, 60), (later, 120)):
        market._assurance_year(origin, market.policy.intervention, lambda a: np.zeros(CONFIG.n_firms))
        market.assurance.quality[:] = .5
        for day in range(origin + 20, 600, 60):
            market.attributes.inflation[:] = .1
            market._assurance_checks(day)
    np.testing.assert_array_equal(earlier.assurance.history["assurance_corrections"][2:],
                                  later.assurance.history["assurance_corrections"][2:])


@pytest.mark.parametrize("rule", ["type_mean", "pairwise"])
def test_zero_imitation_fraction_keeps_types_and_does_not_consume_the_update_stream(rule):
    config = replace(CONFIG, initial_impact_share=.5, imitation_rule=rule, imitation_update_fraction=0.0)
    market = DisclosureMarket(config, DOMAIN)
    before = market.investors.investor_type.copy()
    impact = before == 1
    assert impact.any() and not impact.all()
    wealth = market.investors.wealth(market.prices)
    market.window_wealth = wealth / np.where(impact, 1.1, 1.0)
    market.learning_rng = np.random.default_rng(0)
    state = market.learning_rng.bit_generator.state
    market._imitate()
    np.testing.assert_array_equal(market.investors.investor_type, before)
    assert market.learning_rng.bit_generator.state == state


@pytest.mark.parametrize("first", [None, 0.0, .4, 1.0, np.array([.2, .8, .0])])
def test_zero_discount_fine_is_the_probability_of_ever_being_audited(first):
    later = np.array([0., .25, 1.])
    with np.errstate(invalid="raise", divide="raise"):
        factors = financing_discount_factors(
            np.zeros(3), decision_day=60, horizon_days=240, repricing_interval=60,
            period_days=60, implementation_delay=120, disclosure_delay=20,
            audit_probability=later, first_audit_probability=first,
        )
    first_probability = 0. if first is None else np.broadcast_to(first, later.shape)[0]
    np.testing.assert_allclose(factors[2], [first_probability, 1., 1.], atol=1e-15)
    assert all(np.isfinite(value).all() for value in factors)


def test_zero_audit_targeting_matches_uniform_in_a_complete_market_run():
    config = replace(CONFIG, initial_impact_share=.5, rebalance_speed=.5,
                     financing_spread=.05, financing_exposure_fraction=.5, firm_discount_rate=0.)
    policy = PolicySchedule.constant(DOMAIN)
    baseline = DisclosureMarket(config, DOMAIN, policy).run()
    with np.errstate(invalid="raise", divide="raise"):
        targeted = DisclosureMarket(config, DOMAIN, policy.with_intervention(1, audit_allocation="value")).run()
    np.testing.assert_array_equal(targeted.actions, baseline.actions)
    np.testing.assert_allclose(targeted.amounts, baseline.amounts, atol=1e-14)
    np.testing.assert_allclose(targeted.market.prices, baseline.market.prices, atol=1e-14)


def test_assurance_cost_archives_round_trip_and_old_archives_remain_explicitly_incomplete(tmp_path):
    market = assured_market()
    result = market.run()
    path = tmp_path / "full.npz"
    save_result(path, result)
    restored = load_result(path)
    assert restored.fingerprint() == result.fingerprint()
    np.testing.assert_array_equal(restored.assurance_fee, result.assurance_fee)
    np.testing.assert_array_equal(restored.assurance_effort_cost, result.assurance_effort_cost)

    old_path = tmp_path / "legacy.npz"
    save_result(old_path, result, omit=("assurance_fee", "assurance_effort_cost"))
    old = load_result(old_path)
    assert old.assurance_fee is None and old.assurance_effort_cost is None
    assert old.fingerprint() == replace(result, assurance_fee=None, assurance_effort_cost=None).fingerprint()
    metrics = compute_metrics(old, market.config, DOMAIN)
    assert metrics["costs"]["assurance_costs_status"] == "not_recorded"
    assert metrics["costs"]["private_net_outflow"] is None
    assert metrics["costs"]["resource_cost"] is None
    assert metrics["efficiency"]["undefined_reason"] == "assurance_costs_not_recorded"


def test_new_assurance_runs_identify_the_changed_protocol_without_changing_plain_results(tmp_path):
    market = assured_market()
    identity = run_identity(market.config, DOMAIN, market.policy)
    # The protocol is manifest metadata; source hashes already version identity.
    assert "assurance_protocol" not in identity
    plain_policy = PolicySchedule.constant(DOMAIN)
    assert "assurance_protocol" not in run_identity(CONFIG, DOMAIN, plain_policy)
    result = DisclosureMarket(CONFIG, DOMAIN, plain_policy).run()
    assert result.assurance_fee is None and result.assurance_effort_cost is None
    path = tmp_path / "plain.npz"
    save_result(path, result)
    with np.load(path) as arrays:
        assert "assurance_fee" not in arrays.files and "assurance_effort_cost" not in arrays.files
    assert load_result(path).fingerprint() == result.fingerprint()


def test_assurance_experiment_runs_and_resumes_with_its_frozen_identities(tmp_path):
    domain_path = tmp_path / "domain.json"
    domain_path.write_text(json.dumps(DOMAIN.to_dict()), encoding="utf-8")
    policy = assured_market().policy.intervention.to_dict()
    spec = {
        "name": "assurance-accounting-regression", "purpose": "development",
        "domain": "domain.json", "market": CONFIG.to_dict(), "seeds": [101, 102],
        "scenarios": [{"name": "base"}, {"name": "assured", "policy": {
            "start_day": 1, "changes": policy}}],
        "comparisons": [{"name": "fee_difference", "weights": {"assured": 1, "base": -1}, "primary": True}],
        "analysis": {"primary_metric": "costs.private_net_outflow",
                     "metrics": ["costs.private_net_outflow", "costs.resource_cost"], "confidence_level": .95},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "experiment"
    run_experiment(spec_path, output, workers=1)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete_trials"] == summary["planned_trials"] == 4
    for seed in spec["seeds"]:
        manifest = json.loads((output / f"assured/seed_{seed}/attempt_001/manifest.json").read_text(encoding="utf-8"))
        assert manifest["assurance_protocol"] == ASSURANCE_PROTOCOL
    paired = summary["comparisons"][0]["metrics"]
    np.testing.assert_allclose(paired["costs.private_net_outflow"]["paired_values"], [.012, .012])
    np.testing.assert_array_equal(paired["costs.resource_cost"]["paired_values"], [0., 0.])
    run_experiment(spec_path, output, workers=1, resume=True)
    assert len(list(output.glob("*/seed_*/attempt_*"))) == 4
