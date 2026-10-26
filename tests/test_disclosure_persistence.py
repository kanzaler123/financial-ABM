"""Stage H (research contract section 25): permanent-plus-transitory dynamics and the sampled archive."""

import dataclasses

import numpy as np
import pytest

from abm.disclosure import runner
from abm.disclosure.config import TRADING_DAYS_PER_YEAR, MarketConfig
from abm.disclosure.domain import DomainSpec, load_domain
from abm.disclosure.game import IMPROVE, AttributeState, DisclosureMarket, expected_natural_path, real_persistence

MAIN = "configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json"


def _domain(**changes):
    return dataclasses.replace(load_domain(MAIN), **{"natural_decay": 0.0, **changes})


def _state(domain, size=50_000, seed=7):
    size_class = np.repeat([0, 1, 2], [size // 10, 3 * size // 10, size - 4 * (size // 10)])
    value = np.random.default_rng(seed + 1).lognormal(0.0, 0.9, size)
    return AttributeState.initialize(domain, size_class, np.random.default_rng(seed), intrinsic_value=value)


def test_new_options_validate_and_stay_out_of_default_dicts():
    base = load_domain(MAIN)
    assert "reversion_permanent_share" not in base.to_dict() and "reversion_improvement" not in base.to_dict()
    with pytest.raises(ValueError, match="requires natural_reversion"):
        dataclasses.replace(base, reversion_permanent_share=0.3)
    with pytest.raises(ValueError, match="requires natural_reversion"):
        dataclasses.replace(base, reversion_improvement="permanent")
    with pytest.raises(ValueError, match="below 1"):
        _domain(natural_reversion=0.5, reversion_permanent_share=1.0)
    with pytest.raises(ValueError, match="transitory"):
        _domain(natural_reversion=0.5, reversion_improvement="forever")
    q = _domain(natural_reversion=0.5, reversion_permanent_share=0.3, reversion_improvement="permanent")
    assert DomainSpec.from_dict(q.to_dict()) == q


def test_permanent_share_leaves_initial_draws_and_splits_the_variance():
    pure = _state(_domain(natural_reversion=0.5))
    split = _state(_domain(natural_reversion=0.5, reversion_permanent_share=0.4))
    assert pure.anchor is None and pure.transitory_share == 1.0
    np.testing.assert_array_equal(pure.goodness, split.goodness)
    np.testing.assert_array_equal(pure.cost_scale, split.cost_scale)
    mu, x = split.anchor, split.goodness - split.anchor
    assert abs(mu.var() - 0.4) < 0.01 and abs(x.var() - 0.6) < 0.01 and abs(np.corrcoef(mu, x)[0, 1]) < 0.02
    assert split.transitory_share == pytest.approx(0.6)


def test_zero_share_reverts_exactly_as_before():
    domain = _domain(natural_reversion=0.5)
    state = _state(domain, size=300)
    reference = state.goodness.copy()
    state.shock_rng = np.random.default_rng(11)
    shocks = np.random.default_rng(11)
    low, high = domain.goodness_bounds
    r = np.exp(-0.5 / TRADING_DAYS_PER_YEAR)
    for day in range(1, 40):
        state.advance(day, domain)
        shock = shocks.standard_normal(reference.size)
        mean = reference.mean()
        reference = np.clip(mean + r * (reference - mean) + np.sqrt(1 - r**2) * (shock - shock.mean()), low, high)
        np.testing.assert_array_equal(state.goodness, reference)


def test_transitory_part_keeps_its_variance_and_anchor_stays_put():
    domain = _domain(natural_reversion=1.0, reversion_permanent_share=0.4)
    state = _state(domain, size=20_000)
    mu = state.anchor.copy()
    mean = state.goodness.mean()
    for day in range(1, 2 * TRADING_DAYS_PER_YEAR):
        state.advance(day, domain)
    np.testing.assert_array_equal(state.anchor, mu)
    assert state.goodness.mean() == pytest.approx(mean, abs=1e-9)
    assert (state.goodness - mu).var() == pytest.approx(0.6, abs=0.02)
    assert state.goodness.var() == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("improvement", ["transitory", "permanent"])
def test_improvement_reverts_or_stays(improvement):
    domain = _domain(natural_reversion=1.0, reversion_permanent_share=0.3, reversion_improvement=improvement)
    treated, control = _state(domain, size=400), _state(domain, size=400)
    treated.shock_rng, control.shock_rng = np.random.default_rng(3), np.random.default_rng(3)
    treated.schedule(1, IMPROVE, np.array([5]), np.array([0.2]))
    days = TRADING_DAYS_PER_YEAR
    for day in range(1, days + 1):
        treated.advance(day, domain)
        control.advance(day, domain)
    gap = treated.goodness[5] - control.goodness[5]
    if improvement == "permanent":
        assert treated.anchor[5] - control.anchor[5] == pytest.approx(0.2)
        assert gap == pytest.approx(0.2, abs=1e-12)
    else:
        np.testing.assert_array_equal(treated.anchor, control.anchor)
        # The firm's deviation decays geometrically from the landing day; the mean keeps 1/n of it.
        retained = np.exp(-1.0 / TRADING_DAYS_PER_YEAR) ** (days - 1)
        assert gap == pytest.approx(0.2 / 400 + 0.2 * (1 - 1 / 400) * retained, rel=1e-6)


def test_persistence_and_expected_path_with_an_anchor():
    domain = _domain(natural_reversion=0.8, reversion_permanent_share=0.5)
    assert real_persistence(dataclasses.replace(domain, reversion_improvement="permanent"), 200, 60) is None
    np.testing.assert_array_equal(real_persistence(domain, 200, 60), real_persistence(_domain(natural_reversion=0.8), 200, 60))
    initial = np.array([1.0, -0.5, 0.2, 0.9])
    anchor = np.array([0.4, -0.4, 0.1, 0.2])
    path = expected_natural_path(domain, initial, np.array([0, 30, 252]), anchor=anchor)
    x = initial - anchor
    for row, day in zip(path, (0, 30, 252)):
        np.testing.assert_allclose(row, anchor + x.mean() + np.exp(-0.8 / 252) ** day * (x - x.mean()), rtol=1e-12)


def _config(days=1400):
    return MarketConfig(n_investors=24, n_firms=6, trading_days=days, burn_in_days=50, initial_impact_share=.2)


def test_market_records_anchors_and_sampled_archive(tmp_path):
    domain = _domain(natural_reversion=0.75, reversion_permanent_share=0.3)
    manifest = runner.run_and_archive(_config(), domain, tmp_path / "q", archive="sampled")
    assert "reversion_anchor" in manifest["compact_archive"]["omitted_result_arrays"]
    days = np.arange(runner.SAMPLE_START_DAY, 1401, runner.SAMPLE_INTERVAL_DAYS)
    assert manifest["sampled_archive"]["days"] == days.size == 3
    result = DisclosureMarket(_config(), domain).run()
    assert result.fingerprint() == manifest["result_fingerprint"]
    assert result.reversion_anchor.shape == result.true_goodness.shape
    # Without permanent improvements the anchors never move.
    np.testing.assert_array_equal(result.reversion_anchor, np.broadcast_to(result.reversion_anchor[0], result.reversion_anchor.shape))
    with np.load(tmp_path / "q" / "result.npz") as saved:
        assert "true_goodness" not in saved.files and "reversion_anchor" not in saved.files
        np.testing.assert_array_equal(saved["sampled_days"], days)
        np.testing.assert_array_equal(saved["sampled_public_signal"], result.public_signal[days])
        np.testing.assert_array_equal(saved["sampled_true_goodness"], result.true_goodness[days])
        np.testing.assert_array_equal(saved["sampled_reversion_anchor"], result.reversion_anchor[days])


def test_sampled_archive_of_a_default_run_keeps_the_compact_fingerprint(tmp_path):
    domain = load_domain(MAIN)
    compact = runner.run_and_archive(_config(300), domain, tmp_path / "c", archive="compact")
    sampled = runner.run_and_archive(_config(300), domain, tmp_path / "s", archive="sampled")
    assert compact["result_fingerprint"] == sampled["result_fingerprint"]
    assert sampled["compact_archive"]["omitted_result_arrays"] == list(runner.COMPACT_OMITTED)
    with np.load(tmp_path / "s" / "result.npz") as saved:
        assert saved["sampled_days"].size == 0
