"""Regression cases from the September 2026 correctness audit."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import queue
from unittest.mock import Mock

import numpy as np
import pytest

from abm.config import Announcement, Stage1Config, load_stage1_config
from abm.harness import SimulationResult, build_fundamental_path, run_stage1
from abm.learning import LogitImitator, PerformanceWindow
from abm.manifest import canonical_json_bytes
from abm import mechanism_validation as validation
from abm.population import TraderPopulation
from abm.run_service import controlled_run_worker
from abm.settlement import SettlementEngine
from abm.stage1_freeze import build_freeze_manifest, verify_freeze_manifest, write_freeze_manifest

ROOT = Path(__file__).resolve().parents[1]


def small_config(**changes):
    values = dict(population_size=12, trading_days=4, burn_in_days=0,
                  announcements=(), learning_enabled=False)
    values.update(changes)
    return replace(load_stage1_config(ROOT / 'configs/stage1.json'), **values)


@pytest.mark.parametrize('strategy', ('value', 'trend', 'noise'))
def test_single_strategy_learning_is_json_serializable(strategy, tmp_path):
    config = small_config(learning_enabled=True, learning_interval=1,
                          strategy_shares={name: float(name == strategy)
                                           for name in ('value', 'trend', 'noise')})
    result = run_stage1(config)
    assert result.fingerprint() == run_stage1(config).fingerprint()
    audit = result.learning_audits[-1].to_dict()
    json.dumps(audit, allow_nan=False)
    for name, score in audit['mean_fitness_by_strategy'].items():
        if name != strategy:
            assert score is None
            assert audit['choice_probabilities'][name] == 0.0
    assert np.isfinite(validation.result_metrics(result)['learning_fitness_alignment'])
    result.write(tmp_path / strategy, config)
    assert (tmp_path / strategy / 'stage1_learning.png').is_file()


@pytest.mark.parametrize('drift', (-0.01, 0.01))
def test_zero_volatility_keeps_deterministic_drift(drift):
    config = small_config(fundamental_volatility=0.0, fundamental_drift=drift,
                          announcements=(Announcement(2, 3.0, 'test'),))
    rng = Mock()
    path = build_fundamental_path(config, rng=rng)
    expected = [config.initial_fundamental]
    for day in range(1, config.trading_days + 1):
        expected.append(expected[-1] * np.exp(drift) + (3.0 if day == 2 else 0.0))
    np.testing.assert_allclose(path, expected)
    assert not rng.mock_calls


@pytest.mark.parametrize('field,value', [
    ('population_size', 12.5), ('population_size', True), ('trading_days', 4.5),
    ('seed', 1.5), ('seed', True), ('learning_interval', 1.5),
    ('burn_in_days', 0.5), ('trend_short_lookback_min', 5.5),
    ('learning_enabled', 'false'), ('learning_enabled', 1),
    ('initial_price', True), ('fundamental_drift', '0'),
])
def test_config_rejects_wrong_scalar_types(field, value):
    with pytest.raises((ValueError, TypeError), match=field):
        small_config(**{field: value})


@pytest.mark.parametrize('day', (1.5, True, '2'))
def test_announcement_day_is_not_silently_coerced(day):
    payload = small_config().to_dict()
    payload['announcements'] = [{'day': day, 'fundamental_delta': 1, 'source': 'test'}]
    with pytest.raises((ValueError, TypeError), match='day'):
        Stage1Config.from_dict(payload)


def population(cash=(1000., 1000.), positions=(1., 1.)):
    return TraderPopulation(cash=np.array(cash, dtype=float),
                            positions=np.array(positions, dtype=float),
                            strategies=np.array([0, 1], dtype=np.int8),
                            risk_aversion=np.ones(2))


@pytest.mark.parametrize('bad', (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize('field', ('submitted_orders', 'minimum_positions',
                                  'market_maker_cash', 'cash', 'positions'))
def test_settlement_rejects_nonfinite_values_without_mutation(field, bad):
    pop = population()
    kwargs = dict(population=pop, submitted_orders=np.zeros(2), execution_price=100.,
                  market_maker_cash=1000., market_maker_inventory=10.,
                  minimum_positions=np.zeros(2))
    if field in ('cash', 'positions'):
        getattr(pop, field)[0] = bad
    elif field == 'market_maker_cash':
        kwargs[field] = bad
    else:
        kwargs[field][0] = bad
    before_cash, before_positions = pop.cash.copy(), pop.positions.copy()
    with pytest.raises(ValueError):
        SettlementEngine(0.01).settle(**kwargs)
    np.testing.assert_array_equal(pop.cash, before_cash)
    np.testing.assert_array_equal(pop.positions, before_positions)


@pytest.mark.parametrize('fee', (0.0, 0.01))
def test_margin_covers_fund_other_sales_in_same_batch(fee):
    pop = population(cash=(1000., 0.), positions=(-1., 1.))
    initial_cash = pop.cash.sum()
    initial_shares = pop.positions.sum() + 10.
    result = SettlementEngine(fee).settle(
        population=pop, submitted_orders=np.array([0., -1.]), execution_price=100.,
        market_maker_cash=0., market_maker_inventory=10., minimum_positions=np.zeros(2))
    np.testing.assert_allclose(result.executed_orders, [1., -1.])
    assert result.market_maker_cash >= 0
    assert pop.cash.sum() + result.market_maker_cash == pytest.approx(initial_cash)
    assert pop.positions.sum() + result.market_maker_inventory == pytest.approx(initial_shares)


@pytest.mark.parametrize('status_queue', [queue.Queue(maxsize=1), Mock()])
def test_full_or_closed_status_queue_cannot_abort_simulation(status_queue, tmp_path, monkeypatch):
    config = small_config()
    if isinstance(status_queue, Mock):
        status_queue.put_nowait.side_effect = ValueError('queue closed')
    commands, completion = queue.Queue(), queue.Queue()
    commands.put({'action': 'detach'})
    monkeypatch.setattr(SimulationResult, 'write', lambda *args, **kwargs: tmp_path)
    controlled_run_worker(config.to_dict(), str(tmp_path), queue.Queue(maxsize=1),
                          commands, completion, status_queue)
    message = completion.get_nowait()
    assert message['state'] == 'completed'
    assert message['fingerprint'] == run_stage1(config).fingerprint()


def protocol():
    return validation.MechanismProtocol(version='audit', purpose='regression',
                                         seeds=(11,), excluded_development_seeds=(12,), criteria={})


@pytest.mark.parametrize('filename', ['settlement.py', 'learning.py', 'telemetry.py'])
def test_checkpoint_identity_covers_all_execution_dependencies(filename, monkeypatch):
    original = validation.sha256_file
    before = validation._protocol_identity(small_config(), protocol())
    monkeypatch.setattr(validation, 'sha256_file',
                        lambda path: 'changed' if path.name == filename else original(path))
    assert validation._protocol_identity(small_config(), protocol()) != before


@pytest.mark.parametrize('mutation', ('duplicate', 'unknown_seed', 'nonfinite', 'missing_metric'))
def test_invalid_checkpoint_rejected_before_any_new_simulations(mutation, tmp_path, monkeypatch):
    config, proto = small_config(), protocol()
    row = dict(scenario='full', seed=11, **{name: 0.0 for name in validation.METRIC_NAMES})
    rows = [row]
    if mutation == 'duplicate':
        rows.append(dict(row))
    elif mutation == 'unknown_seed':
        row['seed'] = 999
    elif mutation == 'nonfinite':
        row['daily_volatility'] = float('nan')
    else:
        row.pop('daily_volatility')
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps(dict(schema_version='1.0.0',
        identity=validation._protocol_identity(config, proto), rows=rows)), encoding='utf-8')
    runner = Mock(side_effect=AssertionError('must reject checkpoint before executing work'))
    monkeypatch.setattr(validation, '_run_metric_task', runner)
    with pytest.raises(ValueError, match='checkpoint'):
        validation.run_protocol(config, proto, checkpoint_path=checkpoint, resume=True)
    runner.assert_not_called()


def freeze_project(root):
    for name in ('src/abm/core.py', 'web/src/App.tsx', 'web/package.json',
                 'web/package-lock.json', 'pyproject.toml', 'requirements-lock.txt',
                 'configs/formal.json', 'configs/protocol.json'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')
    return build_freeze_manifest(root, formal_config=root / 'configs/formal.json',
                                formal_protocol=root / 'configs/protocol.json')


def test_freeze_rejects_incomplete_hash_inventory(tmp_path):
    manifest = freeze_project(tmp_path)
    manifest['sha256'].pop('src/abm/core.py')
    identity = {key: manifest[key] for key in ('schema_version', 'formal_config', 'formal_protocol', 'sha256')}
    manifest['freeze_id'] = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    path = write_freeze_manifest(tmp_path / 'freeze.json', manifest)
    with pytest.raises(ValueError, match='inventory'):
        verify_freeze_manifest(path, project_root=tmp_path)


@pytest.mark.parametrize('new_file', ('src/abm/new.py', 'src/abm/sub/module.py', 'web/src/components/New.tsx'))
def test_freeze_rejects_source_added_after_freezing(tmp_path, new_file):
    path = write_freeze_manifest(tmp_path / 'freeze.json', freeze_project(tmp_path))
    added = tmp_path / new_file
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_text('# new source\n', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        verify_freeze_manifest(path, project_root=tmp_path)


def test_extreme_temperature_does_not_remove_the_best_strategy():
    pop = population()
    window = PerformanceWindow.create(2)
    window.gross_return_sum[:] = [1.0, 2.0]
    window.gross_return_square_sum[:] = [1.0, 4.0]
    window.observations = 1
    audit = LogitImitator(temperature=1e-320, risk_penalty=0.0, update_fraction=1.0).update(
        day=1, population=pop, performance=window, rng=np.random.default_rng(1))
    assert audit.choice_probabilities == (0.0, 1.0, 0.0)
    assert np.all(pop.strategies == 1)


@pytest.mark.parametrize('rows', [None, []])
def test_gate_cannot_invent_pass_rates_without_seed_evidence(rows):
    all_rows = [dict(scenario=name, seed=11, **{metric: 0.0 for metric in validation.METRIC_NAMES})
                for name in validation.SCENARIO_NAMES]
    summary = validation._summarize(all_rows)
    with pytest.raises(ValueError, match='per-seed'):
        validation.evaluate_gate(summary, {}, rows)


def test_gate_rejects_summary_that_disagrees_with_seed_evidence():
    rows = [dict(scenario=name, seed=11, **{metric: 0.0 for metric in validation.METRIC_NAMES})
            for name in validation.SCENARIO_NAMES]
    summary = validation._summarize(rows)
    summary['full']['price_cap_hits']['maximum'] = 99.0
    with pytest.raises(ValueError, match='summary'):
        validation.evaluate_gate(summary, {}, rows)


@pytest.mark.parametrize('field, value', [
    ('seeds', [1.5]), ('seeds', [True]), ('seeds', [-1]), ('seeds', '12'),
    ('excluded_development_seeds', [2.5]), ('minimum_burn_in_days', 0.5),
    ('minimum_evaluation_days', True),
])
def test_protocol_integer_values_cannot_be_silently_coerced(tmp_path, field, value):
    payload = dict(version='audit', purpose='test', seeds=[11],
                   excluded_development_seeds=[12], criteria={})
    payload[field] = value
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises((ValueError, TypeError), match=field):
        validation.MechanismProtocol.from_json(path)


def test_missing_returns_do_not_join_nonadjacent_trading_days():
    from abm.french49 import return_statistics
    values = np.sin(np.arange(40, dtype=float))
    values[[3, 7, 9, 12, 17]] = np.nan
    valid_pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    expected = np.corrcoef(values[:-1][valid_pairs], values[1:][valid_pairs])[0, 1]
    stats = return_statistics(values)
    assert stats['observations'] == 35
    assert stats['return_autocorrelation_lag1'] == pytest.approx(expected)


def test_calibration_excludes_the_configured_burn_in():
    from abm.calibration import simulate_seed_group
    config = small_config(trading_days=12, burn_in_days=3)
    paths = simulate_seed_group(config, seeds=(11,), trading_days=12, common_seed=12)
    assert paths['11'].shape == (9,)
