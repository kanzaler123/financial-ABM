"""HTTP boundary and observer isolation tests without a desktop dependency."""
from dataclasses import replace
import json
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from abm.config import load_stage1_config
from abm.web import ActiveRun, RunRegistry, create_app

ROOT = Path(__file__).resolve().parents[1]


def client_with_active_run(tmp_path):
    app = create_app(runs_root=tmp_path / 'runs', configs_root=ROOT / 'configs',
                     reports_root=tmp_path / 'reports', web_root=tmp_path / 'web', auth_token='audit')
    commands = queue.Queue(maxsize=1)
    app.state.registry.active['audit'] = SimpleNamespace(
        process=SimpleNamespace(is_alive=lambda: True), command_queue=commands)
    return TestClient(app, raise_server_exceptions=False), commands


@pytest.mark.parametrize('payload', [None, [], {'action': 'speed', 'value': 'fast'},
                                    {'action': 'speed', 'value': None},
                                    {'action': 'speed', 'value': 0},
                                    {'action': 'speed', 'value': -1},
                                    {'action': 'speed', 'value': True}])
def test_invalid_commands_return_422_and_are_not_enqueued(tmp_path, payload):
    client, commands = client_with_active_run(tmp_path)
    response = client.post('/api/runs/audit/commands', content=json.dumps(payload),
                           headers={'Authorization': 'Bearer audit', 'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert commands.empty()


@pytest.mark.parametrize('body', ['{bad', '{"action":"speed","value":NaN}',
                                  '{"action":"speed","value":Infinity}'])
def test_malformed_and_nonfinite_command_json_is_rejected(tmp_path, body):
    client, commands = client_with_active_run(tmp_path)
    response = client.post('/api/runs/audit/commands', content=body,
                           headers={'Authorization': 'Bearer audit'})
    assert response.status_code == 422
    assert commands.empty()


def test_full_command_queue_returns_backpressure(tmp_path):
    client, commands = client_with_active_run(tmp_path)
    commands.put({'action': 'pause'})
    response = client.post('/api/runs/audit/commands', json={'action': 'resume'},
                           headers={'Authorization': 'Bearer audit'})
    assert response.status_code == 429
    assert commands.qsize() == 1


def test_large_command_body_is_rejected(tmp_path):
    client, commands = client_with_active_run(tmp_path)
    response = client.post('/api/runs/audit/commands', content=' ' * (256 * 1024 + 1),
                           headers={'Authorization': 'Bearer audit'})
    assert response.status_code == 413
    assert commands.empty()


@pytest.mark.parametrize('document', ['[]', 'null', '{"trading_days":"invalid"}'])
def test_corrupt_saved_run_does_not_break_listing(tmp_path, document):
    registry = RunRegistry(tmp_path)
    path = tmp_path / 'corrupt'
    path.mkdir()
    (path / 'config.json').write_text('{}', encoding='utf-8')
    (path / 'summary.json').write_text(document, encoding='utf-8')
    assert registry.list_runs() == []


def test_dead_worker_is_not_reported_as_running(tmp_path):
    config = replace(load_stage1_config(ROOT / 'configs/stage1.json'),
                     population_size=10, trading_days=4, burn_in_days=0, announcements=())
    run = ActiveRun(name='crashed', config=config, output_dir=tmp_path,
                    process=SimpleNamespace(is_alive=lambda: False, exitcode=1),
                    telemetry_queue=queue.Queue(), command_queue=queue.Queue(),
                    completion_queue=queue.Queue(), status_queue=queue.Queue(),
                    state='running', current_day=2)
    status = run.metadata()
    assert status['state'] == 'failed'
    assert status['current_day'] == 2
    assert '1' in status['error']


def test_failed_gate_cannot_be_presented_as_formal_pass(tmp_path):
    reports = tmp_path / 'reports'
    report = reports / 'conflict'
    report.mkdir(parents=True)
    (report / 'mechanism_report.json').write_text(json.dumps({
        'protocol': {'stage': 'formal', 'version': 'audit'},
        'gate': {'passed': False, 'passed_checks': 18, 'total_checks': 22},
        'freeze_verification': {'verified': True},
    }), encoding='utf-8')
    (report / 'final_decision.json').write_text('{"stage1_complete":true}', encoding='utf-8')
    app = create_app(runs_root=tmp_path / 'runs', reports_root=reports,
                     web_root=tmp_path / 'web', auth_token='audit')
    response = TestClient(app).get('/api/reports', headers={'Authorization': 'Bearer audit'})
    assert response.status_code == 200
    assert response.json()[0]['stage1_complete'] is False
    assert response.json()[0]['scientific_status'] == 'formal_fail'
