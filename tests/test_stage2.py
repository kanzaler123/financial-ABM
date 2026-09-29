"""Causal graph and delayed tabular learning acceptance tests."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import numpy as np

from abm.information import InformationGraph
from abm.schemas import Edge, Message
from abm.tabular import CompanyQLearner, ContextualBandit, TRADER_THRESHOLDS, ternary_states

ROOT = Path(__file__).resolve().parents[1]


def _five_nodes():
    fixture = json.loads((ROOT / "data/synthetic/stage2_five_nodes.json").read_text(encoding="utf-8"))
    graph = InformationGraph(tuple(fixture["nodes"]), tuple(Edge.from_dict(x) for x in fixture["edges"]))
    return fixture, graph, Message.from_dict(fixture["message"])


def test_five_node_source_paths_arrival_and_trust_replay_exactly():
    fixture, graph, message = _five_nodes()
    graph.publish(message)
    delivered = []
    for day in range(5):
        now = message.available_at + timedelta(days=day)
        arrived = graph.advance(now)
        delivered.extend(arrived)
        assert all(d.available_at <= now for d in arrived)
        for node in graph.nodes:
            assert all(d.available_at <= now for d in graph.visible(node, now))
    assert len(delivered) == 4
    for actual, expected in zip(delivered, fixture["expected"], strict=True):
        assert actual.source_id == "company-0"
        assert actual.target_id == expected["target"]
        assert actual.path == tuple(expected["path"])
        assert actual.available_at == message.available_at + timedelta(days=expected["delay"])
        assert actual.credibility == pytest.approx(expected["credibility"])
    _, replay, _ = _five_nodes()
    replay.publish(message)
    assert replay.advance(message.available_at + timedelta(days=4)) == tuple(delivered)


def test_scope_blocks_unauthorized_nodes_and_future_messages():
    _, graph, message = _five_nodes()
    private = replace(message, target_scope=("trader-0", "trader-2"),
                      available_at=message.available_at + timedelta(days=5))
    graph.publish(private)
    assert graph.advance(message.available_at + timedelta(days=4)) == ()
    assert graph.visible("trader-0", graph.current_time) == ()
    arrived = graph.advance(message.available_at + timedelta(days=10))
    assert {d.target_id for d in arrived} == {"trader-0", "trader-2"}
    assert graph.visible("trader-1", graph.current_time) == ()
    assert graph.visible("trader-3", graph.current_time) == ()


def test_zero_delay_cycles_and_equal_arrivals_do_not_duplicate_news():
    _, graph, message = _five_nodes()
    graph = InformationGraph(graph.nodes, tuple(replace(e, delay_days=0) for e in reversed(graph.edges)))
    graph.publish(message)
    arrived = graph.advance(message.available_at)
    assert len(arrived) == 4
    assert len({d.target_id for d in arrived}) == 4
    assert graph.advance(message.available_at + timedelta(days=10)) == ()
    with pytest.raises(ValueError, match="already"):
        graph.publish(message)


def test_graph_observers_cannot_mutate_pending_or_recorded_payloads():
    _, graph, message = _five_nodes()
    graph.publish(message)
    message.payload["log_change"] = 100.0
    first = graph.advance(message.available_at)
    first[0].payload["log_change"] = 200.0
    assert graph.visible("trader-1", graph.current_time)[0].payload["log_change"] == 0.03
    later = graph.advance(message.available_at + timedelta(days=4))
    assert all(d.payload["log_change"] == 0.03 for d in later)
    with pytest.raises(ValueError, match="backwards"):
        graph.advance(message.available_at)
    with pytest.raises(ValueError, match="clock"):
        graph.visible("trader-0", graph.current_time + timedelta(days=1))


def test_dynamic_weights_use_past_trades_and_apply_at_release():
    _, original, message = _five_nodes()
    edge = replace(original.edges[0], source_id="trader-0", target_id="trader-1",
                   delay_days=0, trust=1.0)
    graph = InformationGraph(("trader-0", "trader-1"), (edge,))
    graph.publish(replace(message, source_id="trader-0",
                          available_at=message.available_at + timedelta(days=2)))
    graph.advance(message.available_at)
    history = np.array([[1.0, -1.0]])
    updates = graph.adapt_from_trades(graph.nodes, history, (message.available_at,),
                                     message.available_at + timedelta(days=1), rate=0.5)
    assert updates[0]["after_weight"] == 0.5
    assert graph.active_edges(message.available_at)[0].weight == 1.0
    delivered = graph.advance(message.available_at + timedelta(days=2))
    assert delivered[0].credibility == 0.5
    with pytest.raises(ValueError, match="observed"):
        graph.adapt_from_trades(graph.nodes, history, (graph.current_time + timedelta(days=1),),
                               graph.current_time + timedelta(days=2))


def test_inflight_delays_and_trust_are_not_rewritten_by_new_edges():
    _, original, message = _five_nodes()
    edge = replace(original.edges[0], source_id="trader-0", target_id="trader-1", delay_days=3, trust=1.0)
    graph = InformationGraph(("trader-0", "trader-1"), (edge,))
    graph.publish(replace(message, source_id="trader-0"))
    assert graph.advance(message.available_at) == ()
    graph.adapt_from_trades(graph.nodes, np.array([[1.0, -1.0]]), (message.available_at,),
                           message.available_at + timedelta(days=1), rate=1.0)
    assert graph.advance(message.available_at + timedelta(days=3))[0].credibility == 1.0


def test_four_ternary_features_cover_exactly_81_states():
    from itertools import product
    representatives = [[low - 1, (low + high) / 2, high + 1] for low, high in TRADER_THRESHOLDS]
    features = np.array(list(product(*representatives)))
    assert np.array_equal(ternary_states(features, TRADER_THRESHOLDS), np.arange(81))


def test_bandit_recovers_known_best_action_in_all_contexts():
    learner = ContextualBandit(epsilon=0.2)
    ids = np.arange(243)
    groups, states = ids // 81, ids % 81
    optimum = (states + groups) % 4
    rng = np.random.default_rng(411)
    for _ in range(250):
        actions = learner.choose(groups, states, rng)
        rewards = (actions == optimum).astype(float)
        learner.update(ids, groups, states, actions, rewards)
    assert np.array_equal(learner.values[groups, states].argmax(axis=1), optimum)
    assert np.all(learner.counts > 0)


def test_q_learning_recovers_discounted_optimum_in_all_contexts():
    learner = CompanyQLearner(epsilon=0.3, learning_rate=0.2, discount=0.5)
    ids = np.arange(243)
    groups, states = ids // 81, ids % 81
    optimum = (states + groups) % 3
    rng = np.random.default_rng(412)
    for _ in range(500):
        actions = learner.choose(groups, states, rng)
        rewards = (actions == optimum).astype(float)
        learner.update(ids, groups, states, actions, rewards, states)
    assert np.array_equal(learner.values[groups, states].argmax(axis=1), optimum)
    np.testing.assert_allclose(learner.values[groups, states, optimum], 2.0, atol=1e-5)


@pytest.mark.parametrize("kind", [ContextualBandit, CompanyQLearner])
def test_shared_table_updates_ignore_worker_completion_order(kind):
    first, second = kind(), kind()
    ids = np.arange(40)
    groups = np.zeros(40, dtype=int)
    states = np.zeros(40, dtype=int)
    actions = ids % 3
    rewards = np.sin(ids)
    order = np.random.default_rng(81).permutation(40)
    tail = (states,) if kind is CompanyQLearner else ()
    first.update(ids, groups, states, actions, rewards, *tail)
    tail = (states[order],) if kind is CompanyQLearner else ()
    second.update(ids[order], groups[order], states[order], actions[order], rewards[order], *tail)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.counts, second.counts)


def _config(**changes):
    from abm.stage2 import load_simulation_config
    config = load_simulation_config(ROOT / "configs/stage2_five_nodes.json")
    return replace(config, graph_fixture=str(ROOT / config.graph_fixture), **changes)


@pytest.mark.parametrize("legacy_learning", [False, True])
@pytest.mark.parametrize("disable", ["master", "both_learners"])
def test_disabled_stage2_exactly_recovers_stage1(disable, legacy_learning):
    from abm.harness import run_stage1
    from abm.stage2 import create_market_harness
    switches = {"stage2_enabled": False} if disable == "master" else {
        "bandit_enabled": False, "company_learning_enabled": False}
    config = _config(learning_enabled=legacy_learning, **switches)
    expected = run_stage1(config.base_config())
    actual = create_market_harness(config).run()
    assert actual.fingerprint() == expected.fingerprint()
    assert actual.telemetry_events == expected.telemetry_events


def test_choices_are_held_and_credit_waits_for_complete_windows():
    from abm.stage2 import Stage2Harness
    harness = Stage2Harness(_config(trading_days=65))
    result = harness.run()
    frames = [json.loads(e.payload["stage2_json"]) for e in result.telemetry_events]
    for start in (0, 20, 40, 60):
        assert all(f["trader_actions"] == frames[start]["trader_actions"] for f in frames[start:start + 20])
    assert all(f["company_actions"] == frames[0]["company_actions"] for f in frames[:60])
    assert [f["day"] for f in frames if f["trader_rewards"]] == [20, 40, 60]
    assert [f["day"] for f in frames if f["company_rewards"]] == [60]
    assert harness.bandit.counts.sum() == 3 * 4
    assert harness.company_q.counts.sum() == 1
    assert result.stage2_audit["pending_trader_days"] == 5
    assert result.stage2_audit["pending_company_days"] == 5
    assert all(f["factor_preferences"] == frames[0]["factor_preferences"] for f in frames)
    assert all(f["trader_types"] == frames[0]["trader_types"] for f in frames)


def test_rewards_reconcile_to_net_wealth_without_double_counting_fees():
    from abm.stage2 import Stage2Harness
    harness = Stage2Harness(_config(trading_days=60))
    initial_wealth = harness.population.wealth(harness.config.initial_price).copy()
    result = harness.run()
    frames = [json.loads(e.payload["stage2_json"]) for e in result.telemetry_events]
    for end in (19, 39, 59):
        start_wealth = initial_wealth if end == 19 else np.array(frames[end - 20]["wealth"])
        net = (np.array(frames[end]["wealth"]) - start_wealth) / start_wealth
        reward = frames[end]["trader_rewards"]
        np.testing.assert_allclose(np.array(reward["gross_return"]) + reward["transaction_cost"], net, atol=1e-14)
        np.testing.assert_allclose(reward["reward"], net + reward["volatility_penalty"] + np.array(reward["drawdown_penalty"]), atol=1e-14)
    company = frames[-1]["company_rewards"]
    assert np.shape(company["components"]) == (1, 5)
    np.testing.assert_allclose(np.sum(company["components"], axis=1), company["reward"])
    assert max(abs(f["company_cash_error"]) for f in frames) < 1e-6


def test_private_truth_is_not_a_trader_observation():
    from abm.policies import MarketObservation
    from abm.stage2 import Stage2Harness
    first, second = Stage2Harness(_config()), Stage2Harness(_config())
    observation = MarketObservation(100.0, 1.0, np.array([100.0]), public_news_return=7.0)
    one = first.policy.act(observation, first.population, np.random.default_rng(15))
    two = second.policy.act(replace(observation, fundamental_value=1e12, public_news_return=-8.0),
                            second.population, np.random.default_rng(15))
    np.testing.assert_array_equal(one, two)
    assert not hasattr(first.policy.last_observation, "fundamental_value")
    assert first.policy.last_observation.public_news_return == first.macro_news


def test_contrarian_reverses_trend_and_reduce_cannot_increase_exposure():
    from abm.harness import MarketHarness
    from abm.policies import MarketObservation, RuleBasedPolicy
    config = replace(_config().base_config(), idiosyncratic_signal_scale=0,
                     liquidity_need_scale=0, activity_shock_scale=0)
    observation = MarketObservation(110.0, 100.0, np.linspace(100, 110, 80), realized_volatility=.01)
    orders = []
    for action in (1, 2, 3):
        population = MarketHarness(config).population
        population.base_activity_rates.fill(1 - 1e-12)
        population.positions[:] = 0 if action != 3 else np.array([5, -5, 1, -1])
        population.reference_positions.fill(0)
        population.desired_positions[:] = 0 if action != 3 else np.array([100, -100, -100, 100])
        submitted = RuleBasedPolicy(config).act(observation, population, np.random.default_rng(7), actions=np.full(4, action))
        orders.append(submitted)
        if action == 3:
            assert np.all(np.abs(population.positions + submitted) <= np.abs(population.positions))
            assert np.all(submitted * population.positions <= 0)
    assert np.all(orders[0] > 0)
    assert np.all(orders[1] < 0)


def test_no_graph_matches_zero_delay_unit_trust_direct_broadcast():
    from abm.stage2 import EPOCH, Stage2Harness
    direct = Stage2Harness(_config(graph_enabled=False, trading_days=25))
    graph = Stage2Harness(_config(trading_days=25))
    graph.graph = InformationGraph(graph.company_ids + graph.trader_ids, tuple(
        Edge("company-0", trader, "disclosure", 1.0, 1.0, 0, EPOCH.date()) for trader in graph.trader_ids))
    first, second = direct.run(), graph.run()
    np.testing.assert_array_equal(first.prices, second.prices)
    for one, two in zip(first.telemetry_events, second.telemetry_events, strict=True):
        a, b = json.loads(one.payload["stage2_json"]), json.loads(two.payload["stage2_json"])
        assert a["visible_news"] == b["visible_news"]
        assert a["submitted_orders"] == b["submitted_orders"]


def test_every_arrival_is_available_at_the_trading_clock():
    from datetime import datetime
    from abm.stage2 import EPOCH, Stage2Harness
    result = Stage2Harness(_config(trading_days=65, dynamic_graph_enabled=True)).run()
    for event in result.telemetry_events:
        frame = json.loads(event.payload["stage2_json"])
        now = EPOCH + timedelta(days=frame["day"] - 1)
        assert all(datetime.fromisoformat(d["available_at"]) <= now for d in frame["arrivals"])
        assert all(datetime.fromisoformat(u["effective_at"]) > now for u in frame["graph_updates"])


def test_display_drops_do_not_change_results_or_lossless_journal(tmp_path):
    import queue
    from abm.stage2 import Stage2Harness
    from abm.telemetry import NonBlockingQueueSink, read_telemetry
    config = _config(trading_days=65)
    output = tmp_path / "run"
    journal = tmp_path / ".run.events.jsonl"
    sink = NonBlockingQueueSink(queue.Queue(maxsize=1))
    result = Stage2Harness(config, sink, event_log=journal).run()
    assert sink.dropped_count == 64
    assert result.fingerprint() == Stage2Harness(config).run().fingerprint()
    assert read_telemetry(output) == result.telemetry_events
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    assert read_telemetry(output, after_sequence=60) == result.telemetry_events[60:]
    journal.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in result.telemetry_events), encoding="utf-8")
    result.write(output, config)
    assert not journal.exists()
    assert read_telemetry(output) == result.telemetry_events
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == result.telemetry_events[0].run_id
    assert (output / "events.jsonl").is_file()


def _run_seed(seed):
    from abm.stage2 import Stage2Harness
    return Stage2Harness(_config(seed=seed, trading_days=21, dynamic_graph_enabled=True)).run().fingerprint()


def test_serial_and_spawn_parallel_runs_have_identical_results():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    seeds = [431, 432, 433]
    expected = [_run_seed(seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert list(pool.map(_run_seed, seeds)) == expected


def test_native_five_node_live_and_replay_filters_match(tmp_path):
    from abm.desktop import MarketMonitorWindow, _application
    from abm.stage2 import Stage2Harness
    from abm.telemetry import VisualizerConfig, write_telemetry, read_telemetry
    events = Stage2Harness(_config(trading_days=65)).run().telemetry_events
    write_telemetry(events, tmp_path)
    application = _application()
    live = MarketMonitorWindow(mode='live', visualizer_config=VisualizerConfig(mode='live'))
    replay = MarketMonitorWindow(mode='replay', visualizer_config=VisualizerConfig(mode='replay'), replay_events=read_telemetry(tmp_path))
    live.timer.stop()
    replay.timer.stop()
    for event in events:
        live.ingest((event,))
    replay.ingest(read_telemetry(tmp_path))
    assert live.model.signature() == replay.model.signature()
    for window in (live, replay):
        panel = window.stage2_panel
        panel.follow.setChecked(False)
        panel.day.setValue(3)
        panel.trader.setCurrentIndex(2)
        panel.source.setCurrentText('company-0')
        assert len(panel.network.visible_nodes) == 5
        assert panel.current_frame['day'] == 3
        detail = json.loads(panel.details.toPlainText())
        assert detail['trader'] == 'trader-2'
        assert detail['arrivals_in_range'][0]['path'] == ['company-0', 'trader-0', 'trader-2']
        assert detail['arrivals_in_range'][0]['delay_days'] == 2
    assert live.stage2_panel.details.toPlainText() == replay.stage2_panel.details.toPlainText()
    application.processEvents()
    live.close()
    replay.close()


def test_web_discovers_nested_formal_acceptance_and_replays_stage2(tmp_path):
    from fastapi.testclient import TestClient
    from abm.stage2 import Stage2Harness
    from abm.web import create_app
    output = tmp_path / 'runs' / 'five'
    config = _config(trading_days=3)
    result = Stage2Harness(config).run()
    result.write(output, config)
    app = create_app(runs_root=tmp_path / 'runs', configs_root=ROOT / 'configs',
                     reports_root=ROOT / 'reports', web_root=tmp_path / 'none', auth_token='test-token')
    client = TestClient(app)
    headers = {'Authorization': 'Bearer test-token'}
    configs = client.get('/api/configs', headers=headers).json()
    assert {'stage2', 'stage2_five_nodes'} <= {c['id'] for c in configs}
    reports = client.get('/api/reports', headers=headers).json()
    formal = next(r for r in reports if r['id'] == 'stage1_iter6_20260926/formal')
    assert formal['stage1_complete'] is True
    assert formal['passed_checks'] == formal['total_checks'] == 22
    assert all('archive' not in r['id'].split('/') for r in reports)
    assert client.get('/api/reports/stage1_iter6_20260926/formal', headers=headers).status_code == 200
    assert client.get('/api/reports/archive/anything', headers=headers).status_code == 404
    actual = client.get('/api/runs/five/events?after=1', headers=headers).json()
    assert actual == [e.to_dict() for e in result.telemetry_events[1:]]


def test_stage2_spawned_worker_persists_events_despite_detached_display(tmp_path):
    import multiprocessing
    from abm.run_service import controlled_run_worker
    from abm.runner import resolve_code_revision
    from abm.stage2 import Stage2Harness
    from abm.telemetry import read_telemetry
    config = _config(trading_days=5)
    expected = Stage2Harness(config, code_revision=resolve_code_revision()).run()
    output = tmp_path / 'worker'
    context = multiprocessing.get_context('spawn')
    display, commands, completion = context.Queue(1), context.Queue(8), context.Queue(2)
    commands.put({'action': 'detach'})
    process = context.Process(target=controlled_run_worker, args=(config.to_dict(), str(output), display, commands, completion))
    process.start()
    status = completion.get(timeout=30)
    assert status['state'] == 'completed', status
    assert status['fingerprint'] == expected.fingerprint()
    assert status['dropped_frames'] == 4
    assert read_telemetry(output) == expected.telemetry_events
    display.get(timeout=5)
    process.join(timeout=5)
    assert process.exitcode == 0
