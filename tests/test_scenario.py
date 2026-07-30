import json
from pathlib import Path

from abm.manifest import sha256_file
from abm.scenario import load_synthetic_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = PROJECT_ROOT / "data" / "synthetic" / "stage0_10_day.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "stage0_10_day.json"


def test_stage0_scenario_loads_all_contracts() -> None:
    scenario = load_synthetic_scenario(SCENARIO_PATH)

    assert len(scenario.market_states) == 10
    assert len(scenario.agent_states) == 5
    assert len(scenario.company_states) == 2
    assert len(scenario.messages) == 3
    assert len(scenario.edges) == 4
    assert len(scenario.orders) == 10
    assert len(scenario.factor_events) == 2


def test_stage0_market_dates_are_weekdays() -> None:
    scenario = load_synthetic_scenario(SCENARIO_PATH)

    assert all(state.date.weekday() < 5 for state in scenario.market_states)


def test_frozen_fixture_matches_immutable_data_manifest() -> None:
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][0]

    assert artifact["sha256"] == sha256_file(SCENARIO_PATH)

