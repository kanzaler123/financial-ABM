"""The browser sandbox (showcase/sandbox) against the disclosure model it simplifies."""

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abm.disclosure.config import MarketConfig
from abm.disclosure.domain import load_domain
from abm.disclosure.game import DisclosureMarket
from abm.disclosure.policy import PolicyRegime, PolicySchedule

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "showcase" / "sandbox"
sys.path.insert(0, str(SANDBOX))

import reference  # noqa: E402


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_matches(actual, expected, path="fixture"):
    """Same structure, identical integers and strings, floats within 1e-12 of max(|x|, 1e-300)."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert_matches(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-300), path
    else:
        assert actual == expected, path


@pytest.fixture(scope="module")
def regenerated():
    return load_script(ROOT / "reports" / "disclosure_stage_g_20261015" / "fixtures.py").build()


def test_committed_fixtures_still_follow_the_model_code(regenerated):
    core, dynamics = regenerated
    assert_matches(core, json.loads((SANDBOX / "fixtures" / "core_cases.json").read_text(encoding="utf-8")), "core")
    assert_matches(dynamics, json.loads((SANDBOX / "fixtures" / "dynamics_cases.json").read_text(encoding="utf-8")), "dynamics")


def test_snapshot_reading_leaves_the_full_model_run_unchanged():
    # The committed AI-washing domain with the stage F market; the CSI 300 domain is not in the repository.
    spec = json.loads((ROOT / "configs" / "disclosure_stage_f_confirmatory_v1.json").read_text(encoding="utf-8"))
    config = MarketConfig.from_dict({**spec["market"], "seed": 51001, "n_investors": 120, "n_firms": 24,
                                     "trading_days": 360, "burn_in_days": 180})
    domain = load_domain(ROOT / "configs" / "domains" / "ai_washing.json")
    # No proportional fine and a small fixed fine: 5 of 24 firms inflate and 5 have open cases on day 180,
    # so the snapshot decision covers washing, new cases and cases joined without a new fine.
    baseline = replace(PolicyRegime.baseline(domain), penalty=0.0, fixed_fine=0.05 * 4.5614908036639594e-05)
    policy = PolicySchedule(baseline).with_intervention(181, transparency=0.6)
    market = reference.SnapshotMarket(config, domain, policy, snapshot_day=180)
    result = reference.take_snapshot(market)
    assert result.fingerprint() == DisclosureMarket(config, domain, policy).run().fingerprint()
    snapshot = market.snapshot
    firms = reference.SandboxFirms(market.firms.size_class.copy(), market.initial_goodness,
                                   market.attributes.cost_scale.copy(), snapshot["prices"], snapshot["benefit"])
    state = {"goodness": snapshot["goodness"], "inflation": snapshot["inflation"], "cost_scale": market.attributes.cost_scale,
             "pending": [{"due_day": d, "kind": k, "firms": f, "amounts": a} for d, k, f, a in snapshot["pending"]]}
    exported = reference.domain_from({name: list(value) if name == "goodness_bounds" else value
                                      for name, value in ((name, getattr(domain, name)) for name in reference.SANDBOX_DOMAIN_FIELDS)})
    assert (snapshot["actions"] == 1).any() and (snapshot["inflation"] > 0).any()
    for stand_in in (domain, exported):
        choices, _ = reference.decide(reference.snapshot_state(state, stand_in), firms, stand_in, policy.at(180), 180)
        assert np.array_equal(choices.actions, snapshot["actions"])
        np.testing.assert_allclose(choices.amounts, snapshot["amounts"], rtol=1e-12, atol=0)


def test_exported_domains_match_their_configuration_files():
    sandbox = json.loads((ROOT / "showcase" / "data" / "sandbox_firms.json").read_text(encoding="utf-8"))
    available = reference.load_domains(ROOT)
    assert "ai_washing" in available
    for key, domain in available.items():
        payload = sandbox["domains"][key]
        assert payload["name"] == domain.name
        assert reference.exported_parameters_match(payload, domain), key
        assert len(payload["firms"]["benefit"]) == 200
    if "esg" not in available:
        pytest.skip("the CSI 300 domain file is vendor-derived and not in the repository; checked where it exists")


def test_exported_firms_record_the_code_that_produced_them():
    import hashlib

    sandbox = json.loads((ROOT / "showcase" / "data" / "sandbox_firms.json").read_text(encoding="utf-8"))
    for path_key, hash_key in (("generated_by", "script_sha256"), ("reference", "reference_sha256"),
                               ("experiment_spec", "experiment_spec_sha256")):
        assert hashlib.sha256((ROOT / sandbox[path_key]).read_bytes()).hexdigest() == sandbox[hash_key], path_key


def test_mulberry32_reference_values():
    rng = reference.Mulberry32(1)
    values = [rng.random() for _ in range(3)]
    assert all(0 <= value < 1 for value in values)
    assert len(set(values)) == 3
    with pytest.raises(ValueError):
        reference.Mulberry32(-1)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_core_reproduces_the_python_model():
    completed = subprocess.run(["node", str(SANDBOX / "check_core.js")], capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "matches the Python model" in completed.stdout
