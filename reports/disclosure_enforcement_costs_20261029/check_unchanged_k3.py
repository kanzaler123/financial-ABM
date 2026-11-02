"""Contract 28.37: the initial_value audit allocation leaves earlier results and run identities unchanged.

    .venv/Scripts/python.exe -X utf8 reports/disclosure_enforcement_costs_20261029/check_unchanged_k3.py

1. Re-runs the final-stage development run (final_check_v1, theta_hat x seed 38201) with the working-tree source and
   compares its result fingerprint with the archived manifest.
2. Freezes the K3 final-stage and q-scan specs with the working-tree source (in memory; nothing is written) and
   compares every field with the archived experiment.json: only source_hashes and the experiment_id they determine
   may differ, and within source_hashes only the files changed for 28.37.
Archives are only read. Writes check_unchanged_k3.json (LF).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _no_wmi import prefill_uname  # noqa: E402

prefill_uname()

from abm.disclosure.experiments import _inputs, freeze_spec  # noqa: E402
from abm.disclosure.game import DisclosureMarket  # noqa: E402

CHANGED = {"abm/disclosure/policy.py", "abm/disclosure/game.py"}
ARCHIVES = {"final": ("configs/disclosure_stage_k3_final_v1.json", "final_v1"),
            "qscan": ("configs/disclosure_stage_k3_qscan_v1.json", "qscan_v1")}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out: dict = {"contract": "28.37"}
    frozen = load(HERE / "final_check_v1" / "experiment.json")
    (scenario,) = [s for s in frozen["scenarios"] if s["name"] == "theta_hat"]
    attempt = HERE / "final_check_v1" / "theta_hat" / "seed_38201" / "attempt_001"
    manifest = load(attempt / "manifest.json")
    assert manifest["status"] == "complete"
    config, domain, policy = _inputs(scenario, 38201)
    started = time.perf_counter()
    fingerprint = DisclosureMarket(config, domain, policy).run().fingerprint()
    out["rerun"] = {"archive": attempt.relative_to(HERE).as_posix(), "archived": manifest["result_fingerprint"],
                    "rerun": fingerprint, "identical": fingerprint == manifest["result_fingerprint"],
                    "seconds": round(time.perf_counter() - started, 1)}
    ok = out["rerun"]["identical"]
    for label, (spec, archive) in ARCHIVES.items():
        recorded, now = load(HERE / archive / "experiment.json"), freeze_spec(ROOT / spec)
        differing = sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k))
        files = sorted(f for f in set(recorded["source_hashes"]) | set(now["source_hashes"])
                       if recorded["source_hashes"].get(f) != now["source_hashes"].get(f))
        passed = set(differing) <= {"source_hashes", "experiment_id"} and set(files) <= CHANGED
        out[label] = {"spec": spec, "recorded_experiment_id": recorded["experiment_id"],
                      "working_tree_experiment_id": now["experiment_id"], "differing_fields": differing,
                      "differing_source_files": files, "passed": passed}
        ok = ok and passed
    out["passed"] = ok
    (HERE / "check_unchanged_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
