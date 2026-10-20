"""Default options keep earlier results (contract 21.9): rerun stage F development trials with the new source.

  python check_defaults.py [--stage-f-development <stage F development archive>]

Reruns seed 33001 of B, I, F and decay0_B from the frozen stage F development experiment with the current
src/abm (targeted-audit options at their defaults) and requires every result fingerprint to equal the archived one.
The source hash differs, so run ids differ by design; the fingerprint covers the full result. Writes
check_defaults.json; the rerun archives go to work_check/defaults/ (local only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCENARIOS = ("B", "I", "F", "decay0_B")
SEED = 33001


def rerun(scenario: dict, destination: str) -> str:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import run_and_archive
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, SEED)
    return run_and_archive(config, domain, destination, policy, archive="compact")["result_fingerprint"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f-development", type=Path,
                        default=ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1")
    args = parser.parse_args()
    frozen = json.loads((args.stage_f_development / "experiment.json").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in frozen["scenarios"]}
    archived = {}
    for name in SCENARIOS:
        attempts = sorted((args.stage_f_development / name / f"seed_{SEED}").glob("attempt_*"))
        manifests = [json.loads((a / "manifest.json").read_text(encoding="utf-8")) for a in attempts]
        archived[name] = next(m["result_fingerprint"] for m in manifests if m["status"] == "complete")
    work = HERE / "work_check" / "defaults"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(rerun, rows[name], str(work / f"{name}_{SEED}")) for name in SCENARIOS}
        rerun_prints = {name: future.result() for name, future in futures.items()}
    result = {"seed": SEED, "stage_f_development_experiment_id": frozen["experiment_id"],
              "trials": {name: {"archived": archived[name], "rerun": rerun_prints[name],
                                "identical": archived[name] == rerun_prints[name]} for name in SCENARIOS}}
    result["all_identical"] = all(row["identical"] for row in result["trials"].values())
    (HERE / "check_defaults.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    if not result["all_identical"]:
        raise SystemExit("default options changed a stage F result")


if __name__ == "__main__":
    main()
