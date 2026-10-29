"""Reverting targeted audits and assurance keep earlier results (contract 25.9 (d), (e)): rerun archived development
trials with the new source.

  python check_defaults.py

Reruns seed 33001 of B, I, F and decay0_B from the frozen stage F development experiment, seed 34001 of V from the
targeted-audit development experiment (contract 21), seed 35001 of ALL_F16 from the assurance development
experiment (contract 22) and seed 33001 of rev0p32_B and rrw0p5_B from the joint-robustness development experiment
(contract 24), with the current src/abm (targeted audits and assurance now allowed with reverting true values,
the jump rule's room with the expected reversion drift), and requires every result fingerprint to equal the
archived one. The source hash differs, so run ids differ by design; the fingerprint covers the full result. Writes
check_defaults.json; the rerun archives go to work_check/defaults/ (local only).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
TRIALS = (   # (archive, scenario, seed)
    (ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1", "B", 33001),
    (ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1", "I", 33001),
    (ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1", "F", 33001),
    (ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1", "decay0_B", 33001),
    (ROOT / "reports" / "disclosure_targeted_audits_20261020" / "check_v1", "V", 34001),
    (ROOT / "reports" / "disclosure_assurance_20261021" / "check_v1", "ALL_F16", 35001),
    (ROOT / "reports" / "disclosure_joint_robustness_20261023" / "check_v1", "rev0p32_B", 33001),
    (ROOT / "reports" / "disclosure_joint_robustness_20261023" / "check_v1", "rrw0p5_B", 33001),
)


def rerun(scenario: dict, seed: int, destination: str) -> str:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import run_and_archive
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, seed)
    return run_and_archive(config, domain, destination, policy, archive="compact")["result_fingerprint"]


def archived_print(archive: Path, name: str, seed: int) -> str:
    attempts = sorted((archive / name / f"seed_{seed}").glob("attempt_*"))
    manifests = [json.loads((a / "manifest.json").read_text(encoding="utf-8")) for a in attempts]
    return next(m["result_fingerprint"] for m in manifests if m["status"] == "complete")


def main() -> None:
    work = HERE / "work_check" / "defaults"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    rows, archived, experiments = {}, {}, {}
    for archive, name, seed in TRIALS:
        frozen = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
        rows[name, seed] = next(row for row in frozen["scenarios"] if row["name"] == name)
        archived[name, seed] = archived_print(archive, name, seed)
        experiments[name, seed] = frozen["experiment_id"]
    with ProcessPoolExecutor(max_workers=len(TRIALS)) as pool:
        futures = {key: pool.submit(rerun, rows[key], key[1], str(work / f"{key[0]}_{key[1]}")) for key in rows}
        rerun_prints = {key: future.result() for key, future in futures.items()}
    result = {"trials": {f"{name}_{seed}": {"experiment_id": experiments[name, seed],
                                             "archived": archived[name, seed], "rerun": rerun_prints[name, seed],
                                             "identical": archived[name, seed] == rerun_prints[name, seed]}
                         for name, seed in rows}}
    result["all_identical"] = all(row["identical"] for row in result["trials"].values())
    (HERE / "check_defaults.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    if not result["all_identical"]:
        raise SystemExit("default options changed an earlier result")


if __name__ == "__main__":
    main()
