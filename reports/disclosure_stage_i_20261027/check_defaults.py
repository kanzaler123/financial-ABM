"""The rating channel's default keeps earlier results (contract 26.20 (c)): rerun the eight archived development
trials of reports/disclosure_stage_h3_20261027/check_defaults.py with the current source.

  python check_defaults.py

Same trials and rerun as the H3 check (stage F B, I, F, decay0_B; V of contract 21; ALL_F16 of contract 22;
rev0p32_B and rrw0p5_B of contract 24). Every result fingerprint must equal the archived one, and the config each
rerun records in its manifest must equal the archived manifest's field by field (financing_rating_slope = 0 is
left out of identities). Run ids include the source hashes, so they differ by design. Writes check_defaults.json
(LF); the rerun archives go to work_check/defaults/ (local only).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("h3_check_defaults",
                                               ROOT / "reports" / "disclosure_stage_h3_20261027" / "check_defaults.py")
h3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h3)


def rerun(scenario: dict, seed: int, destination: str) -> str:
    """The H3 check's rerun (defined here so that worker processes can import it)."""
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import run_and_archive
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, seed)
    return run_and_archive(config, domain, destination, policy, archive="compact")["result_fingerprint"]


def manifest(archive: Path, name: str, seed: int) -> dict:
    for attempt in sorted((archive / name / f"seed_{seed}").glob("attempt_*")):
        row = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        if row["status"] == "complete":
            return row
    raise LookupError(f"no complete attempt for {name} seed {seed}")


def main() -> None:
    work = HERE / "work_check" / "defaults"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    rows, archived = {}, {}
    for archive, name, seed in h3.TRIALS:
        frozen = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
        rows[name, seed] = next(row for row in frozen["scenarios"] if row["name"] == name)
        archived[name, seed] = (frozen["experiment_id"], manifest(archive, name, seed))
    destinations = {key: work / f"{key[0]}_{key[1]}" for key in rows}
    with ProcessPoolExecutor(max_workers=len(rows)) as pool:
        futures = {key: pool.submit(rerun, rows[key], key[1], str(destinations[key])) for key in rows}
        prints = {key: future.result() for key, future in futures.items()}
    trials = {}
    for key, (experiment_id, old) in archived.items():
        new = _single(destinations[key])
        trials[f"{key[0]}_{key[1]}"] = {
            "experiment_id": experiment_id, "archived": old["result_fingerprint"], "rerun": prints[key],
            "identical": old["result_fingerprint"] == prints[key] == new["result_fingerprint"],
            "config_identical": old["config"] == new["config"],
            "rating_slope_in_config": "financing_rating_slope" in new["config"],
        }
    result = {"trials": trials,
              "all_identical": all(row["identical"] and row["config_identical"] and not row["rating_slope_in_config"]
                                   for row in trials.values())}
    (HERE / "check_defaults.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))
    if not result["all_identical"]:
        raise SystemExit("the default rating slope changed an earlier result")


def _single(destination: Path) -> dict:
    """The manifest run_and_archive wrote for one trial under ``destination``."""
    found = sorted(destination.rglob("manifest.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in found]
    complete = [row for row in rows if row["status"] == "complete"]
    if len(complete) != 1:
        raise LookupError(f"expected one complete manifest under {destination}, found {len(complete)}")
    return complete[0]


if __name__ == "__main__":
    main()
