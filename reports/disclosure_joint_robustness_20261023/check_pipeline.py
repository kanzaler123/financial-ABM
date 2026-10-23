"""Pipeline checks that need full result arrays (contract 24.9), on development seed 33001 only.

  python check_pipeline.py

Reruns rr0p32_B, rr0p5_B, rrw0p32_B and rrw0p5_B from configs/disclosure_joint_robustness_check_v1.json with full
archives in work_check/pipeline/ (local only). Checks, as in contract 20.21:
  R  the cross-sectional mean of true values moves only through landed improvements, and the cross-sectional SD
     stays near its initial level instead of collapsing;
  P  every detection is charged a whole number (at least one) of fixed fines, and none without a detection;
  W  the triple block draws firm wash cost ratios; the joint block has none.
No effect estimates are computed. Writes check_pipeline.json.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
SEED = 33001
NAMES = ("rr0p32_B", "rr0p5_B", "rrw0p32_B", "rrw0p5_B")


def rerun(scenario: dict, destination: str) -> None:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.runner import run_and_archive
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, SEED)
    run_and_archive(config, domain, destination, policy, archive="full")


def main() -> None:
    from abm.disclosure.experiments import freeze_spec
    from abm.disclosure.runner import load_result

    frozen = freeze_spec(ROOT / "configs" / "disclosure_joint_robustness_check_v1.json")
    rows = {row["name"]: row for row in frozen["scenarios"]}
    work = HERE / "work_check" / "pipeline"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=len(NAMES)) as pool:
        for future in [pool.submit(rerun, rows[name], str(work / name)) for name in NAMES
                       if not (work / name / "result.npz").exists()]:
            future.result()
    results = {name: load_result(work / name / "result.npz") for name in NAMES}
    interval = frozen["scenarios"][0]["config"]["firm_decision_interval"]
    fine = float(rows["rr0p32_B"]["policy"]["baseline"]["fixed_fine"])
    out: dict = {"seed": SEED, "R": {}, "P": {}, "W": {}}
    for name, result in results.items():
        true = result.true_goodness
        landed = np.zeros(true.shape[0])
        for period, firm in np.argwhere(result.actions == 0):
            due = result.action_due_day[period, firm]
            if due < true.shape[0]:
                landed[due] += result.amounts[period, firm]
        residual = np.diff(true.mean(axis=1)) - landed[1:] / true.shape[1]
        out["R"][name] = {"cross_section_sd": {str(d): float(true[d].std()) for d in (0, 1200, 2400, 3500)},
                          "max_abs_mean_change_not_from_improvements": float(np.abs(residual).max())}
        prices = result.market.prices[::interval][:result.penalty_fraction.shape[0]]
        counts = result.penalty_fraction * prices / fine
        caught = result.detections.astype(bool)
        rounded = np.round(counts)
        out["P"][name] = {
            "detections": int(caught.sum()), "max_deviation_from_whole_fines": float(np.abs(counts - rounded).max()),
            "all_detections_at_least_one_fine": bool(np.all(rounded[caught] >= 1)),
            "no_fine_without_detection": bool(np.all(counts[~caught] == 0)),
            "fines_per_detection": {str(int(k)): int(v) for k, v in zip(*np.unique(rounded[caught], return_counts=True))}}
        ratios = result.wash_cost_ratio
        out["W"][name] = None if ratios is None else {
            "mean": float(ratios.mean()), "log_sd": float(np.log(ratios).std(ddof=1))}
    for kappa in ("0p32", "0p5"):
        out["R"][f"initial_true_values_equal_rr_rrw_{kappa}"] = bool(np.array_equal(
            results[f"rr{kappa}_B"].true_goodness[0], results[f"rrw{kappa}_B"].true_goodness[0]))
    out["passed"] = (
        all(r["max_abs_mean_change_not_from_improvements"] < 1e-9 and r["cross_section_sd"]["3500"] > .5
            for k, r in out["R"].items() if isinstance(r, dict))
        and all(v for k, v in out["R"].items() if k.startswith("initial_true_values_equal"))
        and all(p["max_deviation_from_whole_fines"] < 1e-6 and p["all_detections_at_least_one_fine"]
                and p["no_fine_without_detection"] and p["detections"] > 0 for p in out["P"].values())
        and all(out["W"][f"rrw{k}_B"] is not None and out["W"][f"rr{k}_B"] is None for k in ("0p32", "0p5")))
    (HERE / "check_pipeline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
