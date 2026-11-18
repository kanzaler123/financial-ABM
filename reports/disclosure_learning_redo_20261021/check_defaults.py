"""Default options keep earlier results (contract 23.11): rerun archived worlds with the new source.

  python check_defaults.py

Reruns, with the current src/abm (new options at their defaults), and requires every result fingerprint to equal
the archived one:
- stage D.2 (reports/disclosure_rl_comparison_20261013/runs/v2, local only): scenario B validation seed 31401
  with the frozen Q-learning table and with the discrete NPV rule; scenario B training world 31501 and the
  pre-policy training world 31201, each from its archived input table (the output table must match as well);
- stage F development seed 33001, scenario B (continuous NPV).
Writes check_defaults.json.
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
D2 = ROOT / "reports" / "disclosure_rl_comparison_20261013" / "runs" / "v2"
STAGE_F = ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1"
D2_RUNS = ("B/validation/seed_31401/q_learning", "B/validation/seed_31401/npv_discrete",
           "B/training/seed_31501", "baseline/training/seed_31201")


def rerun_d2(relative: str) -> dict:
    import numpy as np

    from abm.disclosure.config import MarketConfig
    from abm.disclosure.domain import DomainSpec
    from abm.disclosure.game import DisclosureMarket
    from abm.disclosure.learning import load_frozen, table_fingerprint
    from abm.disclosure.policy import PolicyRegime, PolicySchedule
    from abm.platform_tuning import disable_power_throttling
    from abm.tabular import CompanyQLearner

    disable_power_throttling()
    folder = D2 / relative
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = MarketConfig.from_dict(manifest["config"])
    domain = DomainSpec.from_dict(manifest["domain"])
    payload = manifest["policy"]
    policy = PolicySchedule(PolicyRegime(**payload["baseline"]),
                            None if payload["intervention"] is None else PolicyRegime(**payload["intervention"]),
                            payload["start_day"])
    kwargs, learner = {}, None
    learning = manifest.get("learning")
    if learning is not None:
        frozen, _ = load_frozen(folder / "input_model.npz", expected_sha256=manifest["artifacts"]["input_model.npz"])
        if learning["training"]:
            learner = CompanyQLearner(groups=6, actions=frozen.values.shape[2], epsilon=learning["epsilon"],
                                      learning_rate=frozen.learning_rate, discount=frozen.discount)
            learner.learning_rate_power = frozen.learning_rate_power
            learner.values, learner.counts = np.array(frozen.values), np.array(frozen.counts)
        else:
            learner = frozen
        kwargs = {"learner": learner, "training": learning["training"],
                  "coverage_target": learning.get("coverage_target", 0),
                  "common_prepolicy_sha256": learning.get("common_prepolicy_sha256")}
    result = DisclosureMarket(config, domain, policy, **kwargs).run()
    row = {"archived": manifest["result_fingerprint"], "rerun": result.fingerprint()}
    if learner is not None:
        archived_table = json.loads((folder / "learning.json").read_text(encoding="utf-8"))["final_table_fingerprint"]
        row.update({"archived_table": archived_table, "rerun_table": table_fingerprint(learner)})
    row["identical"] = row["archived"] == row["rerun"] and row.get("archived_table") == row.get("rerun_table")
    return row


def rerun_stage_f(scenario: dict, seed: int) -> str:
    from abm.disclosure.experiments import _inputs
    from abm.disclosure.game import DisclosureMarket
    from abm.platform_tuning import disable_power_throttling

    disable_power_throttling()
    config, domain, policy = _inputs(scenario, seed)
    return DisclosureMarket(config, domain, policy).run().fingerprint()


def main() -> None:
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    frozen = json.loads((STAGE_F / "experiment.json").read_text(encoding="utf-8"))
    scenario = next(row for row in frozen["scenarios"] if row["name"] == "B")
    attempts = sorted((STAGE_F / "B" / "seed_33001").glob("attempt_*"))
    archived_f = next(m["result_fingerprint"] for m in
                      (json.loads((a / "manifest.json").read_text(encoding="utf-8")) for a in attempts)
                      if m["status"] == "complete")
    with ProcessPoolExecutor(max_workers=len(D2_RUNS) + 1) as pool:
        futures = {relative: pool.submit(rerun_d2, relative) for relative in D2_RUNS}
        stage_f = pool.submit(rerun_stage_f, scenario, 33001)
        rows = {f"d2/{relative}": future.result() for relative, future in futures.items()}
        rerun_f = stage_f.result()
    rows["stage_f_development/B/seed_33001"] = {"experiment_id": frozen["experiment_id"], "archived": archived_f,
                                                "rerun": rerun_f, "identical": archived_f == rerun_f}
    result = {"trials": rows, "all_identical": all(row["identical"] for row in rows.values())}
    (HERE / "check_defaults.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    if not result["all_identical"]:
        raise SystemExit("default options changed an earlier result")


if __name__ == "__main__":
    main()
