"""Learning-firm redo: write the development and confirmatory specs for contract section 23.

  python build_spec.py

Writes configs/disclosure_learning_redo_development_v1.json and configs/disclosure_learning_redo_confirmatory_v1.json.
The two differ only in name, purpose and budget (replicates, worlds and seeds). The learner, market, domain and
scenarios are those of stage D.2 (reports/disclosure_rl_comparison_20261013/run_stage_d.py); the build checks
that version A's market config equals the D.2 config field by field.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abm.disclosure.config import MarketConfig  # noqa: E402
from abm.manifest import canonical_json_bytes  # noqa: E402

DOMAIN = "configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json"
CONFIG = MarketConfig(   # contract 19.1 and 19.7, as in run_stage_d.py
    n_investors=1000, n_firms=200, trading_days=3500, burn_in_days=1200,
    initial_impact_share=0.2, taste_median=0.0724, imitation_rule="pairwise",
    financing_exposure_fraction=0.5, financing_spread=0.05, financing_repricing_interval=60,
    financing_horizon_days=1260, firm_discount_rate=0.05, intrinsic_value_log_sd=0.9151544344205937,
    firm_policy="q_learning", firm_action_fractions=(0.04, 0.2, 1.0),
    firm_learning_context="signal_tercile", firm_learning_rate_power=0.85,
    firm_cost_budget_fraction=0.01,
)
VERSIONS = {"A": "signal_tercile", "V": "improvement_stock"}   # contract 23.3
SWITCH_DAY = 1201
FOUR_TIMES_FINE = 0.00018245963214655838  # 4 x CNY 10 million in model units (contract 16.2, 16.5; run_stage_d.py)
SCENARIOS = {"B": {}, "I": {"transparency": 0.6}, "F": {"fixed_fine": FOUR_TIMES_FINE},
             "IF": {"transparency": 0.6, "fixed_fine": FOUR_TIMES_FINE}}
RESERVED = list(range(20261101, 20261131))
BUDGETS = {   # contract 23.1 and 23.8
    "development": {"replicates": 2, "prepolicy_base": 37000, "postpolicy_base": 37200, "block": 100,
                    "prepolicy_worlds": 3, "postpolicy_worlds": 2, "check_seeds": [37401],
                    "validation_seeds": [37501, 37502]},
    "confirmatory": {"replicates": 10, "prepolicy_base": 47000, "postpolicy_base": 45000, "block": 100,
                     "prepolicy_worlds": 96, "postpolicy_worlds": 64, "check_seeds": [46001, 46002, 46003],
                     "validation_seeds": [46101, 46102, 46103, 46104, 46105]},
}


def spec(purpose: str) -> dict:
    budget = BUDGETS[purpose]
    replicates = []
    for index in range(budget["replicates"]):
        # Development post-policy blocks are 37201-37202 and 37301-37302 (contract 23.1).
        post_base = budget["postpolicy_base"] + budget["block"] * index
        pre_base = budget["prepolicy_base"] + budget["block"] * index
        replicates.append({"replicate": index + 1,
                           "prepolicy_training_seeds": list(range(pre_base + 1, pre_base + budget["prepolicy_worlds"] + 1)),
                           "postpolicy_training_seeds": list(range(post_base + 1, post_base + budget["postpolicy_worlds"] + 1))})
    return {
        "schema_version": 1, "name": f"disclosure_learning_redo_{purpose}_v1", "purpose": purpose,
        "contract": "docs/disclosure_research_contract.md section 23",
        "domain": DOMAIN, "domain_sha256": hashlib.sha256((ROOT / DOMAIN).read_bytes()).hexdigest(),
        "config": CONFIG.to_dict(), "versions": VERSIONS, "switch_day": SWITCH_DAY, "scenarios": SCENARIOS,
        "replicates": replicates, "check_seeds": budget["check_seeds"],
        "validation_seeds": budget["validation_seeds"], "reserved_seeds": RESERVED,
        "coverage_target": 20, "archive": "compact", "npv_rules": ["npv_discrete", "npv"],
        "prepolicy_validation_rules": ["q_learning"], "postpolicy_validation_rules": ["q_learning"],
        "window": {"start_day": 1200, "end_day": 3500},
    }


def check(payload: dict) -> None:
    sys.path.insert(0, str(ROOT / "reports" / "disclosure_rl_comparison_20261013"))
    import run_stage_d  # noqa: PLC0415

    if MarketConfig.from_dict(payload["config"]) != run_stage_d.CONFIG:
        raise SystemExit("version A config differs from stage D.2")
    if run_stage_d.SCENARIOS != payload["scenarios"] or run_stage_d.SWITCH_DAY != payload["switch_day"]:
        raise SystemExit("scenarios differ from stage D.2")
    if run_stage_d.COVERAGE_TARGET != payload["coverage_target"] or run_stage_d.RESERVED != payload["reserved_seeds"]:
        raise SystemExit("coverage target or reserved seeds differ from stage D.2")
    seeds = [seed for row in payload["replicates"] for key in ("prepolicy_training_seeds", "postpolicy_training_seeds")
             for seed in row[key]] + payload["check_seeds"] + payload["validation_seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) & set(payload["reserved_seeds"]):
        raise SystemExit("seed blocks overlap")


def main() -> None:
    for purpose in BUDGETS:
        payload = spec(purpose)
        check(payload)
        path = ROOT / "configs" / f"disclosure_learning_redo_{purpose}_v1.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT).as_posix(), "experiment_id", hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
              "file sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
