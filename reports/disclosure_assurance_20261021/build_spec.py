"""Assurance providers: write the development and confirmatory specs for contract section 22 and check them.

Eighteen scenarios. In the main setting (natural decay 0.2): B and I as in stage F; VOL16 (voluntary assurance,
16x liability); ALL_R0 ... ALL_R100 (every firm checked, providers assigned at random, liability ladder
0/1/4/10/16/100 times the statutory CNY 5 million); ALL_F1 and ALL_F16 (every firm checked, firms pick their
provider each year); LRG_R16 and LRG_F16 (only the large size class checked). Without decay (the lower end of the
magnitude range of contract 20.14): B, I, ALL_R1, ALL_R16, ALL_F16. Market, domain, switch day and analysis
metrics come from the stage F builder, so the derivation is identical.

  python build_spec.py [--stage-f <stage F confirmatory archive>]

Writes configs/disclosure_assurance_development_v1.json (seeds 35001-35003) and
configs/disclosure_assurance_confirmatory_v1.json (seeds 43001-43030); the two differ only in name, purpose and
seeds. Then checks that every scenario, with its assurance options set back to the defaults, equals the frozen
stage F scenario it was derived from.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE_F = ROOT / "reports" / "disclosure_stage_f_20261014"

_spec = importlib.util.spec_from_file_location("stage_f_build_specs", STAGE_F / "build_specs.py")
stage_f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_f)

SEEDS = {"development": [35001, 35002, 35003], "confirmatory": list(range(43001, 43031))}   # contract 22.1
# Money as in design_numbers.py (contract 22.2-22.3): CNY 10 million is the stage F fixed fine F1.
WAN = stage_f.F1 / 1000.0
FEE = 50 * WAN * 60 / 252                 # CNY 0.5 million a year, per check (one per 60-day period)
KAPPA = FEE / 0.5**2                      # the fee covers the effort cost of quality 0.5
STATUTORY_LIABILITY = 500 * WAN           # Securities Law art. 213: 10x business income of CNY 0.5 million
ASSURANCE_FIELDS = ("assurance_scope", "assurer_selection", "assurer_liability", "assurance_fee",
                    "assurer_effort_cost", "assurer_count")


def assurance(scope: str, selection: str, dose: float) -> dict:
    return {"assurance_scope": scope, "assurer_selection": selection, "assurer_liability": dose * STATUTORY_LIABILITY,
            "assurance_fee": FEE, "assurer_effort_cost": KAPPA, "assurer_count": 4}


MAIN = {   # scenario -> (policy changes at the switch, stage F scenario it derives from)
    "B": ({}, "B"), "I": (stage_f.POLICIES["I"], "I"),
    "VOL16": (assurance("voluntary", "firm", 16), "B"),
    **{f"ALL_R{dose}": (assurance("all", "assigned", dose), "B") for dose in (0, 1, 4, 10, 16, 100)},
    "ALL_F1": (assurance("all", "firm", 1), "B"), "ALL_F16": (assurance("all", "firm", 16), "B"),
    "LRG_R16": (assurance("large", "assigned", 16), "B"), "LRG_F16": (assurance("large", "firm", 16), "B"),
}
NO_DECAY = ("B", "I", "ALL_R1", "ALL_R16", "ALL_F16")


def rows() -> list[tuple[str, dict, dict | None, str]]:
    """(scenario, policy changes, domain overrides, stage F scenario it derives from)."""
    out = [(name, changes, None, source) for name, (changes, source) in MAIN.items()]
    for name in NO_DECAY:
        changes, source = MAIN[name]
        out.append((f"decay0_{name}", changes, {"natural_decay": 0.0}, f"decay0_{source}"))
    return out


def scenarios() -> list[dict]:
    return [stage_f.scenario(name, changes or None, domain=domain) for name, changes, domain, _ in rows()]


def comparisons() -> list[dict]:
    out = [
        {"name": "C1_ALL_R16_minus_B", "weights": {"ALL_R16": 1, "B": -1}, "primary": True},
        {"name": "C2_ALL_F16_minus_ALL_R16", "weights": {"ALL_F16": 1, "ALL_R16": -1}, "primary": True},
        {"name": "C3_ALL_R1_minus_B", "weights": {"ALL_R1": 1, "B": -1}, "primary": True},
        {"name": "C4_LRG_R16_minus_B_size", "weights": {"LRG_R16": 1, "B": -1}, "primary": True,
         "primary_metric": "size.large_minus_small_sd"},
        {"name": "C5_I_minus_ALL_R16", "weights": {"I": 1, "ALL_R16": -1}, "primary": True},
    ]

    def add(plus: str, minus: str) -> None:
        out.append({"name": f"{plus}_minus_{minus}", "weights": {plus: 1, minus: -1}})

    for dose in (0, 1, 4, 10, 16, 100):
        add(f"ALL_R{dose}", "B")
    for plus, minus in (("ALL_F1", "ALL_R1"), ("LRG_F16", "LRG_R16"), ("LRG_R16", "ALL_R16"), ("VOL16", "B"),
                        ("I", "ALL_F16"), ("I", "B")):
        add(plus, minus)
    for plus, minus in (("ALL_R16", "B"), ("ALL_F16", "ALL_R16"), ("ALL_R1", "B"), ("I", "ALL_R16"), ("I", "B")):
        add(f"decay0_{plus}", f"decay0_{minus}")
    return out


def spec(purpose: str) -> dict:
    stage_f_spec = stage_f.spec("confirmatory", SEEDS["confirmatory"])
    return {
        "name": f"disclosure_assurance_{purpose}_v1", "purpose": purpose, "archive": "compact",
        "_note": "Plan stage E.2, assurance providers (research contract section 22). Development and confirmatory "
                 "specs differ only in name, purpose and seeds; the development run checks the pipeline and may not "
                 "change the design. Primary comparisons C1-C5 are Holm-adjusted.",
        "domain": stage_f_spec["domain"], "market": stage_f_spec["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(), "comparisons": comparisons(), "analysis": stage_f_spec["analysis"],
    }


def reset(row: dict, old_name: str) -> dict:
    """The scenario with its assurance options back at the defaults, written as stage F writes such a scenario."""
    row = json.loads(json.dumps(row))
    row["name"] = old_name
    policy = row["policy"]
    if policy["intervention"] is not None:
        for field in ASSURANCE_FIELDS:
            policy["intervention"].pop(field, None)
        if policy["intervention"] == policy["baseline"]:
            policy["intervention"], policy["start_day"] = None, 0
    return row


def check_derivation(stage_f_archive: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec
    from abm.disclosure.policy import ASSURANCE_DEFAULTS

    new = {row["name"]: row for row in freeze_spec(ROOT / "configs" / "disclosure_assurance_confirmatory_v1.json")["scenarios"]}
    frozen = json.loads((stage_f_archive / "experiment.json").read_text(encoding="utf-8"))
    old = {row["name"]: row for row in frozen["scenarios"]}
    for name, changes, _, source in rows():
        assert reset(new[name], source) == old[source], f"{name} differs from stage F {source}"
        intervention = new[name]["policy"]["intervention"]
        for field in ASSURANCE_FIELDS:
            if field in changes:
                assert intervention.get(field, ASSURANCE_DEFAULTS[field]) == changes[field], (name, field)
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f", type=Path, default=STAGE_F / "confirmatory_v1",
                        help="stage F confirmatory archive (read-only)")
    args = parser.parse_args()
    for purpose in ("development", "confirmatory"):
        path = ROOT / "configs" / f"disclosure_assurance_{purpose}_v1.json"
        path.write_text(json.dumps(spec(purpose), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path, len(scenarios()), "scenarios", len(comparisons()), "comparisons", len(SEEDS[purpose]), "seeds")
    count = check_derivation(args.stage_f)
    print(f"derivation check: {count} scenarios equal their stage F scenario once the assurance options are reset")


if __name__ == "__main__":
    main()
