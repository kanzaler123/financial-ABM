"""Targeted audits: write the development and confirmatory specs for contract section 21 and check them.

Fifteen scenarios. In the main setting (natural decay 0.2): B, I, F as in stage F; A2 (random audits at 2%);
V and V_hidden (audits in proportion to market value, rule public or not); J and J_hidden (half the audits at
random, half shared by firms whose report rose by more than 0.125 SD since the previous audit); VF (V with the
4x fixed fine). Without decay (the lower end of the magnitude range of contract 20.14): B, I, V, J, J_hidden, VF.
Market, domain, switch day and analysis metrics come from the stage F builder, so the derivation is identical.

  python build_spec.py [--stage-f <stage F confirmatory archive>]

Writes configs/disclosure_targeted_audits_development_v1.json (seeds 34001-34003) and
configs/disclosure_targeted_audits_confirmatory_v1.json (seeds 42001-42030); the two differ only in name,
purpose and seeds. Then checks that every scenario, with its audit options set back to the defaults, equals the
frozen stage F scenario it was derived from.
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

SEEDS = {"development": [34001, 34002, 34003], "confirmatory": list(range(42001, 42031))}   # contract 21.1
F4 = 4 * stage_f.F1
VALUE = {"audit_allocation": "value"}
JUMP = {"audit_allocation": "jump"}
HIDDEN = {"audit_rule_public": False}
MAIN = {   # scenario -> (policy changes at the switch, stage F scenario it derives from)
    "B": ({}, "B"), "I": (stage_f.POLICIES["I"], "I"), "F": (stage_f.POLICIES["F"], "F"),
    "A2": ({"audit_probability": 0.02}, "B"),
    "V": (VALUE, "B"), "V_hidden": ({**VALUE, **HIDDEN}, "B"),
    "J": (JUMP, "B"), "J_hidden": ({**JUMP, **HIDDEN}, "B"),
    "VF": ({**VALUE, "fixed_fine": F4}, "F"),
}
NO_DECAY = ("B", "I", "V", "J", "J_hidden", "VF")
AUDIT_FIELDS = ("audit_allocation", "audit_rule_public", "audit_jump_threshold")


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
        {"name": "E1_V_minus_B", "weights": {"V": 1, "B": -1}, "primary": True},
        {"name": "E2_J_minus_B", "weights": {"J": 1, "B": -1}, "primary": True},
        {"name": "E3_J_minus_J_hidden", "weights": {"J": 1, "J_hidden": -1}, "primary": True},
        {"name": "E4_V_minus_B_size", "weights": {"V": 1, "B": -1}, "primary": True,
         "primary_metric": "size.large_minus_small_sd"},
        {"name": "E5_I_minus_VF", "weights": {"I": 1, "VF": -1}, "primary": True},
    ]

    def add(plus: str, minus: str) -> None:
        out.append({"name": f"{plus}_minus_{minus}", "weights": {plus: 1, minus: -1}})

    for plus, minus in (("V", "A2"), ("J", "A2"), ("A2", "B"), ("V", "V_hidden"), ("J_hidden", "B"),
                        ("VF", "F"), ("I", "V"), ("I", "J"), ("I", "B"), ("F", "B"), ("I", "F")):
        add(plus, minus)
    for plus, minus in (("V", "B"), ("J", "B"), ("J", "J_hidden"), ("I", "VF"), ("I", "B")):
        add(f"decay0_{plus}", f"decay0_{minus}")
    return out


def spec(purpose: str) -> dict:
    stage_f_spec = stage_f.spec("confirmatory", SEEDS["confirmatory"])
    return {
        "name": f"disclosure_targeted_audits_{purpose}_v1", "purpose": purpose, "archive": "compact",
        "_note": "Plan stage E.1, targeted audits (research contract section 21). Development and confirmatory specs "
                 "differ only in name, purpose and seeds; the development run checks the pipeline and may not change "
                 "the design. Primary comparisons E1-E5 are Holm-adjusted.",
        "domain": stage_f_spec["domain"], "market": stage_f_spec["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(), "comparisons": comparisons(), "analysis": stage_f_spec["analysis"],
    }


def reset(row: dict, old_name: str) -> dict:
    """The scenario with its audit options back at the defaults, written as stage F writes such a scenario."""
    row = json.loads(json.dumps(row))
    row["name"] = old_name
    policy = row["policy"]
    if policy["intervention"] is not None:
        for field in AUDIT_FIELDS:
            policy["intervention"].pop(field, None)
        policy["intervention"]["audit_probability"] = policy["baseline"]["audit_probability"]
        if policy["intervention"] == policy["baseline"]:
            policy["intervention"], policy["start_day"] = None, 0
    return row


def check_derivation(stage_f_archive: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    new = {row["name"]: row for row in freeze_spec(ROOT / "configs" / "disclosure_targeted_audits_confirmatory_v1.json")["scenarios"]}
    frozen = json.loads((stage_f_archive / "experiment.json").read_text(encoding="utf-8"))
    old = {row["name"]: row for row in frozen["scenarios"]}
    for name, changes, _, source in rows():
        assert reset(new[name], source) == old[source], f"{name} differs from stage F {source}"
        intervention = new[name]["policy"]["intervention"]
        for field in AUDIT_FIELDS:
            if field in changes:
                assert intervention[field] == changes[field], (name, field)
        if changes.get("audit_allocation") == "jump":
            assert intervention["audit_jump_threshold"] == 0.125, name
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f", type=Path, default=STAGE_F / "confirmatory_v1",
                        help="stage F confirmatory archive (read-only)")
    args = parser.parse_args()
    for purpose in ("development", "confirmatory"):
        path = ROOT / "configs" / f"disclosure_targeted_audits_{purpose}_v1.json"
        path.write_text(json.dumps(spec(purpose), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path, len(scenarios()), "scenarios", len(comparisons()), "comparisons", len(SEEDS[purpose]), "seeds")
    count = check_derivation(args.stage_f)
    print(f"derivation check: {count} scenarios equal their stage F scenario once the audit options are reset")


if __name__ == "__main__":
    main()
