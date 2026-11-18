"""Stage H3: write the confirmatory and development specs for contract 25.8 and check their derivation.

The paper's main results are rerun at the stage H point estimate of the natural dynamics, the permanent-plus-
transitory form Q (contract 25.16): natural_decay = 0, natural_reversion = 0.505, reversion_permanent_share = 0.423.
Policy doses are those of the original experiments.

  block                     scenarios                                   dynamics                 source
  core (stage F)            B, I, F, IF, F_2x, F_8x                     Q                        stage F
  targeted audits (21)      A2, V, V_hidden, J, J_hidden, VF            Q                        contract 21 spec
  assurance (22)            ALL_R1, ALL_R16, ALL_F1, ALL_F16,           Q                        contract 22 spec
                            LRG_R16, LRG_F16
  reference                 u0p2_B, u0p2_I, u0p2_F                      uniform decay 0.2        stage F
  permanent improvement     perm_B, perm_I, perm_F                      Q, improvements stay     stage F

24 scenarios. The eight primary questions P1-P8 (contract 25.8) are listed under "primary_questions": five are
runner contrasts of the comparisons below, three are washing shares that analyze.py computes from the archive. The
Holm family is all eight, so analyze.py adjusts them together and no comparison is marked primary for the runner.

  python build_spec.py [--stage-f <stage F confirmatory archive>] [--audits <contract 21 archive>]
                       [--assurance <contract 22 archive>]

Writes configs/disclosure_stage_h3_v1.json (confirmatory, seeds 53001-53030) and
configs/disclosure_stage_h3_check_v1.json (development seeds 36001-36002, pipeline check only); the two differ only
in name, purpose and seeds. Then checks that every scenario, with its natural dynamics set back to the original
ones, equals the frozen scenario of the experiment it derives from, and that the reference block equals stage F
exactly.
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
AUDITS = ROOT / "reports" / "disclosure_targeted_audits_20261020"
ASSURANCE = ROOT / "reports" / "disclosure_assurance_20261021"

_spec = importlib.util.spec_from_file_location("stage_f_build_specs", STAGE_F / "build_specs.py")
stage_f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_f)

SEEDS = {"confirmatory": list(range(53001, 53031)),   # contract 25.8 and 25.14: new, never used
         "development": [36001, 36002]}               # contract 25.8: pipeline check only
# Contract 25.16: the point estimate of Q to three decimals.
Q = {"natural_decay": 0.0, "natural_reversion": 0.505, "reversion_permanent_share": 0.423}
PERMANENT = {**Q, "reversion_improvement": "permanent"}
DYNAMICS = ("natural_decay", "natural_reversion", "reversion_permanent_share", "reversion_improvement")
SPEC_PATHS = {"confirmatory": ROOT / "configs" / "disclosure_stage_h3_v1.json",
              "development": ROOT / "configs" / "disclosure_stage_h3_check_v1.json"}
AUDIT_SPEC = ROOT / "configs" / "disclosure_targeted_audits_confirmatory_v1.json"
ASSURANCE_SPEC = ROOT / "configs" / "disclosure_assurance_confirmatory_v1.json"
CORE = ("B", "I", "F", "IF", "F_2x", "F_8x")
TARGETED = ("A2", "V", "V_hidden", "J", "J_hidden", "VF")
ASSURED = ("ALL_R1", "ALL_R16", "ALL_F1", "ALL_F16", "LRG_R16", "LRG_F16")
REFERENCE = ("B", "I", "F")
MAIN_METRIC = "true_change.equal_weight_sd"
SIZE_METRIC = "size.large_minus_small_sd"


def _rows(path: Path, names: tuple[str, ...]) -> dict[str, dict]:
    """Raw scenario rows (name, policy) of a committed spec."""
    rows = {row["name"]: row for row in json.loads(path.read_text(encoding="utf-8"))["scenarios"]}
    return {name: json.loads(json.dumps(rows[name])) for name in names}


def _core_row(name: str) -> dict:
    changes = stage_f.POLICIES.get(name)
    if changes is None:   # F_2x, F_8x: the stage F dose-response block
        changes = {"fixed_fine": int(name[2:-1]) * stage_f.F1}
    return stage_f.scenario(name, changes or None)


def _with(row: dict, name: str, domain: dict | None) -> dict:
    row = {**json.loads(json.dumps(row)), "name": name}
    if domain:
        assert "domain" not in row
        row["domain"] = dict(domain)
    return row


def scenarios() -> list[dict]:
    rows = [_with(_core_row(name), name, Q) for name in CORE]
    rows += [_with(row, name, Q) for name, row in _rows(AUDIT_SPEC, TARGETED).items()]
    rows += [_with(row, name, Q) for name, row in _rows(ASSURANCE_SPEC, ASSURED).items()]
    rows += [_with(_core_row(name), f"u0p2_{name}", None) for name in REFERENCE]
    rows += [_with(_core_row(name), f"perm_{name}", PERMANENT) for name in REFERENCE]
    return rows


def primary_questions() -> list[dict]:
    """Contract 25.8: two-sided, Holm-adjusted together; direction is the predicted sign."""
    runner = lambda key, name, weights, metric, direction: {  # noqa: E731
        "id": key, "comparison": name, "kind": "runner_metric", "weights": weights, "metric": metric,
        "direction": direction}
    share = lambda key, name, weights, share_name, direction, text: {  # noqa: E731
        "id": key, "comparison": name, "kind": "wash_decision_share", "weights": weights, "share": share_name,
        "direction": direction, "definition": text}
    return [
        runner("P1", "P1_I_minus_F", {"I": 1, "F": -1}, MAIN_METRIC, 1),
        runner("P2", "P2_F_minus_B", {"F": 1, "B": -1}, MAIN_METRIC, 1),
        share("P3", "P3_F_minus_B_large_share", {"F": 1, "B": -1}, "large", 1,
              "large firms' washing decisions / all washing decisions, decision days 1260-3480"),
        runner("P4", "P4_J_minus_J_hidden", {"J": 1, "J_hidden": -1}, MAIN_METRIC, -1),
        share("P5", "P5_V_minus_B_small_share", {"V": 1, "B": -1}, "small", 1,
              "small firms' washing decisions / all washing decisions, decision days 1260-3480"),
        runner("P6", "P6_V_minus_B_size", {"V": 1, "B": -1}, SIZE_METRIC, 1),
        runner("P7", "P7_ALL_F16_minus_ALL_R16", {"ALL_F16": 1, "ALL_R16": -1}, MAIN_METRIC, -1),
        share("P8", "P8_ALL_F16_minus_ALL_R16_laxest_share", {"ALL_F16": 1, "ALL_R16": -1}, "laxest", 1,
              "assured firms' washing decisions made by clients of the provider with that year's lowest quality "
              "/ all washing decisions of assured firms, decision days 1260-3480"),
    ]


def comparisons() -> list[dict]:
    rows = [{"name": q["comparison"], "weights": q["weights"],
             **({"primary_metric": q["metric"]} if q["metric"] != MAIN_METRIC else {})}
            for q in primary_questions() if q["kind"] == "runner_metric"]

    def add(plus: str, minus: str) -> None:
        rows.append({"name": f"{plus}_minus_{minus}", "weights": {plus: 1, minus: -1}})

    # Contract 25.8, auxiliary (unadjusted).
    add("I", "B")
    add("V", "A2")
    add("J", "A2")
    add("F", "F_2x")
    add("F_8x", "F")
    for name in REFERENCE:   # estimate vs the original main setting, seed by seed
        add(name, f"u0p2_{name}")
    for prefix in ("u0p2_", "perm_"):
        add(prefix + "I", prefix + "F")
        add(prefix + "F", prefix + "B")
        add(prefix + "I", prefix + "B")
    return rows


def spec(purpose: str) -> dict:
    stage_f_spec = stage_f.spec("confirmatory", list(range(41001, 41031)))
    return {
        "name": SPEC_PATHS[purpose].stem, "purpose": purpose, "archive": "compact",
        "_note": "Stage H3 (research contract section 25.8): the paper's main results rerun at the stage H point "
                 "estimate of the natural dynamics (form Q, kappa_T 0.505, s_P 0.423), with a reference block under "
                 "the original uniform decay 0.2 and a block where improvements do not revert. Development and "
                 "confirmatory specs differ only in name, purpose and seeds; the development run checks the "
                 "pipeline and may not change the design. The Holm family is the eight primary_questions, adjusted "
                 "together in analyze.py; no comparison is primary for the runner.",
        "domain": stage_f_spec["domain"], "market": stage_f_spec["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(), "comparisons": comparisons(), "primary_questions": primary_questions(),
        "analysis": stage_f_spec["analysis"],
    }


def reset(row: dict, name: str, decay: float) -> dict:
    """The scenario with its natural dynamics set back to the original ones."""
    row = json.loads(json.dumps(row))
    row["name"] = name
    domain = row["domain"]
    assert domain["natural_decay"] == 0.0 and domain["natural_reversion"] == Q["natural_reversion"]
    assert domain["reversion_permanent_share"] == Q["reversion_permanent_share"]
    for key in DYNAMICS:
        domain.pop(key, None)
    domain["natural_decay"] = decay
    return row


def _frozen(archive: Path) -> dict[str, dict]:
    return {row["name"]: row for row in json.loads((archive / "experiment.json").read_text(encoding="utf-8"))["scenarios"]}


def check_derivation(stage_f_archive: Path, audits_archive: Path, assurance_archive: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    frozen = freeze_spec(SPEC_PATHS["confirmatory"])
    new = {row["name"]: row for row in frozen["scenarios"]}
    assert list(new) == [row["name"] for row in scenarios()] and len(new) == 24
    sources = {"stage F": _frozen(stage_f_archive), "contract 21": _frozen(audits_archive),
               "contract 22": _frozen(assurance_archive)}
    blocks = [("stage F", name, name) for name in CORE] + [("contract 21", name, name) for name in TARGETED]
    blocks += [("contract 22", name, name) for name in ASSURED] + [("stage F", f"perm_{n}", n) for n in REFERENCE]
    for source, name, original in blocks:
        old = sources[source][original]
        row = new[name]
        expected = {**Q, **({"reversion_improvement": "permanent"} if name.startswith("perm_") else {})}
        assert {key: row["domain"].get(key) for key in DYNAMICS if key in row["domain"]} == expected, name
        assert reset(row, original, old["domain"]["natural_decay"]) == old, f"{name} differs from {source} {original}"
    for name in REFERENCE:
        assert {**new[f"u0p2_{name}"], "name": name} == sources["stage F"][name], f"u0p2_{name} differs from stage F"
    # Every main-block scenario shares one domain apart from its policy; the original experiments share stage F's.
    domains = {json.dumps(new[name]["domain"], sort_keys=True) for name in CORE + TARGETED + ASSURED}
    configs = {json.dumps(row["config"], sort_keys=True) for row in new.values()}
    assert len(domains) == 1 and len(configs) == 1
    check = {row["name"]: row for row in freeze_spec(SPEC_PATHS["development"])["scenarios"]}
    assert check == new
    print(f"derivation check: {len(blocks)} scenarios equal their original scenario once the dynamics are reset "
          f"(stage F {len(CORE) + len(REFERENCE)}, contract 21 {len(TARGETED)}, contract 22 {len(ASSURED)}); "
          f"u0p2_ block equals stage F B, I, F exactly; development scenarios equal the confirmatory ones")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f", type=Path, default=STAGE_F / "confirmatory_v1",
                        help="stage F confirmatory archive (read-only)")
    parser.add_argument("--audits", type=Path, default=AUDITS / "audits_v1",
                        help="contract 21 confirmatory archive (read-only)")
    parser.add_argument("--assurance", type=Path, default=ASSURANCE / "assurance_v1",
                        help="contract 22 confirmatory archive (read-only)")
    args = parser.parse_args()
    for purpose, path in SPEC_PATHS.items():
        payload = spec(purpose)
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path, len(payload["scenarios"]), "scenarios", len(payload["comparisons"]), "comparisons",
              len(payload["primary_questions"]), "primary questions", len(payload["seeds"]), "seeds")
    check_derivation(args.stage_f, args.audits, args.assurance)


if __name__ == "__main__":
    main()
