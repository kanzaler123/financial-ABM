"""Joint robustness test: write the confirmatory and development specs for contract 24 and check their derivation.

Two assumptions of the stage F main setting are changed together, mean reversion of true values (R) and one fixed
fine per inflated report (P), and then with firm-level wash cost ratios (W) as well. Three reference blocks rerun
the main setting and each single change on the same new seeds, so every contrast pairs within one run.

  block        dynamics             fine basis   w            prefix
  main         uniform decay 0.2    case         0.2          (none)
  P only       uniform decay 0.2    report       0.2          rep_
  R only       reversion 0.32/0.5   case         0.2          rev0p32_, rev0p5_
  R x P        reversion 0.32/0.5   report       0.2          rr0p32_, rr0p5_
  R x P x W    reversion 0.32/0.5   report       lognormal    rrw0p32_, rrw0p5_

Each block has B, I (tau 0.6) and F (fixed fine 4x); 24 scenarios. The option values are those of contract 20.21.

  python build_spec.py [--stage-f <stage F confirmatory archive>]

Writes configs/disclosure_joint_robustness_v1.json (confirmatory, seeds 44001-44030) and
configs/disclosure_joint_robustness_check_v1.json (development seeds 33001-33002, pipeline check only); the two
differ only in name, purpose and seeds. Then checks that every scenario, with its options set back to the defaults,
equals the frozen stage F scenario it derives from, and that the R-only and P-only scenarios equal those of the
contract 20.21 spec.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE_F = ROOT / "reports" / "disclosure_stage_f_20261014"

_spec = importlib.util.spec_from_file_location("stage_f_build_specs", STAGE_F / "build_specs.py")
stage_f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_f)

SEEDS = {"confirmatory": list(range(44001, 44031)),   # new, never used (contract 24.1)
         "development": [33001, 33002]}               # stage F development seeds; pipeline check only
REVERSIONS = (0.32, 0.5)                               # contract 20.21 group R
WASH_LOG_SD = math.log(2)                              # contract 20.21 group W
POLICIES = {name: stage_f.POLICIES[name] for name in ("B", "I", "F")}
SPEC_PATHS = {"confirmatory": ROOT / "configs" / "disclosure_joint_robustness_v1.json",
              "development": ROOT / "configs" / "disclosure_joint_robustness_check_v1.json"}
ROBUSTNESS_SPEC = ROOT / "configs" / "disclosure_robustness_v1.json"
REPORT = {"fixed_fine_basis": "report"}


def blocks() -> dict[str, tuple[dict | None, dict | None]]:
    """Prefix -> (domain changes, baseline policy changes)."""
    out: dict = {"": (None, None), "rep_": (None, REPORT)}
    for kappa in REVERSIONS:
        reversion = {"natural_decay": 0.0, "natural_reversion": kappa}
        tag = stage_f.label(kappa)
        out[f"rev{tag}_"] = (reversion, None)
        out[f"rr{tag}_"] = (reversion, REPORT)
        out[f"rrw{tag}_"] = ({**reversion, "wash_cost_log_sd": WASH_LOG_SD}, REPORT)
    return out


JOINT = [f"rr{stage_f.label(k)}_" for k in REVERSIONS] + [f"rrw{stage_f.label(k)}_" for k in REVERSIONS]


def scenarios() -> list[dict]:
    return [stage_f.scenario(prefix + name, changes or None, domain=domain, baseline=baseline)
            for prefix, (domain, baseline) in blocks().items() for name, changes in POLICIES.items()]


def comparisons() -> list[dict]:
    rows = []

    def add(name: str, weights: dict, primary: bool = False) -> None:
        row = {"name": name, "weights": weights}
        if primary:
            row["primary"] = True
        rows.append(row)

    # Contract 24.5: eight primary questions, I - F and F - B in each joint setting.
    for prefix in JOINT:
        add(f"J_{prefix}I_minus_F", {prefix + "I": 1, prefix + "F": -1}, primary=True)
        add(f"J_{prefix}F_minus_B", {prefix + "F": 1, prefix + "B": -1}, primary=True)
    # Contract 24.7, auxiliary.
    for prefix in blocks():
        add(f"{prefix}I_minus_B", {prefix + "I": 1, prefix + "B": -1})
        if prefix not in JOINT:
            for plus, minus in (("I", "F"), ("F", "B")):
                add(f"{prefix}{plus}_minus_{minus}", {prefix + plus: 1, prefix + minus: -1})
    for kappa in REVERSIONS:
        tag = stage_f.label(kappa)
        for plus, minus in (("I", "F"), ("F", "B")):
            # (R x P) - (R only) - (P only) + (main)
            add(f"interaction_{tag}_{plus}_minus_{minus}", {
                f"rr{tag}_{plus}": 1, f"rr{tag}_{minus}": -1, f"rev{tag}_{plus}": -1, f"rev{tag}_{minus}": 1,
                f"rep_{plus}": -1, f"rep_{minus}": 1, plus: 1, minus: -1})
            add(f"wash_cost_spread_{tag}_{plus}_minus_{minus}", {
                f"rrw{tag}_{plus}": 1, f"rrw{tag}_{minus}": -1, f"rr{tag}_{plus}": -1, f"rr{tag}_{minus}": 1})
    return rows


def spec(purpose: str) -> dict:
    stage_f_spec = stage_f.spec("confirmatory", SEEDS["confirmatory"])
    return {
        "name": SPEC_PATHS[purpose].stem, "purpose": purpose, "archive": "compact",
        "_note": "Joint robustness test (research contract section 24): mean reversion x one fixed fine per inflated "
                 "report, and with firm-level wash cost ratios, plus reference blocks on the same seeds. Development "
                 "and confirmatory specs differ only in name, purpose and seeds; the development run checks the "
                 "pipeline and may not change the design.",
        "domain": stage_f_spec["domain"], "market": stage_f_spec["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(), "comparisons": comparisons(), "analysis": stage_f_spec["analysis"],
    }


def reset(row: dict, name: str) -> dict:
    """The scenario with every option of this test set back to its default (omitted, as the serializers do)."""
    row = json.loads(json.dumps(row))
    row["name"] = name
    domain = row["domain"]
    if domain.pop("natural_reversion", None) is not None:
        assert domain["natural_decay"] == 0.0
    if "wash_cost_log_sd" in domain:
        assert domain.pop("wash_cost_log_sd") == WASH_LOG_SD
    for regime in ("baseline", "intervention"):
        if row["policy"][regime] is not None and "fixed_fine_basis" in row["policy"][regime]:
            assert row["policy"][regime].pop("fixed_fine_basis") == "report"
    return row


def check_derivation(stage_f_archive: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    new = {row["name"]: row for row in freeze_spec(SPEC_PATHS["confirmatory"])["scenarios"]}
    frozen = json.loads((stage_f_archive / "experiment.json").read_text(encoding="utf-8"))
    old = {row["name"]: row for row in frozen["scenarios"]}
    count = 0
    for prefix, (domain, _) in blocks().items():
        for name in POLICIES:
            source = f"decay0_{name}" if domain else name
            assert reset(new[prefix + name], source) == old[source], f"{prefix}{name} differs from stage F {source}"
            count += 1
    print(f"derivation check: {count} scenarios equal their stage F scenario once the options are reset")
    robustness = {row["name"]: row for row in freeze_spec(ROBUSTNESS_SPEC)["scenarios"]}
    shared = [prefix + name for prefix in blocks() if prefix.startswith(("rev", "rep")) for name in POLICIES]
    for name in shared:
        assert new[name] == robustness[name], f"{name} differs from the contract 20.21 scenario"
    print(f"derivation check: {len(shared)} R-only and P-only scenarios equal those of contract 20.21")
    # The joint blocks differ from the R-only block only in the fine basis (and the wash cost spread).
    for kappa in REVERSIONS:
        tag = stage_f.label(kappa)
        for name in POLICIES:
            joint = reset(new[f"rr{tag}_{name}"], name)
            triple = reset(new[f"rrw{tag}_{name}"], name)
            single = reset(new[f"rev{tag}_{name}"], name)
            assert joint == single == triple
    print("derivation check: joint, triple and R-only blocks agree once the options are reset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f", type=Path, default=STAGE_F / "confirmatory_v1",
                        help="stage F confirmatory archive (read-only)")
    args = parser.parse_args()
    for purpose, path in SPEC_PATHS.items():
        path.write_text(json.dumps(spec(purpose), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path, len(scenarios()), "scenarios", len(comparisons()), "comparisons",
              sum(row.get("primary", False) for row in comparisons()), "primary", len(SEEDS[purpose]), "seeds")
    check_derivation(args.stage_f)


if __name__ == "__main__":
    main()
