"""Stage I model runs: write the confirmatory and development specs of contract 26.9 and check their derivation.

The source is the H3 spec configs/disclosure_stage_h3_v1.json (contract 25.17). Only financing_spread changes, as a
scenario-level market override; every other setting is H3's (natural dynamics Q, policy doses, contract scale).

  block          scenarios                                                spread                   names
  confirmatory   H3 main block (18) and perm_ block (3)                   s*                       as in H3
  sensitivity    B, I, F, J, J_hidden, V, ALL_R16, ALL_F16 (cover P1-P8)  s0, s_lo, s_hi, s_28     s0_B, slo_B, ...

The spreads come from calibration_target.json (contract 26.7). Contract 26.9: without a positive estimate, or with
s* within 10% of 0.05, there is no confirmatory block; s_lo exists only when the lower end of the interval is
positive, s_hi only when the upper end is. s0 = 0.05 is the base market's value, so the s0_ scenarios carry no
override and equal H3's scenarios apart from the name. The eight primary questions P1-P8 are H3's (same scenario
names, contrasts, measures and predicted signs), Holm-adjusted together in analyze.py; no comparison is primary for
the runner.

  python build_spec.py [--target calibration_target.json] [--h3-archive <H3 confirmatory archive>]
                       [--out-dir configs]

Writes configs/disclosure_stage_i_v1.json (seeds 57001-57030) and configs/disclosure_stage_i_check_v1.json
(development seeds 38101-38102, pipeline check only); the two differ only in name, purpose and seeds. Then checks
that every scenario, with its spread set back to 0.05, equals H3's frozen scenario of the same base name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H3_DIR = ROOT / "reports" / "disclosure_stage_h3_20261027"
H3_SPEC = ROOT / "configs" / "disclosure_stage_h3_v1.json"

_spec = importlib.util.spec_from_file_location("h3_build_spec", H3_DIR / "build_spec.py")
h3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h3)

SEEDS = {"confirmatory": list(range(57001, 57031)),   # contract 26.9 and 26.10: new, never used
         "development": [38101, 38102]}               # contract 26.9: pipeline check only
CURRENT = 0.05                                        # s0, the base market's financing_spread
CONFIRMATORY = h3.CORE + h3.TARGETED + h3.ASSURED + tuple(f"perm_{name}" for name in h3.REFERENCE)
SENSITIVITY = ("B", "I", "F", "J", "J_hidden", "V", "ALL_R16", "ALL_F16")
PREFIXES = {"s0": "s0_", "s_lo": "slo_", "s_hi": "shi_", "s_28": "s28_"}
NAMES = {"confirmatory": "disclosure_stage_i_v1", "development": "disclosure_stage_i_check_v1"}


def spreads(target: dict) -> tuple[float | None, dict[str, float]]:
    """s* for the confirmatory block (None without one) and the sensitivity spreads, contract 26.9."""
    rerun = target["rerun_rule_26_9"] == "confirmatory rerun at s*"
    s_star = target["s_star_adopted"] if rerun else None
    sensitivity = {key: target["sensitivity_spreads"][key] for key in PREFIXES if key in target["sensitivity_spreads"]}
    assert sensitivity["s0"] == CURRENT
    return s_star, sensitivity


def _with(row: dict, name: str, spread: float) -> dict:
    row = {**json.loads(json.dumps(row)), "name": name}
    assert "market" not in row
    if spread != CURRENT:
        row["market"] = {"financing_spread": spread}
    return row


def scenarios(target: dict) -> list[dict]:
    s_star, sensitivity = spreads(target)
    source = {row["name"]: row for row in json.loads(H3_SPEC.read_text(encoding="utf-8"))["scenarios"]}
    rows = [_with(source[name], name, s_star) for name in CONFIRMATORY] if s_star is not None else []
    for key, spread in sensitivity.items():
        rows += [_with(source[name], PREFIXES[key] + name, spread) for name in SENSITIVITY]
    return rows


def comparisons(target: dict) -> list[dict]:
    s_star, sensitivity = spreads(target)
    rows = []

    def add(name: str, weights: dict, metric: str | None = None) -> None:
        rows.append({"name": name, "weights": weights, **({"primary_metric": metric} if metric else {})})

    if s_star is not None:
        for q in h3.primary_questions():           # P1, P2, P4, P6, P7 at s*
            if q["kind"] == "runner_metric":
                add(q["comparison"], q["weights"], None if q["metric"] == h3.MAIN_METRIC else q["metric"])
        for plus, minus in (("I", "B"), ("V", "A2"), ("J", "A2"), ("F", "F_2x"), ("F_8x", "F"),
                            ("perm_I", "perm_F"), ("perm_F", "perm_B"), ("perm_I", "perm_B")):
            add(f"{plus}_minus_{minus}", {plus: 1, minus: -1})
    for key, prefix in ((key, PREFIXES[key]) for key in sensitivity):
        for q in h3.primary_questions():           # the same runner contrasts at every sensitivity spread
            if q["kind"] == "runner_metric":
                add(prefix + q["comparison"], {prefix + name: weight for name, weight in q["weights"].items()},
                    None if q["metric"] == h3.MAIN_METRIC else q["metric"])
        add(f"{prefix}I_minus_{prefix}B", {prefix + "I": 1, prefix + "B": -1})
    return rows


def spec(purpose: str, target: dict) -> dict:
    base = json.loads(H3_SPEC.read_text(encoding="utf-8"))
    s_star, sensitivity = spreads(target)
    return {
        "name": NAMES[purpose], "purpose": purpose, "archive": "compact",
        "_note": "Stage I (research contract section 26.9): the H3 design (form Q, kappa_T 0.505, s_P 0.423) with "
                 "financing_spread set by the in-sample premium. Confirmatory block at s* (H3 main and perm_ blocks, "
                 "H3's names), sensitivity block of eight core scenarios at s0, s_lo, s_hi and s_28 (prefixed names), "
                 "all on the same seeds. Development and confirmatory specs differ only in name, purpose and seeds; "
                 "the development run checks the pipeline and may not change the design. The Holm family is the "
                 "eight primary_questions at s*, adjusted together in analyze.py; no comparison is primary for the "
                 "runner. The sensitivity block is exploratory.",
        "financing_spreads": {"s_star": s_star, **sensitivity},
        "domain": base["domain"], "market": base["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(target), "comparisons": comparisons(target),
        "primary_questions": h3.primary_questions() if s_star is not None else [],
        "analysis": base["analysis"],
    }


def check_derivation(paths: dict[str, Path], target: dict, h3_archive: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    s_star, sensitivity = spreads(target)
    frozen = freeze_spec(paths["confirmatory"])
    old = {row["name"]: row for row in json.loads((h3_archive / "experiment.json").read_text(encoding="utf-8"))["scenarios"]}
    expected = {name: s_star for name in CONFIRMATORY} if s_star is not None else {}
    for key, spread in sensitivity.items():
        expected.update({PREFIXES[key] + name: spread for name in SENSITIVITY})
    new = {row["name"]: row for row in frozen["scenarios"]}
    assert list(new) == list(expected), "scenario list"
    for name, row in new.items():
        base_name = next((name[len(p):] for p in PREFIXES.values() if name.startswith(p)), name)
        assert row["config"]["financing_spread"] == expected[name], name
        reset = json.loads(json.dumps(row))
        reset["name"], reset["config"]["financing_spread"] = base_name, CURRENT
        assert reset == old[base_name], f"{name} differs from H3 {base_name} once the spread is reset"
    configs = {json.dumps({**row["config"], "financing_spread": None}, sort_keys=True) for row in new.values()}
    assert len(configs) == 1
    check = {row["name"]: row for row in freeze_spec(paths["development"])["scenarios"]}
    assert check == new
    print(f"derivation check: {len(new)} scenarios equal H3's frozen scenario once financing_spread is set back to "
          f"{CURRENT}; spreads {sorted(set(expected.values()))}; development scenarios equal the confirmatory ones")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, default=HERE / "calibration_target.json")
    parser.add_argument("--h3-archive", type=Path, default=H3_DIR / "h3_v1", help="H3 confirmatory archive (read-only)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    target = json.loads(args.target.read_text(encoding="utf-8"))
    paths = {purpose: args.out_dir / f"{name}.json" for purpose, name in NAMES.items()}
    for purpose, path in paths.items():
        payload = spec(purpose, target)
        if args.out_dir.resolve() != (ROOT / "configs").resolve():   # a trial run elsewhere: absolute domain path
            payload["domain"] = str((ROOT / "configs" / payload["domain"]).resolve())
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path, len(payload["scenarios"]), "scenarios", len(payload["comparisons"]), "comparisons",
              len(payload["primary_questions"]), "primary questions", len(payload["seeds"]), "seeds")
    check_derivation(paths, target, args.h3_archive)


if __name__ == "__main__":
    main()
