"""Stage I model runs, v2: write the confirmatory and development specs of contract 26.20 (f) and check their derivation.

The source is the H3 spec configs/disclosure_stage_h3_v1.json (contract 25.17). Only financing_rating_slope (the
direct rating channel lambda of contract 26.20 (a)) changes, as a scenario-level market override; financing_spread
stays 0.05 and every other setting is H3's (natural dynamics Q, policy doses, contract scale).

  block          scenarios                                                slope                                names
  confirmatory   H3 main block (18) and perm_ block (3)                   lambda*                              as in H3
  sensitivity    B, I, F, J, J_hidden, V, ALL_R16, ALL_F16 (cover P1-P8)  lambda0 = 0, lambda_lo, lambda_hi,   l0_B, llo_B,
                                                                          lambda_FE (exploratory, 26.20 (e))   lhi_B, lfe_B

The slopes come from calibration_target_v2.json (contract 26.20 (d), (e)). lambda0 = 0 is the base market's value,
so the l0_ scenarios carry no override and equal H3's scenarios apart from the name. The eight primary questions
P1-P8 are H3's (same scenario names, contrasts, measures and predicted signs), Holm-adjusted together in analyze.py;
no comparison is primary for the runner. The v1 design (financing_spread only, contract 26.7) failed at s* in
development (contract 26.19); its specs and archives are kept and no longer used.

  python build_spec.py [--target calibration_target_v2.json] [--h3-archive <H3 confirmatory archive>]
                       [--out-dir configs]

Writes configs/disclosure_stage_i_v2.json (seeds 57001-57030) and configs/disclosure_stage_i_check_v2.json
(development seeds 38101-38102, pipeline check only); the two differ only in name, purpose and seeds. Then checks
that every scenario, with its slope set back to 0, equals H3's frozen scenario of the same base name.
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

SEEDS = {"confirmatory": list(range(57001, 57031)),   # contract 26.10 and 26.20 (f): never used in any run
         "development": [38101, 38102]}               # contract 26.20 (f): pipeline check only
CURRENT = 0.0                                         # lambda0, the base market's financing_rating_slope
CONFIRMATORY = h3.CORE + h3.TARGETED + h3.ASSURED + tuple(f"perm_{name}" for name in h3.REFERENCE)
SENSITIVITY = ("B", "I", "F", "J", "J_hidden", "V", "ALL_R16", "ALL_F16")
PREFIXES = {"lambda0": "l0_", "lambda_lo": "llo_", "lambda_hi": "lhi_", "lambda_fe": "lfe_"}
NAMES = {"confirmatory": "disclosure_stage_i_v2", "development": "disclosure_stage_i_check_v2"}


def slopes(target: dict) -> tuple[float, dict[str, float]]:
    """lambda* for the confirmatory block and the sensitivity slopes, contract 26.20 (d), (e)."""
    assert target["rerun_rule_26_20"] == "confirmatory rerun at lambda*"
    values = target["financing_rating_slope"]
    sensitivity = {"lambda0": CURRENT, **{key: values[key] for key in ("lambda_lo", "lambda_hi", "lambda_fe")}}
    assert all(value > 0 for key, value in sensitivity.items() if key != "lambda0") and values["lambda_star"] > 0
    return values["lambda_star"], sensitivity


def _with(row: dict, name: str, slope: float) -> dict:
    row = {**json.loads(json.dumps(row)), "name": name}
    assert "market" not in row
    if slope != CURRENT:
        row["market"] = {"financing_rating_slope": slope}
    return row


def scenarios(target: dict) -> list[dict]:
    star, sensitivity = slopes(target)
    source = {row["name"]: row for row in json.loads(H3_SPEC.read_text(encoding="utf-8"))["scenarios"]}
    rows = [_with(source[name], name, star) for name in CONFIRMATORY]
    for key, slope in sensitivity.items():
        rows += [_with(source[name], PREFIXES[key] + name, slope) for name in SENSITIVITY]
    return rows


def comparisons(target: dict) -> list[dict]:
    _, sensitivity = slopes(target)
    rows = []

    def add(name: str, weights: dict, metric: str | None = None) -> None:
        rows.append({"name": name, "weights": weights, **({"primary_metric": metric} if metric else {})})

    for q in h3.primary_questions():               # P1, P2, P4, P6, P7 at lambda*
        if q["kind"] == "runner_metric":
            add(q["comparison"], q["weights"], None if q["metric"] == h3.MAIN_METRIC else q["metric"])
    for plus, minus in (("I", "B"), ("V", "A2"), ("J", "A2"), ("F", "F_2x"), ("F_8x", "F"),
                        ("perm_I", "perm_F"), ("perm_F", "perm_B"), ("perm_I", "perm_B")):
        add(f"{plus}_minus_{minus}", {plus: 1, minus: -1})
    for key, prefix in ((key, PREFIXES[key]) for key in sensitivity):
        for q in h3.primary_questions():           # the same runner contrasts at every sensitivity slope
            if q["kind"] == "runner_metric":
                add(prefix + q["comparison"], {prefix + name: weight for name, weight in q["weights"].items()},
                    None if q["metric"] == h3.MAIN_METRIC else q["metric"])
        add(f"{prefix}I_minus_{prefix}B", {prefix + "I": 1, prefix + "B": -1})
    return rows


def spec(purpose: str, target: dict) -> dict:
    base = json.loads(H3_SPEC.read_text(encoding="utf-8"))
    star, sensitivity = slopes(target)
    return {
        "name": NAMES[purpose], "purpose": purpose, "archive": "compact",
        "_note": "Stage I v2 (research contract section 26.20): the H3 design (form Q, kappa_T 0.505, s_P 0.423, "
                 "financing_spread 0.05) with a direct response of the debt rate to the public rating, "
                 "financing_rating_slope lambda, set by the in-sample premium. Confirmatory block at lambda* (H3 "
                 "main and perm_ blocks, H3's names), sensitivity block of eight core scenarios at lambda0 = 0, "
                 "lambda_lo, lambda_hi and lambda_FE (prefixed names; lambda_FE is an exploratory point added after "
                 "the estimate), all on the same seeds. Development and confirmatory specs differ only in name, "
                 "purpose and seeds; the development run checks the pipeline and may not change the design. The "
                 "Holm family is the eight primary_questions at lambda*, adjusted together in analyze.py; no "
                 "comparison is primary for the runner. The sensitivity block is exploratory.",
        "financing_rating_slopes": {"lambda_star": star, **sensitivity},
        "domain": base["domain"], "market": base["market"], "seeds": SEEDS[purpose],
        "scenarios": scenarios(target), "comparisons": comparisons(target),
        "primary_questions": h3.primary_questions(),
        "analysis": base["analysis"],
    }


def check_derivation(paths: dict[str, Path], target: dict, h3_archive: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    star, sensitivity = slopes(target)
    frozen = freeze_spec(paths["confirmatory"])
    old = {row["name"]: row for row in json.loads((h3_archive / "experiment.json").read_text(encoding="utf-8"))["scenarios"]}
    expected = {name: star for name in CONFIRMATORY}
    for key, slope in sensitivity.items():
        expected.update({PREFIXES[key] + name: slope for name in SENSITIVITY})
    new = {row["name"]: row for row in frozen["scenarios"]}
    assert list(new) == list(expected), "scenario list"
    for name, row in new.items():
        base_name = next((name[len(p):] for p in PREFIXES.values() if name.startswith(p)), name)
        # A zero slope is left out of the frozen config (MarketConfig.to_dict), so l0_ rows carry no key.
        assert row["config"].get("financing_rating_slope", CURRENT) == expected[name], name
        assert row["config"]["financing_spread"] == 0.05, name
        reset = json.loads(json.dumps(row))
        reset["name"] = base_name
        reset["config"].pop("financing_rating_slope", None)
        assert reset == old[base_name], f"{name} differs from H3 {base_name} once the slope is reset"
    configs = {json.dumps({k: v for k, v in row["config"].items() if k != "financing_rating_slope"}, sort_keys=True)
               for row in new.values()}
    assert len(configs) == 1
    check = {row["name"]: row for row in freeze_spec(paths["development"])["scenarios"]}
    assert check == new
    print(f"derivation check: {len(new)} scenarios equal H3's frozen scenario once financing_rating_slope is set "
          f"back to 0; slopes {sorted(set(expected.values()))}; development scenarios equal the confirmatory ones")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, default=HERE / "calibration_target_v2.json")
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
