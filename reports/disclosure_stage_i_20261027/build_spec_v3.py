"""Stage I v3 (research contract section 26.24 (f)): the confirmatory and development specs at the v3 dynamics.

The source is the frozen v2 spec configs/disclosure_stage_i_v2.json (contract 26.21). Only the natural dynamics of
every scenario change, to the v3 estimate theta-hat of estimation_v3.json (selected form, parameters to three
decimals); the archive becomes "sampled" so that the rating moments can be read from the archive (26.24 (f)); the
seeds are new. lambda* is the v2 value unless calibration_verify_v3.json adjusted it (26.24 (b)), in which case the
lambda* block uses the adjusted slope. The sensitivity slopes are those of 26.21. With form U the perm_ block has no
meaning (improvements are already permanent) and is dropped with its comparisons; with R or Q it keeps
reversion_improvement = "permanent".

  python build_spec_v3.py

Writes configs/disclosure_stage_i_v3.json (seeds 60001-60030) and configs/disclosure_stage_i_check_v3.json
(development seeds 38101-38102); the two differ only in name, purpose and seeds. Checks that every scenario, with
its dynamics (and lambda*, if adjusted) set back, equals the frozen v2 scenario of the same name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = ROOT / "configs"
V2 = CONFIGS / "disclosure_stage_i_v2.json"
SEEDS = {"confirmatory": list(range(60001, 60031)),   # contract 26.24 (f), (j)
         "development": [38101, 38102]}               # pipeline check only
NAMES = {"confirmatory": "disclosure_stage_i_v3", "development": "disclosure_stage_i_check_v3"}
DYNAMICS = ("natural_decay", "natural_reversion", "reversion_permanent_share")
V2_DYNAMICS = {"natural_decay": 0.0, "natural_reversion": 0.505, "reversion_permanent_share": 0.423}
KEYS = {"delta": "natural_decay", "kappa": "natural_reversion", "kappa_T": "natural_reversion",
        "s_P": "reversion_permanent_share"}


def v3_dynamics() -> tuple[str, dict]:
    estimation = json.loads((HERE / "estimation_v3.json").read_text(encoding="utf-8"))
    selected = estimation["selected"]
    assert estimation["proceed_26_24_e"]["proceed"], "contract 26.24 (e) does not allow the confirmatory rerun"
    dynamics = {"natural_decay": 0.0} if selected != "U" else {}
    dynamics.update({KEYS[k]: round(v, 3) for k, v in estimation["forms"][selected]["estimate"].items() if k != "sigma_m"})
    return selected, dynamics


def slopes(v2: dict) -> dict:
    out = dict(v2["financing_rating_slopes"])
    check = json.loads((HERE / "calibration_verify_v3.json").read_text(encoding="utf-8"))
    if not check["within_10pct"]:
        out["lambda_star"] = float(f"{check['adjusted_slope_bp']!r}e-4")
    return out


def spec(purpose: str) -> dict:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    selected, dynamics = v3_dynamics()
    values = slopes(v2)
    rows = []
    for row in v2["scenarios"]:
        if selected == "U" and row["name"].startswith("perm_"):
            continue
        row = json.loads(json.dumps(row))
        domain = {k: v for k, v in row["domain"].items() if k not in DYNAMICS}
        row["domain"] = {**dynamics, **domain}
        if row.get("market", {}).get("financing_rating_slope") == v2["financing_rating_slopes"]["lambda_star"]:
            row["market"]["financing_rating_slope"] = values["lambda_star"]
        rows.append(row)
    names = {row["name"] for row in rows}
    comparisons = [c for c in v2["comparisons"] if set(c["weights"]) <= names]
    return {
        "name": NAMES[purpose], "purpose": purpose, "archive": "sampled",
        "_note": "Stage I v3 (research contract section 26.24): the v2 design (H3's scenarios, the lambda* block "
                 "and the sensitivity block at lambda0, lambda_lo, lambda_hi and lambda_FE) with the natural "
                 f"dynamics re-estimated under option A at lambda* (form {selected}, {json.dumps(dynamics)}), "
                 "sampled archive. Development and confirmatory specs differ only in name, purpose and seeds; the "
                 "development run checks the pipeline and may not change the design. The Holm family is the eight "
                 "primary_questions at lambda*, adjusted together in analyze.py; the sensitivity block is "
                 "exploratory.",
        "natural_dynamics": {"form": selected, **dynamics},
        "financing_rating_slopes": values,
        "domain": v2["domain"], "market": v2["market"], "seeds": SEEDS[purpose],
        "scenarios": rows, "comparisons": comparisons, "primary_questions": v2["primary_questions"],
        "analysis": v2["analysis"],
    }


def check(paths: dict[str, Path]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    v2 = json.loads(V2.read_text(encoding="utf-8"))
    old = {row["name"]: row for row in freeze_spec(V2)["scenarios"]}
    new = {row["name"]: row for row in freeze_spec(paths["confirmatory"])["scenarios"]}
    selected, dynamics = v3_dynamics()
    star_v2, star = v2["financing_rating_slopes"]["lambda_star"], slopes(v2)["lambda_star"]
    for name, row in new.items():
        for key in DYNAMICS:
            assert row["domain"].get(key, 0.0) == dynamics.get(key, 0.0), (name, key)
        reset = json.loads(json.dumps(row))
        for key in DYNAMICS:
            reset["domain"].pop(key, None)
        if reset["config"].get("financing_rating_slope") == star:
            reset["config"]["financing_rating_slope"] = star_v2
        base = json.loads(json.dumps(old[name]))
        for key in DYNAMICS:
            base["domain"].pop(key, None)
        assert reset == base, f"{name} differs from v2 beyond the dynamics and lambda*"
    dropped = sorted(set(old) - set(new))
    assert all(name.startswith("perm_") for name in dropped) and (bool(dropped) == (selected == "U"))
    development = {row["name"]: row for row in freeze_spec(paths["development"])["scenarios"]}
    assert development == new
    a, b = (json.loads(path.read_text(encoding="utf-8")) for path in paths.values())
    assert {k for k in a if a[k] != b[k]} == {"name", "purpose", "seeds"}
    print(f"derivation check: {len(new)} scenarios equal v2's once the dynamics ({selected}, {dynamics}) and "
          f"lambda* ({star} vs {star_v2}) are set back; dropped {dropped or 'none'}; development spec differs "
          "only in name, purpose and seeds")


def main() -> None:
    paths = {purpose: CONFIGS / f"{name}.json" for purpose, name in NAMES.items()}
    for purpose, path in paths.items():
        payload = spec(purpose)
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path.relative_to(ROOT).as_posix(), len(payload["scenarios"]), "scenarios",
              len(payload["comparisons"]), "comparisons", len(payload["seeds"]), "seeds")
    check(paths)


if __name__ == "__main__":
    main()
