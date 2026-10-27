"""Stage I, rating-persistence check (research contract section 26.22 (b)): write the replay specs and check them.

The stage I v2 archives are compact (public_signal is in COMPACT_OMITTED), so the H2 target moments cannot be read
from them. The model is deterministic: the same scenarios and seeds with archive "sampled" replay the same worlds
and add the quarterly sampled public signal. run_id leaves out purpose and archive, so a replay has the run_id and
the result fingerprint of the run it replays; moments.py checks both.

  spec                                         scenarios                                 seeds
  configs/disclosure_stage_i_moments_v1.json   B, l0_B, llo_B, lhi_B, lfe_B of v2       57001-57030  (150 runs)
  configs/disclosure_stage_i_moments_check_v1.json   the same                           38101-38102  (10 runs)

The check spec replays the v2 development archive check_v2 (pipeline check of the replay and of moments.py only).
The five scenarios are the rating slopes lambda*, lambda0 = 0, lambda_lo, lambda_hi and lambda_FE. Checks: every
frozen scenario equals the frozen v2 scenario of the same name; l0_B equals H2's final-stage scenario q_hat
(configs/disclosure_persistence_final_v1.json) apart from the name; the two specs differ only in name, purpose and
seeds.

  python build_moments_spec.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = ROOT / "configs"
V2 = {"estimation": CONFIGS / "disclosure_stage_i_v2.json", "development": CONFIGS / "disclosure_stage_i_check_v2.json"}
H2_FINAL = CONFIGS / "disclosure_persistence_final_v1.json"
SCENARIOS = {"B": "lambda_star", "l0_B": "lambda0", "llo_B": "lambda_lo", "lhi_B": "lambda_hi", "lfe_B": "lambda_fe"}
NAMES = {"estimation": "disclosure_stage_i_moments_v1", "development": "disclosure_stage_i_moments_check_v1"}


def spec(purpose: str) -> dict:
    base = json.loads(V2[purpose].read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in base["scenarios"]}
    return {
        "name": NAMES[purpose], "purpose": purpose, "archive": "sampled",
        "_note": "Stage I rating-persistence check (research contract section 26.22 (b)): a replay with the sampled "
                 "archive of the B scenarios of the stage I v2 spec on the same seeds (confirmatory seeds: "
                 "disclosure_stage_i_v2; development seeds: disclosure_stage_i_check_v2) at the five rating "
                 "slopes (B lambda*, l0_B "
                 "lambda0 = 0, llo_B lambda_lo, lhi_B lambda_hi, lfe_B lambda_FE), to compute "
                 "the H2 target moments (lags 1, 2, 4, 8, 12 of the quarterly public signal). Each replay must have "
                 "the run_id and result fingerprint of the run it replays. No comparison, no hypothesis test.",
        "financing_rating_slopes": base["financing_rating_slopes"],
        "domain": base["domain"], "market": base["market"], "seeds": base["seeds"],
        "scenarios": [rows[name] for name in SCENARIOS], "comparisons": [], "analysis": base["analysis"],
    }


def check(paths: dict[str, Path]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    new = {row["name"]: row for row in freeze_spec(paths["estimation"])["scenarios"]}
    v2 = {row["name"]: row for row in freeze_spec(V2["estimation"])["scenarios"]}
    assert list(new) == list(SCENARIOS), "scenario list"
    slopes = json.loads(V2["estimation"].read_text(encoding="utf-8"))["financing_rating_slopes"]
    for name, key in SCENARIOS.items():
        assert new[name] == v2[name], f"{name} differs from the frozen v2 scenario"
        assert new[name]["config"].get("financing_rating_slope", 0.0) == slopes[key], name
    q_hat = freeze_spec(H2_FINAL)["scenarios"][0]
    assert q_hat["name"] == "q_hat"
    assert {**new["l0_B"], "name": "q_hat"} == q_hat, "l0_B differs from H2's q_hat apart from the name"
    development = {row["name"]: row for row in freeze_spec(paths["development"])["scenarios"]}
    assert development == new, "development scenarios differ"
    a, b = (json.loads(path.read_text(encoding="utf-8")) for path in (paths["estimation"], paths["development"]))
    assert {k for k in a if a[k] != b[k]} == {"name", "purpose", "seeds"}
    print(f"derivation check: {len(new)} scenarios equal the frozen v2 scenarios; slopes "
          f"{[new[name]['config'].get('financing_rating_slope', 0.0) for name in SCENARIOS]}; l0_B equals H2's q_hat "
          "apart from the name; the development spec differs only in name, purpose and seeds")


def main() -> None:
    paths = {purpose: CONFIGS / f"{name}.json" for purpose, name in NAMES.items()}
    for purpose, path in paths.items():
        payload = spec(purpose)
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path.relative_to(ROOT).as_posix(), len(payload["scenarios"]), "scenarios", len(payload["seeds"]), "seeds")
    check(paths)


if __name__ == "__main__":
    main()
