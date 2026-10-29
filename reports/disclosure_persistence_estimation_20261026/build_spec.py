"""Stage H estimation grid: write the estimation and development specs for contract 25 and check their derivation.

Every scenario is the stage F baseline B (no policy switch, main setting of contract 20.1) with one natural dynamic
of the true level (contract 25.2):

  form   domain changes                                                     grid
  U      natural_decay = delta                                               delta in 0, 0.05, 0.1, 0.15, 0.2, 0.3
  R      natural_decay = 0, natural_reversion = kappa                        kappa in 0.1, ..., 0.6, 0.8, 1.0
  Q      natural_decay = 0, natural_reversion = kappa_T,                     kappa_T in 0.25, ..., 3.0 x
         reversion_permanent_share = s_P                                     s_P in 0.1, ..., 0.7

The archive is "sampled": the public signal and the true level every 63 trading days from day 1260 (contract 25.3).

  python build_spec.py [--stage-f <stage F confirmatory archive>]

Writes configs/disclosure_persistence_grid_v1.json (purpose estimation, seeds 48001-48010 for every grid point,
contract 25.3) and configs/disclosure_persistence_grid_check_v1.json (development seeds 36001-36002, one point per
form, pipeline check only). Then checks that every scenario, with its dynamics set back to the stage F values, equals
the frozen stage F scenario B, and that U at 0.2 and 0 equal stage F B and decay0_B exactly.
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

DECAYS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
KAPPAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
KAPPAS_T = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SEEDS = {"estimation": list(range(48001, 48011)),   # contract 25.3, grid stage
         "development": [36001, 36002]}             # contract 25.10, pipeline check only
CHECK_POINTS = ("u_d0p2", "r_k0p5", "q_k1_s0p4")     # one grid point per form for the development run
SPEC_PATHS = {"estimation": ROOT / "configs" / "disclosure_persistence_grid_v1.json",
              "development": ROOT / "configs" / "disclosure_persistence_grid_check_v1.json"}
DYNAMICS = ("natural_decay", "natural_reversion", "reversion_permanent_share")


def grid() -> dict[str, dict]:
    """Scenario name -> domain changes, in the order U, R, Q."""
    points = {f"u_d{stage_f.label(d)}": {"natural_decay": d} for d in DECAYS}
    points.update({f"r_k{stage_f.label(k)}": {"natural_decay": 0.0, "natural_reversion": k} for k in KAPPAS})
    points.update({f"q_k{stage_f.label(k)}_s{stage_f.label(s)}": {
        "natural_decay": 0.0, "natural_reversion": k, "reversion_permanent_share": s} for k in KAPPAS_T for s in SHARES})
    return points


def spec(purpose: str) -> dict:
    stage_f_spec = stage_f.spec("confirmatory", list(range(41001, 41031)))
    names = list(grid()) if purpose == "estimation" else list(CHECK_POINTS)
    return {
        "name": SPEC_PATHS[purpose].stem, "purpose": purpose, "archive": "sampled",
        "_note": "Stage H persistence estimation (research contract section 25): the stage F baseline B under each "
                 "natural dynamic of the grid, archived with the public signal sampled every 63 trading days from "
                 "day 1260. The development spec runs one grid point per form on development seeds to check the "
                 "pipeline and may not change the design.",
        "domain": stage_f_spec["domain"], "market": stage_f_spec["market"], "seeds": SEEDS[purpose],
        "scenarios": [stage_f.scenario(name, domain=grid()[name]) for name in names],
        "comparisons": [], "analysis": stage_f_spec["analysis"],
    }


def check_derivation(stage_f_archive: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    old = {row["name"]: row for row in json.loads((stage_f_archive / "experiment.json").read_text(encoding="utf-8"))["scenarios"]}
    new = {row["name"]: row for row in freeze_spec(SPEC_PATHS["estimation"])["scenarios"]}
    assert list(new) == list(grid())
    for name, row in new.items():
        reset = json.loads(json.dumps(row))
        reset["name"] = "B"
        for key in DYNAMICS:
            reset["domain"].pop(key, None)
        base = json.loads(json.dumps(old["B"]))
        base["domain"].pop("natural_decay")
        assert reset == base, f"{name} differs from stage F B beyond the natural dynamics"
        for key, value in grid()[name].items():
            assert row["domain"].get(key, 0.0) == value, (name, key)
    for name, source in (("u_d0p2", "B"), ("u_d0", "decay0_B")):
        assert {**new[name], "name": source} == old[source], f"{name} differs from stage F {source}"
    check = {row["name"]: row for row in freeze_spec(SPEC_PATHS["development"])["scenarios"]}
    assert all(check[name] == new[name] for name in CHECK_POINTS)
    print(f"derivation check: {len(new)} scenarios equal stage F B once the dynamics are reset; "
          "u_d0p2 = B and u_d0 = decay0_B exactly; development points equal their grid points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f", type=Path, default=STAGE_F / "confirmatory_v1",
                        help="stage F confirmatory archive (read-only)")
    args = parser.parse_args()
    for purpose, path in SPEC_PATHS.items():
        payload = spec(purpose)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path, len(payload["scenarios"]), "scenarios", len(payload["seeds"]), "seeds")
    check_derivation(args.stage_f)


if __name__ == "__main__":
    main()
