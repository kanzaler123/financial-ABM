"""Stage I v3 (research contract section 26.24): specs of the feasibility check F0, the estimation grid and, after
the grid estimate, the final stage and the recovery check, all under the direct rating channel at lambda*.

Every scenario is the frozen stage I v2 scenario B (H3's baseline at lambda* = 0.002575, financing_spread 0.05)
with one natural dynamic of the true level, as in H2 (contract 25.2); the archive is "sampled" (contract 25.3).

  spec                                              scenarios                                          seeds
  configs/disclosure_stage_i_feasibility_v3.json    F0: u_d0, u_d0p1, r_k0p05, q_k0p05_s0p9,          58901-58905
                                                    q_k0p1_s0p7, q_k0p505_s0p423 (= v2's B)
  configs/disclosure_stage_i_grid_v3.json           U 6 + R 9 + Q 81 = 96 points (26.24 (d))           58001-58010
  configs/disclosure_stage_i_grid_check_v3.json     u_d0p2, r_k0p5, q_k1_s0p4 (pipeline check only)     38101-38102
  configs/disclosure_stage_i_final_v3.json          the selected form at its estimate (after the grid)  58001-58030
  configs/disclosure_stage_i_recovery_v3.json       the same scenario                                   59001-59030

  python build_grid_spec_v3.py            # F0, grid and grid check, then the derivation check
  python build_grid_spec_v3.py final      # final stage and recovery from estimation_v3.json

Derivation checks: every scenario equals v2's frozen B once its dynamics are set back to v2's; every grid point that
H2's grid also has equals H2's frozen point once financing_rating_slope is removed; q_k0p505_s0p423 equals v2's B
apart from the name; the grid-check points equal their grid points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = ROOT / "configs"
V2 = CONFIGS / "disclosure_stage_i_v2.json"
H2_GRID = CONFIGS / "disclosure_persistence_grid_v1.json"
SLOPE = 0.002575                                      # lambda*, contract 26.21
DECAYS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
KAPPAS_T = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FEASIBILITY = {"u_d0": {"natural_decay": 0.0},
               "u_d0p1": {"natural_decay": 0.1},
               "r_k0p05": {"natural_decay": 0.0, "natural_reversion": 0.05},
               "q_k0p05_s0p9": {"natural_decay": 0.0, "natural_reversion": 0.05, "reversion_permanent_share": 0.9},
               "q_k0p1_s0p7": {"natural_decay": 0.0, "natural_reversion": 0.1, "reversion_permanent_share": 0.7},
               "q_k0p505_s0p423": {"natural_decay": 0.0, "natural_reversion": 0.505,
                                   "reversion_permanent_share": 0.423}}
CHECK_POINTS = ("u_d0p2", "r_k0p5", "q_k1_s0p4")     # as H2's development run
SEEDS = {"feasibility": list(range(58901, 58906)), "grid": list(range(58001, 58011)),
         "grid_check": [38101, 38102], "final": list(range(58001, 58031)), "recovery": list(range(59001, 59031))}
DYNAMICS = ("natural_decay", "natural_reversion", "reversion_permanent_share")
KEYS = {"delta": "natural_decay", "kappa": "natural_reversion", "kappa_T": "natural_reversion",
        "s_P": "reversion_permanent_share"}


def label(value: float) -> str:
    return f"{value:g}".replace(".", "p")   # as reports/disclosure_stage_f_20261014/build_specs.py


def grid() -> dict[str, dict]:
    """Scenario name -> domain dynamics, in the order U, R, Q (H2's names)."""
    points = {f"u_d{label(d)}": {"natural_decay": d} for d in DECAYS}
    points.update({f"r_k{label(k)}": {"natural_decay": 0.0, "natural_reversion": k} for k in KAPPAS})
    points.update({f"q_k{label(k)}_s{label(s)}": {
        "natural_decay": 0.0, "natural_reversion": k, "reversion_permanent_share": s} for k in KAPPAS_T for s in SHARES})
    return points


def row(name: str, dynamics: dict) -> dict:
    return {"name": name, "domain": dict(dynamics), "market": {"financing_rating_slope": SLOPE}}


def spec(stage: str, rows: list[dict], note: str) -> dict:
    base = json.loads(V2.read_text(encoding="utf-8"))
    assert base["financing_rating_slopes"]["lambda_star"] == SLOPE
    purpose = "development" if stage == "grid_check" else "estimation"
    return {"name": f"disclosure_stage_i_{stage}_v3", "purpose": purpose, "archive": "sampled",
            "_note": f"Stage I v3 (research contract section 26.24): {note} The stage I v2 baseline B at lambda* = "
                     f"{SLOPE} (financing_spread 0.05), sampled archive (public signal every 63 trading days from "
                     "day 1260). No comparison, no hypothesis test.",
            "domain": base["domain"], "market": base["market"], "seeds": SEEDS[stage],
            "scenarios": rows, "comparisons": [], "analysis": base["analysis"]}


def write(stage: str, payload: dict) -> Path:
    path = CONFIGS / f"disclosure_stage_i_{stage}_v3.json"
    path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(path.relative_to(ROOT).as_posix(), len(payload["scenarios"]), "scenarios", len(payload["seeds"]), "seeds")
    return path


def check(paths: dict[str, Path]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    v2_b = next(r for r in freeze_spec(V2)["scenarios"] if r["name"] == "B")
    h2 = {r["name"]: r for r in freeze_spec(H2_GRID)["scenarios"]}

    def base_equal(frozen: dict) -> None:
        reset = json.loads(json.dumps(frozen))
        reset["name"] = "B"
        for key in DYNAMICS:
            reset["domain"].pop(key, None)
        target = json.loads(json.dumps(v2_b))
        for key in DYNAMICS:
            target["domain"].pop(key, None)
        assert reset == target, f"{frozen['name']} differs from v2's B beyond the natural dynamics"
        assert frozen["config"]["financing_rating_slope"] == SLOPE and frozen["config"]["financing_spread"] == 0.05

    shared = 0
    for stage, path in paths.items():
        frozen = {r["name"]: r for r in freeze_spec(path)["scenarios"]}
        wanted = {"feasibility": FEASIBILITY, "grid": grid(), "grid_check": {n: grid()[n] for n in CHECK_POINTS},
                  "final": None, "recovery": None}[stage]
        if wanted is not None:
            assert list(frozen) == list(wanted), f"{stage}: scenario list"
        for name, scenario in frozen.items():
            base_equal(scenario)
            if wanted is not None:
                for key in DYNAMICS:
                    assert scenario["domain"].get(key, 0.0) == wanted[name].get(key, 0.0), (stage, name, key)
            if name in h2 and stage == "grid":
                plain = json.loads(json.dumps(scenario))
                plain["config"].pop("financing_rating_slope")
                assert plain == h2[name], f"{name} differs from H2's grid point beyond the rating slope"
                shared += 1
        if stage == "feasibility":
            assert {**frozen["q_k0p505_s0p423"], "name": "B"} == v2_b, "q_k0p505_s0p423 differs from v2's B"
        if stage == "grid_check":
            grid_frozen = {r["name"]: r for r in freeze_spec(CONFIGS / "disclosure_stage_i_grid_v3.json")["scenarios"]}
            assert all(frozen[n] == grid_frozen[n] for n in CHECK_POINTS)
    print(f"derivation check: every scenario equals v2's B beyond its natural dynamics at lambda* = {SLOPE}; "
          f"{shared} grid points equal H2's once the slope is removed (H2 has {len(h2)}); "
          "q_k0p505_s0p423 equals v2's B; check points equal their grid points")


def final_specs() -> dict[str, Path]:
    estimation = json.loads((HERE / "estimation_v3.json").read_text(encoding="utf-8"))
    selected = estimation["selected"]
    estimate = estimation["forms"][selected]["estimate"]
    dynamics = {} if selected == "U" else {"natural_decay": 0.0}
    dynamics.update({KEYS[name]: round(value, 3) for name, value in estimate.items() if name != "sigma_m"})
    name = f"{selected.lower()}_hat"
    paths = {}
    for stage in ("final", "recovery"):
        paths[stage] = write(stage, spec(stage, [row(name, dynamics)],
                                         f"the {stage} run at the grid-stage estimate of form {selected} "
                                         f"({json.dumps(dynamics)}), contract 25.3 and 25.7."))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", nargs="?", default="grid", choices=("grid", "final"))
    args = parser.parse_args()
    if args.stage == "final":
        check(final_specs())
        return
    paths = {
        "feasibility": write("feasibility", spec("feasibility", [row(n, d) for n, d in FEASIBILITY.items()],
                                                 "feasibility check F0 (26.24 (c)).")),
        "grid": write("grid", spec("grid", [row(n, d) for n, d in grid().items()],
                                   "the estimation grid (26.24 (d)), common random numbers.")),
    }
    paths["grid_check"] = write("grid_check", spec("grid_check", [row(n, grid()[n]) for n in CHECK_POINTS],
                                                   "development run of the grid, one point per form; pipeline "
                                                   "check only, may not change the design."))
    check(paths)


if __name__ == "__main__":
    main()
