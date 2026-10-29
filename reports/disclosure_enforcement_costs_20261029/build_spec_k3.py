"""Stage K3a specs: the Jacobian points at theta_0 (contract 28.5) and their development check (contract 28.7).

    python -I -X utf8 build_spec_k3.py

Every scenario is the H3 baseline B (configs/disclosure_stage_h3_v1.json, lambda = 0) with the fine per detected
case set to F_K (contract 28.14) and the parameters of estimate_k3.PARAMS at theta_0 or at theta_0 +- 10% of one
parameter's range (in unit coordinates; c on the log scale). Sampled archive (contract 28.4).

Writes configs/disclosure_stage_k3a_v1.json (purpose estimation, 13 points x seeds 68001-68010) and
configs/disclosure_stage_k3a_check_v1.json (development seeds 38201-38202; theta_0 and the two kappa_T points), then
checks the derivation: theta_0 equals H3 B in every field except baseline_fixed_fine, and each difference point
differs from theta_0 only in its own parameter's fields.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H3_SPEC = ROOT / "configs/disclosure_stage_h3_v1.json"
SEEDS = {"k3a": list(range(68001, 68011)), "development": [38201, 38202]}       # contract 28.8
SPEC_PATHS = {"k3a": ROOT / "configs/disclosure_stage_k3a_v1.json",
              "development": ROOT / "configs/disclosure_stage_k3a_check_v1.json"}
CHECK_POINTS = ("theta0", "fd_kappa_T_up", "fd_kappa_T_down")
FIELDS = {"kappa_T": ["natural_reversion"], "s_P": ["reversion_permanent_share"], "c": ["real_cost_scale"],
          "eta": ["cost_level_elasticity"], "psi": ["size_cost_multipliers"], "w": ["wash_cost_ratio"]}


def _estimator():
    spec = importlib.util.spec_from_file_location("estimate_k3_params", HERE / "estimate_k3.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def fine_k() -> float:
    est = json.loads((HERE / "k1_estimates.json").read_text(encoding="utf-8"))
    return float(est["F_K"]["model_fine_used_in_K3_model_units"])


def fd_name(param: str, sign: int) -> str:
    return f"fd_{param}_{'up' if sign > 0 else 'down'}"


def points() -> dict[str, dict]:
    ek = _estimator()
    theta0 = {p.name: p.theta0 for p in ek.PARAMS}
    out = {"theta0": ek.domain_changes(theta0, q=0.01, fine=fine_k())}
    for p in ek.PARAMS:
        u0 = p.to_unit(p.theta0)
        for sign in (+1, -1):
            u = u0 + sign * ek.FD_STEP
            assert 0.0 <= u <= 1.0, (p.name, u)
            out[fd_name(p.name, sign)] = ek.domain_changes({**theta0, p.name: p.from_unit(u)}, q=0.01, fine=fine_k())
    return out


def spec(purpose: str) -> dict:
    h3 = json.loads(H3_SPEC.read_text(encoding="utf-8"))
    names = list(points()) if purpose == "k3a" else list(CHECK_POINTS)
    return {
        "name": SPEC_PATHS[purpose].stem, "purpose": "estimation" if purpose == "k3a" else "development",
        "archive": "sampled",
        "_note": "Stage K3a (research contract 28.5): the H3 baseline B with the K1 fine per detected case F_K, at "
                 "theta_0 and at central-difference points of +-10% of each cost or dynamics parameter's range. "
                 "Sampled archive. The development spec checks the pipeline only and may not change the design.",
        "domain": h3["domain"], "market": h3["market"], "seeds": SEEDS[purpose],
        "scenarios": [{"name": name, "domain": points()[name]} for name in names],
        "comparisons": [], "analysis": h3["analysis"],
    }


def check_derivation() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from abm.disclosure.experiments import freeze_spec

    h3 = {row["name"]: row for row in freeze_spec(H3_SPEC)["scenarios"]}
    new = {row["name"]: row for row in freeze_spec(SPEC_PATHS["k3a"])["scenarios"]}
    base = json.loads(json.dumps(h3["B"]))
    t0 = json.loads(json.dumps(new["theta0"]))
    assert t0["domain"]["baseline_fixed_fine"] == fine_k()
    assert t0["policy"]["baseline"]["fixed_fine"] == fine_k()
    t0["domain"]["baseline_fixed_fine"] = base["domain"]["baseline_fixed_fine"]
    t0["policy"]["baseline"]["fixed_fine"] = base["policy"]["baseline"]["fixed_fine"]
    t0["name"] = "B"
    assert t0 == base, "theta0 differs from H3 B beyond the fine"
    for name, row in new.items():
        if name == "theta0":
            continue
        param = name[3:].rsplit("_", 1)[0]
        diff = {k for k in row["domain"] if row["domain"][k] != new["theta0"]["domain"][k]}
        assert diff == set(FIELDS[param]), (name, diff)
        assert {k: v for k, v in row.items() if k not in ("name", "domain")} == \
            {k: v for k, v in new["theta0"].items() if k not in ("name", "domain")}, name
    check = {row["name"]: row for row in freeze_spec(SPEC_PATHS["development"])["scenarios"]}
    assert all(check[name] == new[name] for name in CHECK_POINTS)
    print(f"derivation check: theta0 = H3 B except the fine ({fine_k():.6e}); {len(new) - 1} difference points "
          "each change only their own parameter; development points equal their K3a points")


def main() -> None:
    for purpose, path in SPEC_PATHS.items():
        payload = spec(purpose)
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path.name, len(payload["scenarios"]), "scenarios", len(payload["seeds"]), "seeds")
    for name, changes in points().items():
        print(name, {k: v for k, v in changes.items() if k not in ("natural_decay", "baseline_audit_probability",
                                                                    "baseline_fixed_fine")})
    check_derivation()


if __name__ == "__main__":
    main()
