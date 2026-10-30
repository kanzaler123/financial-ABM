"""Stage K3 specs after K3a: the global design (28.6 (a)), the local design (28.6 (b)), the final stage and the
recovery runs (28.6 (c), (d)) and the (q, w) set (28.6 (e)).

    python -I -X utf8 build_spec_k3_stages.py stage1
    python -I -X utf8 build_spec_k3_stages.py stage2 [--recentre]
    python -I -X utf8 build_spec_k3_stages.py final
    python -I -X utf8 build_spec_k3_stages.py qw

The identified parameters come from identification_k3a.json (contract 28.5, recorded in the contract); the others
stay at theta_0. Every scenario is the K3a theta_0 scenario (H3 B with F_K) with the domain fields of
estimate_k3.domain_changes at its parameter point. Each command writes its spec and its development spec
(development seeds 38201-38202, two points), then checks that every scenario differs from theta_0 only in the fields
of the parameters it moves.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ek = _load("estimate_k3")
bs = _load("build_spec_k3")
SEEDS = {"stage1": list(range(63001, 63006)), "stage2": list(range(64001, 64011)),
         "stage2_recentre": list(range(64011, 64021)), "final": list(range(65001, 65031)),
         "recovery": list(range(66001, 66031)), "qw": list(range(67001, 67006)), "development": [38201, 38202]}
Q_GRID = (0.0025, 0.005, 0.01, 0.02, 0.04)
W_GRID = (0.05, 0.1, 0.2, 0.4, 0.8)


def identified() -> list[str]:
    return json.loads((HERE / "identification_k3a.json").read_text(encoding="utf-8"))["identified"]


def theta_of(u_kept: np.ndarray, names: list[str]) -> dict[str, float]:
    theta = {p.name: p.theta0 for p in ek.PARAMS}
    for name, u in zip(names, u_kept):
        p = next(p for p in ek.PARAMS if p.name == name)
        theta[name] = p.from_unit(float(u))
    return theta


def scenario(name: str, theta: dict, q: float = 0.01) -> dict:
    return {"name": name, "domain": ek.domain_changes(theta, q=q, fine=bs.fine_k())}


def write(label: str, scenarios: list[dict], seeds: list[int], note: str) -> None:
    h3 = json.loads(bs.H3_SPEC.read_text(encoding="utf-8"))
    for purpose, rows, sd, suffix in (("estimation", scenarios, seeds, ""),
                                      ("development", scenarios[:2], SEEDS["development"], "_check")):
        path = ROOT / f"configs/disclosure_stage_k3_{label}{suffix}_v1.json"
        payload = {"name": path.stem, "purpose": purpose, "archive": "sampled", "_note": note,
                   "domain": h3["domain"], "market": h3["market"], "seeds": sd, "scenarios": rows,
                   "comparisons": [], "analysis": h3["analysis"]}
        path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(path.name, len(rows), "scenarios", len(sd), "seeds")
    check(label, scenarios)


def check(label: str, scenarios: list[dict]) -> None:
    theta0 = bs.points()["theta0"]
    for row in scenarios:
        moved = {k for k in row["domain"] if row["domain"][k] != theta0[k]}
        allowed = {f for n in identified() for f in bs.FIELDS[n]} | ({"baseline_audit_probability", "wash_cost_ratio"}
                                                                        if label == "qw" else set())
        assert moved <= allowed, (row["name"], moved)
    print(f"derivation check: {len(scenarios)} scenarios move only the fields of {sorted(allowed)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["stage1", "stage2", "final", "qw"])
    ap.add_argument("--recentre", action="store_true")
    args = ap.parse_args()
    names = identified()
    k = len(names)
    if args.step == "stage1":
        U = ek.sobol(k, 32 * k, 28904)
        rows = [scenario(f"s1_{i:03d}", theta_of(u, names)) for i, u in enumerate(U)]
        write("stage1", rows, SEEDS["stage1"], f"Stage K3 global design (contract 28.6 (a)): {32 * k} scrambled Sobol "
              f"points (seed 28904) over {names}, others at theta_0, seeds 63001-63005, sampled archive.")
    elif args.step == "stage2":
        s1 = json.loads((HERE / ("stage2_k3.json" if args.recentre else "stage1_k3.json")).read_text(encoding="utf-8"))
        centre = np.array(s1["u_hat"] if args.recentre else s1["u_best"])
        lo, hi = ek.local_box(centre)
        U = ek.sobol(k, 16 * k, 28905, lo, hi)
        label = "stage2r" if args.recentre else "stage2"
        seeds = SEEDS["stage2_recentre" if args.recentre else "stage2"]
        rows = [scenario(f"s2_{i:03d}", theta_of(u, names)) for i, u in enumerate(U)]
        write(label, rows, seeds, f"Stage K3 local design (contract 28.6 (b)): {16 * k} scrambled Sobol points "
              f"(seed 28905) in the box centre +- 0.15 (unit coordinates) over {names}, sampled archive.")
    elif args.step == "final":
        est = json.loads((HERE / "stage2_k3.json").read_text(encoding="utf-8"))
        theta = {n: round(v, 3) for n, v in est["theta_hat"].items()}
        rows = [scenario("theta_hat", theta)]
        write("final", rows, SEEDS["final"], f"Stage K3 final stage (contract 28.6 (c)) at theta_hat {theta}.")
        write("recovery", [scenario("theta_hat", theta)], SEEDS["recovery"],
              f"Stage K3 recovery runs (contract 28.6 (d)) at theta_hat {theta}.")
    else:
        est = json.loads((HERE / "stage2_k3.json").read_text(encoding="utf-8"))
        theta = {n: round(v, 3) for n, v in est["theta_hat"].items()}
        rows = [scenario(f"qw_q{q}_w{w}".replace(".", "p"), {**theta, "w": w}, q=q) for q in Q_GRID for w in W_GRID]
        write("qw", rows, SEEDS["qw"], f"Stage K3 (q, w) set (contract 28.6 (e)) around theta_hat {theta}.")


if __name__ == "__main__":
    main()
