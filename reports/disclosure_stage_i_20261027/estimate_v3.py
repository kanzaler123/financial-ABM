"""Stage I v3 (research contract section 26.24): feasibility check F0 and the H2 estimator on the v3 grid.

  python estimate_v3.py --validate-h2
      this wrapper, given H2's grid, reproduces H2's estimation_v1.json from H2's archives (grid, final, recovery)
  python estimate_v3.py feasibility feasibility_v3
      F0 (26.24 (c)): seed-mean moments (sigma_m = 0) of the six points; "feasible" when, at every lag, the largest
      of the six is at least the data moment; writes feasibility_v3.json and .md
  python estimate_v3.py grid grid_v3 [--final final_v3] [--recovery recovery_v3]
      the estimator of contract 25.4-25.7 on the v3 grid (26.24 (d)); with --final, the condition of 26.24 (e):
      at every lag |g_k| <= max(b_k, |g_k^H2|), g the data minus the final-stage model moments, b the final
      stage's 95% band, g^H2 the gaps of H2's final stage; writes estimation_v3.json and .md

Every function of the estimator is H2's (reports/disclosure_persistence_estimation_20261026/estimate.py, loaded
unchanged); only its FORMS table, the grid axes and scenario names, is replaced by the v3 grid of
build_grid_spec_v3.py. Archive arguments are directory names next to this script, or paths. Only finished
archives are read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from _no_wmi import prefill_uname

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H2_DIR = ROOT / "reports" / "disclosure_persistence_estimation_20261026"
sys.path.insert(0, str(ROOT / "src"))
prefill_uname()  # importing scipy can block on a hung WMI service (see _no_wmi.py)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h2 = _load("h2_estimate", H2_DIR / "estimate.py")
grid_spec = _load("build_grid_spec_v3", HERE / "build_grid_spec_v3.py")
H2_FORMS = dict(h2.FORMS)
V3_FORMS = {
    "U": (("delta",), (grid_spec.DECAYS,), lambda d: f"u_d{grid_spec.label(d)}"),
    "R": (("kappa",), (grid_spec.KAPPAS,), lambda k: f"r_k{grid_spec.label(k)}"),
    "Q": (("kappa_T", "s_P"), (grid_spec.KAPPAS_T, grid_spec.SHARES),
          lambda k, s: f"q_k{grid_spec.label(k)}_s{grid_spec.label(s)}"),
}


def archive_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() or path.exists() else HERE / name


def data_target() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((H2_DIR / "data_moments.json").read_text(encoding="utf-8"))["targets"]["R1"]
    assert data["lags"] == list(h2.LAGS)
    return np.array(data["moments"]), np.array(data["bootstrap_cov"])


def run_estimation(grid_archive: Path, final: Path | None, recovery: Path | None) -> dict:
    """H2's main() without its output: the same calls in the same order (estimate.py lines 267-300)."""
    frozen = json.loads((grid_archive / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    target, data_cov = data_target()
    grid = h2.grid_moments(grid_archive, seeds)
    out = {"grid_experiment_id": frozen["experiment_id"], "target": "R1", "grid_seeds": seeds,
           **h2.estimate_all(grid, target, data_cov)}
    if final:
        spec = json.loads((final / "experiment.json").read_text(encoding="utf-8"))
        scenario = spec["scenarios"][0]["name"]
        form = h2.Form(out["selected"], grid[out["selected"]])
        theta = np.array(list(out["forms"][out["selected"]]["estimate"].values()))
        out["final"] = {"experiment_id": spec["experiment_id"], "scenario": scenario,
                        **h2.final_stage(form, theta, final, scenario, spec["spec"]["seeds"], target, data_cov)}
    if recovery:
        spec = json.loads((recovery / "experiment.json").read_text(encoding="utf-8"))
        seeds_r = spec["spec"]["seeds"]
        groups = [seeds_r[i:i + 10] for i in range(0, len(seeds_r), 10)]
        out["recovery"] = {"experiment_id": spec["experiment_id"],
                           **h2.recovery(grid, recovery, spec["scenarios"][0]["name"], groups, out)}
    return out


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{prefix}[{i}]")
    else:
        yield prefix, node


def validate_h2() -> None:
    h2.FORMS = H2_FORMS
    out = json.loads(json.dumps(run_estimation(H2_DIR / "grid_v1", H2_DIR / "final_v1", H2_DIR / "recovery_v1")))
    old = json.loads((H2_DIR / "estimation_v1.json").read_text(encoding="utf-8"))
    a, b = dict(leaves(out)), dict(leaves(old))
    assert a.keys() == b.keys(), sorted(a.keys() ^ b.keys())[:10]
    worst, different = 0.0, []
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, float) and isinstance(y, (int, float)) and not isinstance(y, bool):
            if x != y:
                rel = abs(x - y) / max(abs(y), 1e-300)
                worst = max(worst, rel)
                if rel > 1e-12:
                    different.append(key)
        elif x != y:
            different.append(key)
    result = {"reference": "reports/disclosure_persistence_estimation_20261026/estimation_v1.json",
              "fields": len(a), "fields_identical": sum(a[k] == b[k] for k in a),
              "max_relative_difference": worst, "fields_beyond_1e-12": different, "reproduced": not different}
    (HERE / "estimate_v3_validate_h2.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))
    if different:
        raise SystemExit("the wrapper does not reproduce H2")


def feasibility(archive: Path) -> None:
    frozen = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    target, data_cov = data_target()
    points = {}
    for row in frozen["scenarios"]:
        draws = h2.point_moments({seed: h2.finished_signal(archive, row["name"], seed) for seed in seeds})[0]  # sigma_m 0
        points[row["name"]] = {"dynamics": {k: v for k, v in row["domain"].items() if k in grid_spec.DYNAMICS},
                               "mean": draws.mean(axis=0).tolist(),
                               "seed_sd": draws.std(axis=0, ddof=1).tolist(), "seeds": len(seeds)}
    highest = np.max([p["mean"] for p in points.values()], axis=0)
    reached = (highest >= target).tolist()
    out = {"experiment_id": frozen["experiment_id"], "contract": "26.24 (c)", "lags": list(h2.LAGS),
           "data": target.tolist(), "data_band95": (1.96 * np.sqrt(np.diag(data_cov))).tolist(), "points": points,
           "highest": highest.tolist(), "highest_at_least_data": reached, "feasible": all(reached)}
    (HERE / "feasibility_v3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    lines = ["# F0：λ* 下的可行性检查（契约第 26.24 条 (c)）", "",
             f"`{frozen['experiment_id']}`，种子 {seeds[0]}–{seeds[-1]}，σ_m = 0，种子平均矩。", "",
             "| 点 | 滞后 1 | 2 | 4 | 8 | 12 |", "|---|---:|---:|---:|---:|---:|",
             "| 数据 | " + " | ".join(f"{v:.4f}" for v in target) + " |"]
    lines += [f"| {name} | " + " | ".join(f"{v:.4f}" for v in p["mean"]) + " |" for name, p in points.items()]
    lines += ["| 六点中的最大值 | " + " | ".join(f"{v:.4f}" for v in highest) + " |", "",
              f"判定：{'可行' if out['feasible'] else '不可行'}（每个滞后的最大值不低于数据：{reached}）。"]
    (HERE / "feasibility_v3.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))


def proceed_condition(final: dict) -> dict:
    """Contract 26.24 (e)."""
    old = json.loads((H2_DIR / "estimation_v1.json").read_text(encoding="utf-8"))["final"]["gap"]
    gap, band = np.array(final["gap"]), np.array(final["band95"])
    limit = np.maximum(band, np.abs(old))
    ok = (np.abs(gap) <= limit).tolist()
    return {"gap": gap.tolist(), "band95": band.tolist(), "h2_gap": old, "limit": limit.tolist(),
            "lag_ok": ok, "proceed": all(ok)}


def estimate(grid_archive: Path, final: Path | None, recovery: Path | None) -> None:
    h2.FORMS = V3_FORMS
    out = run_estimation(grid_archive, final, recovery)
    if "final" in out:
        out["proceed_26_24_e"] = proceed_condition(out["final"])
    (HERE / "estimation_v3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    lines = ["# v3 自然动态的估计（契约第 26.24 条 (d)、(e)）", "", f"网格 `{out['grid_experiment_id']}`。", "",
             "| 形式 | 点估计 | T | p | 拟合 |", "|---|---|---:|---:|---|"]
    for name, row in out["forms"].items():
        lines.append(f"| {name} | {json.dumps({k: round(v, 4) for k, v in row['estimate'].items()})} | "
                     f"{row['T']:.2f} | {row['T_p_value']:.3g} | {row['fits']} |")
    lines += ["", f"选中：{out['selected']}（{'拟合' if out['selected_fits'] else '没有形式拟合，取 T 最小者'}）。"]
    if "final" in out:
        f, c = out["final"], out["proceed_26_24_e"]
        lines += ["", "## 最终阶段", "", "| | 滞后 1 | 2 | 4 | 8 | 12 |", "|---|---:|---:|---:|---:|---:|",
                  "| 数据 | " + " | ".join(f"{v:.4f}" for v in f["moments_data"]) + " |",
                  "| 模型 | " + " | ".join(f"{v:.4f}" for v in f["moments_model"]) + " |",
                  "| 差 g | " + " | ".join(f"{v:+.4f}" for v in c["gap"]) + " |",
                  "| 95% 带宽 b | " + " | ".join(f"{v:.4f}" for v in c["band95"]) + " |",
                  "| H2 的差 | " + " | ".join(f"{v:+.4f}" for v in c["h2_gap"]) + " |",
                  "| 满足条件 | " + " | ".join(str(v) for v in c["lag_ok"]) + " |", "",
                  f"T = {f['T']:.2f}，p = {f['T_p_value']:.3g}。第 26.24 条 (e)：{'进入确认性重跑' if c['proceed'] else '停止，由用户决定'}。"]
    if "recovery" in out:
        lines += ["", f"恢复检验：{'恢复' if out['recovery']['recovered'] else '未恢复'}。"]
        lines += [f"- 种子 {g['seeds'][0]}–{g['seeds'][1]}：选中 {g['selected']}，"
                  f"{json.dumps({k: round(v, 4) for k, v in g['estimate'].items()})}，在主区间内 {g['inside_main_ci95']}"
                  for g in out["recovery"]["groups"]]
    (HERE / "estimation_v3.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("feasibility", "grid"))
    parser.add_argument("archive", nargs="?")
    parser.add_argument("--final")
    parser.add_argument("--recovery")
    parser.add_argument("--validate-h2", action="store_true")
    args = parser.parse_args()
    if args.validate_h2:
        validate_h2()
    elif args.command == "feasibility":
        feasibility(archive_path(args.archive))
    elif args.command == "grid":
        estimate(archive_path(args.archive), args.final and archive_path(args.final),
                 args.recovery and archive_path(args.recovery))
    else:
        parser.error("give --validate-h2 or a command")


if __name__ == "__main__":
    main()
