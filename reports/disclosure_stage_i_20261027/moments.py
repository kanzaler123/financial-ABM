"""Stage I rating-persistence check (research contract section 26.22 (b)). Read-only on the archives.

  python moments.py --validate-h2                               # this code path reproduces H2's final stage
  python moments.py moments_check_v1 check_v2 --smoke           # development: replay check and counts only
  python moments.py stage_i_moments_v1 stage_i_v2               # after the confirmatory run (contract 26.22 (a))

1. Replay check: every trial of the replay archive (archive "sampled") is complete and has the run_id and result
   fingerprint of the same scenario and seed in the reference archive (the compact v2 archive it replays). One
   mismatch stops the script before any moment is computed.
2. Moments, as H2 computes them, by calling reports/disclosure_persistence_estimation_20261026/estimate.py: public
   signal sampled every 63 trading days from day 1260 (36 dates), normal scores per date, pooled lag correlations
   at lags 1, 2, 4, 8, 12, sigma_m = 0 (H2's estimate of form Q). For each rating slope, estimate.final_stage at
   H2's Q estimate on H2's grid: seed-mean moments, gap to the data, 95% half-width, fit statistic T and its p.
3. Paired difference against lambda0 (l0_B) seed by seed, with a paired 95% t interval; l0_B against H2's final
   stage (seeds 48001-48030, the same model on other seeds) with a Welch 95% interval as a replication check.
4. "Material deviation" (contract 26.22 (b)), judged at lambda* and reported by the same rule at lambda_lo,
   lambda_hi and lambda_FE: some lag k whose paired interval excludes 0 and whose mean |difference| exceeds the
   data's own 95% half-width 1.96 SE_data,k (from the bootstrap covariance of data_moments.json).
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
prefill_uname()  # importing scipy.stats can block on a hung WMI service (see _no_wmi.py)
from scipy.stats import t as student_t  # noqa: E402

from abm.disclosure.experiments import paired_statistics  # noqa: E402

_spec = importlib.util.spec_from_file_location("h2_estimate", H2_DIR / "estimate.py")
estimate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(estimate)

LAGS = estimate.LAGS
SLOPES = {"B": "lambda_star", "l0_B": "lambda0", "llo_B": "lambda_lo", "lhi_B": "lambda_hi", "lfe_B": "lambda_fe"}
REFERENCE, STAR = "l0_B", "B"
H2_GRID, H2_FINAL = H2_DIR / "grid_v1", H2_DIR / "final_v1"


# -- H2's ingredients ---------------------------------------------------------------------------------------------

def h2() -> dict:
    estimation = json.loads((H2_DIR / "estimation_v1.json").read_text(encoding="utf-8"))
    data = json.loads((H2_DIR / "data_moments.json").read_text(encoding="utf-8"))["targets"][estimation["target"]]
    assert data["lags"] == list(LAGS) and estimation["selected"] == "Q"
    grid_seeds = estimation["grid_seeds"]
    _, axes, name = estimate.FORMS["Q"]
    shape = tuple(len(axis) for axis in axes)
    values = np.empty(shape + (len(estimate.SIGMAS), len(grid_seeds), len(LAGS)))
    for index in np.ndindex(*shape):                  # estimate.grid_moments, form Q only
        point = name(*(axis[i] for axis, i in zip(axes, index)))
        values[index] = estimate.point_moments({s: estimate.finished_signal(H2_GRID, point, s) for s in grid_seeds})
    data_cov = np.array(data["bootstrap_cov"])
    return {"estimation": estimation, "form": estimate.Form("Q", values),
            "theta": np.array(list(estimation["forms"]["Q"]["estimate"].values())),
            "target": np.array(data["moments"]), "data_cov": data_cov,
            "threshold": 1.96 * np.sqrt(np.diag(data_cov))}


def seed_draws(archive: Path, scenario: str, seeds: list[int], sigma: float) -> np.ndarray:
    """Per-seed moments exactly as estimate.final_stage computes them, shape (seeds, lags)."""
    out = []
    for seed in seeds:
        signal = estimate.finished_signal(archive, scenario, seed)
        out.append(estimate.seed_moments(signal, sigma, estimate.noise(seed, signal.shape)))
    return np.array(out)


def final_row(ingredients: dict, archive: Path, scenario: str, seeds: list[int]) -> dict:
    return estimate.final_stage(ingredients["form"], ingredients["theta"], archive, scenario, seeds,
                                ingredients["target"], ingredients["data_cov"])


def validate_h2(ingredients: dict) -> dict:
    """This code path on H2's final-stage archive must give H2's recorded final stage."""
    recorded = ingredients["estimation"]["final"]
    frozen = json.loads((H2_FINAL / "experiment.json").read_text(encoding="utf-8"))
    assert frozen["experiment_id"] == recorded["experiment_id"]
    row = final_row(ingredients, H2_FINAL, recorded["scenario"], frozen["spec"]["seeds"])
    for key in ("moments_model", "gap", "band95", "T", "T_p_value"):
        np.testing.assert_allclose(row[key], recorded[key], rtol=1e-12, atol=0, err_msg=key)
    assert row["within_band"] == recorded["within_band"]
    return {"experiment_id": recorded["experiment_id"], "T": row["T"], "T_p_value": row["T_p_value"],
            "moments_model": row["moments_model"], "matches_estimation_v1": True}


# -- replay check ---------------------------------------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict]:
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["experiment_id"] != frozen["experiment_id"]:
        raise ValueError(f"{run}: summary and experiment manifest disagree")
    return frozen, summary


def run_id(run: Path, record: dict) -> str:
    return json.loads((run / record["path"] / "manifest.json").read_text(encoding="utf-8"))["run_id"]


def replay_check(replay: Path, reference: Path) -> dict:
    frozen, summary = load(replay)
    _, ref_summary = load(reference)
    ref = {(r["scenario"], r["seed"]): r for r in ref_summary["records"]}
    seeds, names = frozen["spec"]["seeds"], [row["name"] for row in frozen["scenarios"]]
    records = {(r["scenario"], r["seed"]): r for r in summary["records"]}
    problems = []
    for name in names:
        for seed in seeds:
            mine, theirs = records.get((name, seed)), ref.get((name, seed))
            if not mine or mine["status"] != "complete":
                problems.append(f"{name} {seed}: replay not complete")
            elif not theirs or theirs["status"] != "complete":
                problems.append(f"{name} {seed}: no completed run in the reference archive")
            elif mine["result_fingerprint"] != theirs["result_fingerprint"]:
                problems.append(f"{name} {seed}: result fingerprint differs")
            elif run_id(replay, mine) != run_id(reference, theirs):
                problems.append(f"{name} {seed}: run_id differs")
    return {"replay": replay.relative_to(ROOT).as_posix(), "replay_experiment_id": frozen["experiment_id"],
            "reference": reference.relative_to(ROOT).as_posix(), "reference_experiment_id": ref_summary["experiment_id"],
            "purpose": frozen["spec"]["purpose"], "scenarios": names, "seeds": seeds,
            "trials": len(names) * len(seeds), "identical": not problems, "problems": problems}


# -- moments and judgments ---------------------------------------------------------------------------------------------

def interval(values: np.ndarray) -> dict:
    out = paired_statistics([float(v) for v in values], 0.95)
    return {key: out[key] for key in ("mean", "standard_error", "ci_low", "ci_high", "n")}


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1)))
    mean, half = float(a.mean() - b.mean()), float(student_t.ppf(0.975, df)) * se
    return {"mean": mean, "standard_error": se, "ci_low": mean - half, "ci_high": mean + half, "df": df}


def deviation(paired: list[dict], threshold: np.ndarray) -> dict:
    lags = [lag for lag, row, limit in zip(LAGS, paired, threshold)
            if (row["ci_low"] > 0 or row["ci_high"] < 0) and abs(row["mean"]) > limit]
    return {"material_deviation": bool(lags), "lags": lags}


def moments(ingredients: dict, replay: Path, seeds: list[int]) -> dict:
    sigma = float(ingredients["theta"][-1])
    draws = {name: seed_draws(replay, name, seeds, sigma) for name in SLOPES}
    final = json.loads((H2_FINAL / "experiment.json").read_text(encoding="utf-8"))
    h2_draws = seed_draws(H2_FINAL, ingredients["estimation"]["final"]["scenario"], final["spec"]["seeds"], sigma)
    threshold = ingredients["threshold"]
    out = {"lags": list(LAGS), "sigma_m": sigma, "h2_estimate": ingredients["estimation"]["forms"]["Q"]["estimate"],
           "data": ingredients["target"].tolist(), "data_half_width_95": threshold.tolist(), "slopes": {}}
    for name, key in SLOPES.items():
        row = final_row(ingredients, replay, name, seeds)
        np.testing.assert_allclose(row["moments_model"], draws[name].mean(axis=0), rtol=1e-14, atol=0)
        entry = {"scenario": name, "slope": key, "moments": row["moments_model"], "gap_to_data": row["gap"],
                 "band95": row["band95"], "within_band": row["within_band"], "T": row["T"], "T_p_value": row["T_p_value"]}
        if name != REFERENCE:
            paired = [interval(draws[name][:, k] - draws[REFERENCE][:, k]) for k in range(len(LAGS))]
            entry["paired_minus_lambda0"] = paired
            entry["judgment"] = deviation(paired, threshold)
        out["slopes"][key] = entry
    out["replication_l0_B_minus_h2_final"] = [welch(draws[REFERENCE][:, k], h2_draws[:, k]) for k in range(len(LAGS))]
    out["h2_final_moments"] = h2_draws.mean(axis=0).tolist()
    star = out["slopes"][SLOPES[STAR]]["judgment"]
    out["material_deviation_at_lambda_star"] = star["material_deviation"]
    return out


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def table(result: dict) -> list[str]:
    m = result["moments"]
    lags = " | ".join(str(lag) for lag in m["lags"])
    out = ["# 方案 A 下的评级持续性复核（契约第 26.22 条 (b)）", "",
           f"重放 `{result['replay_check']['replay']}`（`{result['replay_check']['replay_experiment_id'][:12]}…`）对照 "
           f"`{result['replay_check']['reference']}`：{result['replay_check']['trials']} 个运行，run_id 与结果指纹全部相同。", "",
           f"H2 的 Q 估计（κ_T {m['h2_estimate']['kappa_T']:.3f}，s_P {m['h2_estimate']['s_P']:.3f}，σ_m {m['sigma_m']:g}）；"
           "矩与 T 由 H2 的 `estimate.final_stage` 计算。", "",
           f"| | 滞后 {lags} | T | p |", "|---|" + "---|" * (len(m["lags"]) + 2),
           "| 数据 | " + " | ".join(f"{v:.4f}" for v in m["data"]) + " | | |",
           "| 数据 95% 半宽（门槛） | " + " | ".join(f"{v:.4f}" for v in m["data_half_width_95"]) + " | | |",
           "| H2 最终阶段（种子 48001–48030） | " + " | ".join(f"{v:.4f}" for v in m["h2_final_moments"]) + " | | |"]
    for key, row in m["slopes"].items():
        out.append(f"| {key}（{row['scenario']}） | " + " | ".join(f"{v:.4f}" for v in row["moments"])
                   + f" | {row['T']:.1f} | {row['T_p_value']:.2g} |")
    out += ["", f"## 与 λ₀ 的逐种子配对差（95% t 区间）", "", f"| | 滞后 {lags} | 实质偏离 |",
            "|---|" + "---|" * (len(m["lags"]) + 1)]
    for key, row in m["slopes"].items():
        if "paired_minus_lambda0" not in row:
            continue
        cells = [f"{fmt(p['mean'])} [{fmt(p['ci_low'])}, {fmt(p['ci_high'])}]" for p in row["paired_minus_lambda0"]]
        verdict = "是（滞后 " + "、".join(map(str, row["judgment"]["lags"])) + "）" if row["judgment"]["material_deviation"] else "否"
        out.append(f"| {key} | " + " | ".join(cells) + f" | {verdict} |")
    rep = m["replication_l0_B_minus_h2_final"]
    out += ["", "## 复现核对：l0_B 减 H2 最终阶段（不同种子，Welch 95% 区间）", "",
            "| " + " | ".join(f"滞后 {lag}" for lag in m["lags"]) + " |", "|" + "---|" * len(m["lags"]),
            "| " + " | ".join(f"{fmt(r['mean'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" for r in rep) + " |", "",
            f"λ* 下的判定：{'实质偏离' if m['material_deviation_at_lambda_star'] else '无实质偏离'}（契约第 26.22 条 (b)）。"]
    return out


def write(path: Path, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", nargs="?", help="replay archive next to this script, e.g. stage_i_moments_v1")
    parser.add_argument("reference", nargs="?", help="the compact archive it replays, e.g. stage_i_v2")
    parser.add_argument("--smoke", action="store_true", help="development archive: replay check and counts only")
    parser.add_argument("--validate-h2", action="store_true", help="reproduce H2's final stage with this code path")
    args = parser.parse_args()
    if args.validate_h2:
        out = validate_h2(h2())
        write(HERE / "moments_validate_h2.json", out)
        print(json.dumps(out, indent=2))
        return 0
    replay, reference = HERE / args.replay, HERE / args.reference
    check = replay_check(replay, reference)
    if check["purpose"] == "development" and not args.smoke:
        raise SystemExit("development archives are checked with --smoke only")
    stem = f"{replay.name}_smoke" if args.smoke else "moments_v1"
    if not check["identical"]:
        write(HERE / f"{stem}.json", {"replay_check": check})
        print(json.dumps(check, indent=2, ensure_ascii=False))
        return 1                                       # contract 26.22 (b): stop, no moments
    if args.smoke:
        shapes = {name: [list(estimate.finished_signal(replay, name, seed).shape) for seed in check["seeds"]]
                  for name in check["scenarios"]}
        out = {"replay_check": check, "sampled_signal_shapes": shapes}
        write(HERE / f"{stem}.json", out)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    result = {"replay_check": check, "moments": moments(h2(), replay, check["seeds"])}
    write(HERE / "moments_v1.json", result)
    write(HERE / "moments_v1.md", "\n".join(table(result)) + "\n")
    print("\n".join(table(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
