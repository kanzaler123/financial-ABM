"""Extract compact committed analysis JSON for MAS upgrade studies (§§29–32).

Reads the seven local confirmatory archives under ``runs/disclosure_mas_*`` plus
the committed audit summaries under ``reports/disclosure_mas_upgrade_20261125/``
and writes ``mas_upgrade_analysis.json`` so that ``paper/make_figures.py`` can
reproduce every §§29–32 figure strictly from committed files under ``reports/``.

Run from repository root:
    python reports/disclosure_mas_upgrade_20261125/extract_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
OUT_PATH = REPORT_DIR / "mas_upgrade_analysis.json"

STUDIES = [
    {
        "key": "sec29_stackelberg",
        "run_dir": "disclosure_mas_stackelberg_confirmatory_v1",
        "audit_file": "audit_summary.json",
        "baseline": "B",
        "compute_hit_rate": True,
        "window_start": 1200,
    },
    {
        "key": "sec30_empirical_belief",
        "run_dir": "disclosure_mas_empirical_belief_confirmatory_v1",
        "audit_file": "audit_summary.json",
        "baseline": "EB_B",
        "compute_hit_rate": False,
        "window_start": 1200,
    },
    {
        "key": "sec31_assurance",
        "run_dir": "disclosure_mas_assurance_confirmatory_v1",
        "audit_file": "audit_summary.json",
        "baseline": "B",
        "compute_hit_rate": False,
        "window_start": 1200,
    },
    {
        "key": "sec32A_q_dynamics",
        "run_dir": "disclosure_mas_q_dynamics_confirmatory_v1",
        "audit_file": "audit_sec32.json",
        "baseline": "Q_B",
        "compute_hit_rate": True,
        "window_start": 1200,
    },
    {
        "key": "sec32B_phase_pareto",
        "run_dir": "disclosure_mas_phase_pareto_confirmatory_v1",
        "audit_file": "audit_sec32.json",
        "baseline": "B",
        "compute_hit_rate": False,
        "window_start": 1200,
    },
    {
        "key": "sec32C1_granularity",
        "run_dir": "disclosure_mas_granularity_confirmatory_v1",
        "audit_file": "audit_sec32.json",
        "baseline": "NPV_B",
        "compute_hit_rate": False,
        "window_start": 1200,
    },
    {
        "key": "sec32C2_ai_washing",
        "run_dir": "disclosure_mas_ai_washing_confirmatory_v1",
        "audit_file": "audit_sec32.json",
        "baseline": "AI_B",
        "compute_hit_rate": True,
        "window_start": 600,
    },
]


def paired_t_summary(diffs: list[float]) -> dict[str, Any]:
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr))
    if n < 2:
        return {"n": n, "mean": mean, "se": 0.0, "ci_low": mean, "ci_high": mean, "p_value": 1.0}
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    if se == 0.0:
        p_val = 0.0 if mean != 0.0 else 1.0
        half = 0.0
    else:
        t_stat = mean / se
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        half = float(stats.t.ppf(0.975, df=n - 1) * se)
    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p_value": p_val,
        "pos": pos,
        "neg": neg,
    }


def extract_hit_rate(run_root: Path, rec: dict[str, Any], window_start: int) -> tuple[int, int]:
    npz_path = run_root / rec["path"] / "result.npz"
    z = np.load(npz_path)
    n_dec = int(rec["metrics"]["window"]["decision_observations"])
    aud_w = z["audit_selected"][-n_dec:]
    det_w = z["detections"][-n_dec:]
    det_sum = int(np.sum(det_w))
    expected_det = int(rec["metrics"]["actions"]["detections"])
    if det_sum != expected_det:
        raise ValueError(f"Detection mismatch for {rec['scenario']} seed {rec['seed']}: {det_sum} vs {expected_det}")
    return det_sum, int(np.sum(aud_w))


def get_nested(d: dict[str, Any], dotted: str) -> float | None:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    return float(cur)


def extract_study(cfg: dict[str, Any]) -> dict[str, Any]:
    run_root = ROOT / "runs" / cfg["run_dir"]
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    audit_all = json.loads((REPORT_DIR / cfg["audit_file"]).read_text(encoding="utf-8"))
    audit = audit_all[cfg["key"]]
    failed_seeds = set(int(s) for s in audit["failed_seeds"])

    records_by_sc: dict[str, dict[int, dict[str, Any]]] = {}
    scenario_order: list[str] = [s["name"] for s in summary["scenarios"]]
    for rec in summary["records"]:
        records_by_sc.setdefault(rec["scenario"], {})[int(rec["seed"])] = rec

    all_seeds = sorted(next(iter(records_by_sc.values())).keys())
    clean_seeds = [s for s in all_seeds if s not in failed_seeds]

    # Scenario descriptive means
    scenarios_out: dict[str, Any] = {}
    base_name = cfg["baseline"]
    base_recs = records_by_sc[base_name]

    for sc_name in scenario_order:
        sc_recs = records_by_sc[sc_name]

        def mean_metric(fn, seeds: list[int]) -> float | None:
            vals = [fn(sc_recs[s]) for s in seeds]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        desc: dict[str, Any] = {
            "true_change_eq": mean_metric(lambda r: r["metrics"]["true_change"]["equal_weight_sd"], all_seeds),
            "true_change_cap": mean_metric(lambda r: r["metrics"]["true_change"]["fixed_start_cap_weight_sd"], all_seeds),
            "inflation_mean_sd": mean_metric(lambda r: r["metrics"]["inflation"]["mean_sd"], all_seeds),
            "wash_count": mean_metric(lambda r: r["metrics"]["actions"]["wash"], all_seeds),
            "decisions": mean_metric(lambda r: r["metrics"]["actions"]["decisions"], all_seeds),
            "wash_share": mean_metric(
                lambda r: r["metrics"]["actions"]["wash"] / r["metrics"]["actions"]["decisions"], all_seeds
            ),
            "wash_share_large": mean_metric(
                lambda r: r["metrics"]["actions"]["wash_share_by_size"]["large"], all_seeds
            ),
            "wash_share_mid": mean_metric(
                lambda r: r["metrics"]["actions"]["wash_share_by_size"]["mid"], all_seeds
            ),
            "wash_share_small": mean_metric(
                lambda r: r["metrics"]["actions"]["wash_share_by_size"]["small"], all_seeds
            ),
            "detections": mean_metric(lambda r: r["metrics"]["actions"]["detections"], all_seeds),
            "resource_cost_bp": mean_metric(
                lambda r: 10000.0 * r["metrics"]["costs"]["resource_cost_fraction_initial_cap"], all_seeds
            ),
            "assurance_effort_cost": mean_metric(
                lambda r: r["metrics"]["costs"].get("assurance_effort_cost"), all_seeds
            ),
            "unweighted_mean_quality": mean_metric(
                lambda r: (r["metrics"].get("assurance") or {}).get("unweighted_mean_quality"), all_seeds
            ),
            "client_weighted_mean_quality": mean_metric(
                lambda r: (r["metrics"].get("assurance") or {}).get("client_weighted_mean_quality"), all_seeds
            ),
            "q_gap": mean_metric(
                lambda r: (r["metrics"].get("assurance") or {}).get("client_minus_unweighted_quality_gap"), all_seeds
            ),
            "excess_laxest_wash_concentration": mean_metric(
                lambda r: (r["metrics"].get("assurance") or {}).get("excess_laxest_wash_concentration"), all_seeds
            ),
            "raw_laxest_wash_share": mean_metric(
                lambda r: (r["metrics"].get("assurance") or {}).get("raw_laxest_wash_share"), all_seeds
            ),
        }

        if cfg["compute_hit_rate"]:
            det_total = 0
            aud_total = 0
            per_seed_hr: list[float] = []
            for s in all_seeds:
                d_s, a_s = extract_hit_rate(run_root, sc_recs[s], cfg["window_start"])
                det_total += d_s
                aud_total += a_s
                if a_s > 0:
                    per_seed_hr.append(d_s / a_s)
            desc["audits_mean"] = aud_total / len(all_seeds)
            desc["detections_npz_mean"] = det_total / len(all_seeds)
            desc["hit_rate_pooled"] = det_total / aud_total if aud_total > 0 else None
            desc["hit_rate_mean"] = float(np.mean(per_seed_hr)) if per_seed_hr else None

        # Paired vs-baseline diffs (full n=30 and clean subset)
        if sc_name != base_name:
            diffs_eq = [
                sc_recs[s]["metrics"]["true_change"]["equal_weight_sd"]
                - base_recs[s]["metrics"]["true_change"]["equal_weight_sd"]
                for s in all_seeds
            ]
            diffs_eq_clean = [
                sc_recs[s]["metrics"]["true_change"]["equal_weight_sd"]
                - base_recs[s]["metrics"]["true_change"]["equal_weight_sd"]
                for s in clean_seeds
            ]
            diffs_cap = [
                sc_recs[s]["metrics"]["true_change"]["fixed_start_cap_weight_sd"]
                - base_recs[s]["metrics"]["true_change"]["fixed_start_cap_weight_sd"]
                for s in all_seeds
            ]
            diffs_wash = [
                sc_recs[s]["metrics"]["actions"]["wash"] / sc_recs[s]["metrics"]["actions"]["decisions"]
                - base_recs[s]["metrics"]["actions"]["wash"] / base_recs[s]["metrics"]["actions"]["decisions"]
                for s in all_seeds
            ]
            desc["vs_baseline_eq"] = paired_t_summary(diffs_eq)
            desc["vs_baseline_eq_clean"] = paired_t_summary(diffs_eq_clean)
            desc["vs_baseline_cap"] = paired_t_summary(diffs_cap)
            desc["vs_baseline_wash_share"] = paired_t_summary(diffs_wash)

        # For sec32C1, also compute paired diff vs EB_B for EB scenarios
        if cfg["key"] == "sec32C1_granularity" and sc_name.startswith("EB_") and sc_name != "EB_B":
            eb_base = records_by_sc["EB_B"]
            diffs_eb = [
                sc_recs[s]["metrics"]["true_change"]["equal_weight_sd"]
                - eb_base[s]["metrics"]["true_change"]["equal_weight_sd"]
                for s in all_seeds
            ]
            desc["vs_eb_baseline_eq"] = paired_t_summary(diffs_eb)

        scenarios_out[sc_name] = desc

    # Comparisons from summary.json (keeping key metrics compact, no raw paired_values bloat)
    comparisons_out: dict[str, Any] = {}
    for comp in summary["comparisons"]:
        c_metrics: dict[str, Any] = {}
        for m_name, m_obj in comp["metrics"].items():
            if m_obj.get("status") != "complete":
                continue
            pv = m_obj.get("paired_values", [])
            entry = {
                "mean": m_obj["mean"],
                "se": m_obj["standard_error"],
                "ci_low": m_obj["ci_low"],
                "ci_high": m_obj["ci_high"],
                "p_value": m_obj["p_value"],
                "n": m_obj["n"],
                "pos": int(sum(1 for v in pv if v > 0)),
                "neg": int(sum(1 for v in pv if v < 0)),
            }
            if "holm_p_value" in m_obj:
                entry["holm_p_value"] = m_obj["holm_p_value"]
            c_metrics[m_name] = entry
        comparisons_out[comp["name"]] = {
            "primary": bool(comp.get("primary")),
            "weights": comp["weights"],
            "metrics": c_metrics,
        }

    return {
        "experiment_id": summary["experiment_id"],
        "source_digest": audit["source_digest"],
        "seed_range": [all_seeds[0], all_seeds[-1]],
        "planned_trials": summary["planned_trials"],
        "complete_trials": summary["complete_trials"],
        "trials_failing_core_validation": audit["trials_failing_core_validation"],
        "failed_seeds": audit["failed_seeds"],
        "clean_seed_count": audit["clean_seed_count"],
        "preswitch_parity_pairs_checked": audit["preswitch_parity_pairs_checked"],
        "preswitch_parity_mismatches": audit["preswitch_parity_mismatches"],
        "baseline": base_name,
        "scenario_order": scenario_order,
        "scenarios": scenarios_out,
        "comparisons": comparisons_out,
        "clean_primary_sensitivity": audit["clean_primary_sensitivity"],
    }


def main() -> None:
    out: dict[str, Any] = {
        "schema_version": "mas_upgrade_analysis_v1",
        "generated_from": "runs/disclosure_mas_*_confirmatory_v1/summary.json",
        "studies": {},
    }
    for cfg in STUDIES:
        print(f"Extracting {cfg['key']}...")
        out["studies"][cfg["key"]] = extract_study(cfg)

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size:,} bytes)")

    # Quick sanity verification against README / README_sec32 headline numbers
    s29 = out["studies"]["sec29_stackelberg"]["scenarios"]
    print("\n--- Sanity check §29 hit rates (pooled / mean) ---")
    for k in ["B", "V", "J", "A2", "S_obs", "S_obs_hid", "S_obs_sig", "S_oracle_eq", "S_oracle_hid"]:
        print(f"  {k:13s}: eq={s29[k]['true_change_eq']:.4f}  hr_pool={s29[k]['hit_rate_pooled']*100:.1f}%  hr_mean={s29[k]['hit_rate_mean']*100:.1f}%")

    s32b = out["studies"]["sec32B_phase_pareto"]["scenarios"]
    print("\n--- Sanity check §32B sweep ---")
    for k in out["studies"]["sec32B_phase_pareto"]["scenario_order"]:
        v = s32b[k]
        eff = f"{v['assurance_effort_cost']:.5f}" if v["assurance_effort_cost"] is not None else "—"
        print(f"  {k:10s}: eq={v['true_change_eq']:.4f}  wash={v['wash_share']*100:5.1f}%  res={v['resource_cost_bp']:.2f}bp  eff={eff}")

    s32c2 = out["studies"]["sec32C2_ai_washing"]["scenarios"]
    print("\n--- Sanity check §32C-2 ai_washing ---")
    for k in out["studies"]["sec32C2_ai_washing"]["scenario_order"]:
        v = s32c2[k]
        vs_all = v.get("vs_baseline_eq", {}).get("mean", 0.0)
        vs_cln = v.get("vs_baseline_eq_clean", {}).get("mean", 0.0)
        hr = f"{v['hit_rate_pooled']*100:.1f}%" if v.get("hit_rate_pooled") is not None else "—"
        print(f"  {k:22s}: eq={v['true_change_eq']:.4f}  vsB={vs_all:+.4f} (clean {vs_cln:+.4f})  wash={v['wash_share']*100:5.1f}%  hr={hr}")


if __name__ == "__main__":
    main()
