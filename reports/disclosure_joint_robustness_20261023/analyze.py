"""Joint robustness test analysis (research contract section 24). Read-only on the archive.

  python analyze.py joint_v1                    # confirmatory: claims, predictions, decomposition, tables
  python analyze.py check_v1 --pipeline-only    # development: pipeline checks only (contract 24.8)
  python analyze.py check_v1 --smoke            # development: runs the full analysis, prints and writes no estimate

Writes <run>_analysis.json and <run>_tables.md next to this script (the pipeline-only mode writes
<run>_pipeline.json). Contrasts are per-seed paired differences within this run. The eight primary questions take
the runner's Holm-adjusted p values (contract 24.5); everything else is a paired 95% t interval without
multiplicity adjustment. Decision-level statistics cover the decision days of the policy window (1260-3480).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from _no_wmi import prefill_uname

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
prefill_uname()  # importing scipy.stats can block on a hung WMI service (see _no_wmi.py)
from abm.disclosure.experiments import paired_statistics  # noqa: E402

SWITCH, PERIOD = 1201, 60
WINDOW = slice(1200 // PERIOD + 1, None)   # decision days in (1200, 3500]
INFLATE = 1
SIZES = ("large", "mid", "small")
TRUE, CAP_WEIGHTED = "true_change.equal_weight_sd", "true_change.fixed_start_cap_weight_sd"
POLICIES = ("B", "I", "F")
KAPPAS = {"0.32": "0p32", "0.5": "0p5"}
BLOCKS = {"main": "", "P": "rep_",
          **{f"R{k}": f"rev{t}_" for k, t in KAPPAS.items()},
          **{f"RP{k}": f"rr{t}_" for k, t in KAPPAS.items()},
          **{f"RPW{k}": f"rrw{t}_" for k, t in KAPPAS.items()}}
JOINT = [f"RP{k}" for k in KAPPAS] + [f"RPW{k}" for k in KAPPAS]
# tau*(w_i, kappa) > 0.6 share of firms under the design distribution (contract 24.6 T4).
DESIGN_SHARE_ABOVE_06 = {"0.32": 0.9705, "0.5": 0.9966}


# -- loading ------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows: dict = {}
    for record in summary["records"]:
        rows.setdefault(record["scenario"], {})[record["seed"]] = record
    return summary, rows


def complete(record: dict | None) -> bool:
    return bool(record) and record["status"] == "complete"


def core_valid(record: dict | None) -> bool:
    return complete(record) and all(record["validation"][gate]["status"] == "PASS" for gate in ("mechanical", "stylized"))


def metric(record: dict, key: str) -> float:
    value = record["metrics"]
    for part in key.split("."):
        value = value[part]
    return float(value)


class Arrays:
    """Result arrays of completed trials, loaded once (compact archives keep everything used here)."""

    NAMES = ("actions", "amounts", "inflation", "financing_rate", "detections", "market_price_cap_hits", "wash_cost_ratio")

    def __init__(self, run: Path):
        self.run, self.cache = run, {}

    def get(self, record: dict) -> dict:
        key = (record["scenario"], record["seed"])
        if key not in self.cache:
            with np.load(self.run / record["path"] / "result.npz") as data:
                self.cache[key] = {name: data[name] for name in self.NAMES if name in data.files}
        return self.cache[key]


def stat(values: list[float]) -> dict | None:
    if len(values) < 2:
        return None
    out = paired_statistics(values, 0.95)
    return {key: out[key] for key in ("mean", "standard_error", "ci_low", "ci_high", "p_value", "n")}


def first_cap_day(data: dict) -> int | None:
    hits = np.flatnonzero(data["market_price_cap_hits"])
    return int(hits[0]) if hits.size else None


# -- per-trial quantities -----------------------------------------------------------

def trial(record: dict, data: dict) -> dict:
    actions = record["metrics"]["actions"]
    out = {"true_change": metric(record, TRUE), "inflation_mean": metric(record, "inflation.mean_sd"),
           "wash_share": actions["wash"] / actions["decisions"], "washes": int(actions["wash"]),
           **{f"wash_share_{size}": float(actions["wash_share_by_size"][size]) for size in SIZES},
           "fines_fraction_initial_cap": metric(record, "costs.fine_transfer_fraction_initial_cap")}
    wash = data["actions"][WINDOW] == INFLATE
    # Inflation recorded on a decision day is after that day's audit, i.e. what the firm holds when deciding.
    holding = data["inflation"][::PERIOD][WINDOW] > 0
    out["continuation_share_of_washing"] = float((wash & holding).sum() / wash.sum()) if wash.any() else None
    out["washes_at_period_cap"] = int((wash & (data["amounts"][WINDOW] >= 0.25 - 1e-12)).sum())
    out["detections"] = int(data["detections"][WINDOW].sum())
    return out


def paired(cells: dict, plus: str, minus: str, key: str, seeds: list[int]) -> dict | None:
    values = [cells[plus][s][key] - cells[minus][s][key] for s in seeds
              if s in cells[plus] and s in cells[minus] and cells[plus][s][key] is not None
              and cells[minus][s][key] is not None]
    return stat(values)


# -- pipeline checks ---------------------------------------------------------------

def pipeline_checks(run: Path, summary: dict, rows: dict, arrays: Arrays, seeds: list[int]) -> dict:
    windows = {(r["metrics"]["window"]["start_day"], r["metrics"]["window"]["end_day"])
               for scenario in rows.values() for r in scenario.values() if complete(r)}
    counts: Counter = Counter()
    for record in summary["records"]:
        for attempt in record["attempts"]:
            if attempt["status"] != "complete":
                manifest = run / attempt["path"] / "manifest.json"
                kind = (json.loads(manifest.read_text(encoding="utf-8")).get("error_type", "unknown")
                        if manifest.exists() else "no manifest")
                counts[f"{attempt['status']}: {kind}"] += 1
    pairing = {}
    for block, prefix in BLOCKS.items():
        compared = differing = 0
        for seed in seeds:
            series = [arrays.get(rows[prefix + name][seed]) for name in POLICIES if complete(rows[prefix + name].get(seed))]
            series = [(d["inflation"][:SWITCH], d["financing_rate"][:SWITCH], d["actions"][:SWITCH // PERIOD + 1])
                      for d in series]
            for other in series[1:]:
                compared += 1
                differing += not all(np.array_equal(x, y) for x, y in zip(series[0], other))
        pairing[block] = {"pairs_compared": compared, "pairs_differing_before_switch": differing}
    ratios = {}
    for seed in seeds:
        for kappa in KAPPAS:
            drawn = [arrays.get(rows[BLOCKS[f"RPW{kappa}"] + name][seed])["wash_cost_ratio"] for name in POLICIES
                     if complete(rows[BLOCKS[f"RPW{kappa}"] + name].get(seed))]
            if drawn:
                ratios[f"{seed}_{kappa}"] = {
                    "identical_across_scenarios": all(np.array_equal(drawn[0], d) for d in drawn[1:]),
                    "mean": float(drawn[0].mean()), "log_sd": float(np.log(drawn[0]).std(ddof=1))}
    others = [arrays.get(rows[prefix + name][seed]) for block, prefix in BLOCKS.items() if not block.startswith("RPW")
              for name in POLICIES for seed in seeds if complete(rows[prefix + name].get(seed))]
    return {"experiment_id": summary["experiment_id"], "planned": summary["planned_trials"],
            "complete": summary["complete_trials"],
            "failed": [{"scenario": r["scenario"], "seed": r["seed"], "error": r.get("error")}
                       for r in summary["records"] if r["status"] != "complete"],
            "attempts_not_complete": dict(sorted(counts.items())),
            "failing_core_validation": summary["trials_failing_core_validation"],
            "core_validation_failures": sorted(
                [{"scenario": r["scenario"], "seed": r["seed"],
                  "price_cap_hits": r["validation"]["mechanical"]["price_cap_hits"],
                  "first_cap_day": first_cap_day(arrays.get(r))}
                 for scenario in rows.values() for r in scenario.values() if complete(r) and not core_valid(r)],
                key=lambda f: (f["scenario"], f["seed"])),
            "metric_windows": sorted(windows), "pre_switch_pairing": pairing, "wash_cost_ratios": ratios,
            "wash_cost_ratio_absent_outside_triple_block": all("wash_cost_ratio" not in d for d in others)}


# -- judgments ----------------------------------------------------------------------

def claim(row: dict) -> str:
    """Contract 24.5: every primary question predicts a positive contrast."""
    holm = row["holm_p_value"]
    if holm is None or holm >= 0.05:
        return "unresolved"
    return "supported" if row["mean"] > 0 else "unsupported"


def primary_questions(summary: dict, rows: dict, seeds: list[int]) -> dict:
    out = {}
    for comparison in summary["comparisons"]:
        if not comparison.get("primary"):
            continue
        result = comparison["metrics"][TRUE]
        row = {key: result.get(key) for key in ("mean", "ci_low", "ci_high", "p_value", "holm_p_value", "n")}
        row["claim"] = claim(row) if result["status"] == "complete" else "unresolved"
        valid = [s for s in seeds if all(core_valid(rows[name].get(s)) for name in comparison["weights"])]
        row["without_core_validation_failures"] = stat(
            [sum(w * metric(rows[name][s], TRUE) for name, w in comparison["weights"].items()) for s in valid])
        out[comparison["name"]] = row
    return out


def ranking(primary: dict) -> dict:
    out = {}
    for block in JOINT:
        prefix = BLOCKS[block]
        i_f, f_b = primary[f"J_{prefix}I_minus_F"]["claim"], primary[f"J_{prefix}F_minus_B"]["claim"]
        if i_f == f_b == "supported":
            verdict = "排序保留"
        elif "unsupported" in (i_f, f_b):
            verdict = "；".join(text for claim_, text in ((i_f, "I 与 F 反转"), (f_b, "F 与 B 反转"))
                               if claim_ == "unsupported")
        else:
            verdict = "未分辨"
        out[block] = {"I_minus_F": i_f, "F_minus_B": f_b, "verdict": verdict}
    return out


def below_zero(s: dict | None) -> bool:
    return s is not None and s["ci_high"] < 0


def above_zero(s: dict | None) -> bool:
    return s is not None and s["ci_low"] > 0


def predictions(cells: dict, rows: dict, arrays: Arrays, seeds: list[int]) -> dict:
    out: dict = {}
    # T1: wash share I < F < B in every joint setting; washes at the period cap are reported as the exception.
    t1 = {}
    for block in JOINT:
        p = BLOCKS[block]
        t1[block] = {"I_minus_F": paired(cells, p + "I", p + "F", "wash_share", seeds),
                     "F_minus_B": paired(cells, p + "F", p + "B", "wash_share", seeds),
                     "I_washes_at_period_cap": sum(cells[p + "I"][s]["washes_at_period_cap"] for s in cells[p + "I"]),
                     "I_washes": sum(cells[p + "I"][s]["washes"] for s in cells[p + "I"])}
    failing = [f"{b} {k}" for b, row in t1.items() for k in ("I_minus_F", "F_minus_B") if not below_zero(row[k])]
    out["T1"] = {"settings": t1, "not_met": failing, "verdict": "成立" if not failing else "不成立"}
    # T2: every seed has washing under I in every joint setting.
    t2 = {block: sum(cells[BLOCKS[block] + "I"][s]["washes"] > 0 for s in cells[BLOCKS[block] + "I"]) for block in JOINT}
    out["T2"] = {"seeds_with_washing_under_I": t2, "seeds": len(seeds),
                 "verdict": "成立" if all(v == len(seeds) for v in t2.values()) else "不成立"}
    # T3: per-report fines cut washing under I relative to R only.
    t3 = {k: paired(cells, f"rr{t}_I", f"rev{t}_I", "wash_share", seeds) for k, t in KAPPAS.items()}
    out["T3"] = {"rr_minus_rev_I_wash_share": t3, "verdict": "成立" if all(below_zero(v) for v in t3.values()) else "不成立"}
    # T4: heterogeneous w raises washing under I and F.
    t4 = {f"{k}_{name}": paired(cells, f"rrw{t}_{name}", f"rr{t}_{name}", "wash_share", seeds)
          for k, t in KAPPAS.items() for name in ("I", "F")}
    shares = {}
    for kappa, tag in KAPPAS.items():
        values = []
        for s in seeds:
            record = rows[f"rrw{tag}_I"].get(s)
            if complete(record):
                ratios = arrays.get(record)["wash_cost_ratio"]
                values.append(float((ratios < ratio_cutoff(kappa, 0.6)).mean()))
        shares[kappa] = {"drawn": float(np.mean(values)) if values else None, "design": DESIGN_SHARE_ABOVE_06[kappa]}
    out["T4"] = {"rrw_minus_rr_wash_share": t4, "share_of_firms_with_threshold_above_0p6": shares,
                 "not_met": [k for k, v in t4.items() if not above_zero(v)],
                 "verdict": "成立" if all(above_zero(v) for v in t4.values()) else "不成立"}
    return out


def ratio_cutoff(kappa: str, tau: float) -> float:
    """tau*(w_i) = 1 - sqrt(w_i) A_r(kappa) / A_w exceeds tau iff w_i is below this (contract 24.6)."""
    real = {"0.32": 2.0077, "0.5": 1.5100}[kappa]
    return ((1 - tau) * 3.8305 / real) ** 2


def secondary(summary: dict) -> dict:
    return {comparison["name"]: {key: {field: comparison["metrics"][key].get(field) for field in ("mean", "ci_low", "ci_high", "n")}
                                 for key in (TRUE, CAP_WEIGHTED)}
            for comparison in summary["comparisons"] if not comparison.get("primary")}


def describe(cells: dict, seeds: list[int]) -> dict:
    out = {}
    for name, by_seed in cells.items():
        keys = sorted({key for values in by_seed.values() for key in values})
        row = {}
        for key in keys:
            values = [by_seed[s][key] for s in seeds if s in by_seed and by_seed[s].get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        row["seeds_with_washing"] = sum(by_seed[s]["washes"] > 0 for s in by_seed)
        row["complete"] = len(by_seed)
        out[name] = row
    return out


# -- tables --------------------------------------------------------------------------

def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−")


def interval(s: dict | None, digits: int = 3) -> str:
    return "—" if s is None else f"{fmt(s['mean'], digits)} [{fmt(s['ci_low'], digits)}, {fmt(s['ci_high'], digits)}]"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def tables(result: dict) -> list[str]:
    out = ["# 联合稳健性表格", "", f"experiment_id `{result['experiment_id']}`", "",
           "## 主要问题（政策窗口第 1200–3500 日等权真实变化，SD，配对 95% t 区间，Holm 校正）", "",
           "| 问题 | 均值 [区间] | Holm p | 判定 | 剔除核心验证失败后 |", "|---|---|---:|---|---|"]
    for name, row in result["primary"].items():
        holm = row["holm_p_value"]
        out.append(f"| {name} | {interval(row)} | {'—' if holm is None else f'{holm:.3g}'} | {row['claim']} | "
                   f"{interval(row['without_core_validation_failures'])} |")
    out += ["", "| 设定 | I − F | F − B | 排序 |", "|---|---|---|---|"]
    for block, row in result["ranking"].items():
        out.append(f"| {block} | {row['I_minus_F']} | {row['F_minus_B']} | {row['verdict']} |")
    out += ["", "## 辅助对比（未校正）", "", "| 对比 | 等权 | 市值加权 |", "|---|---|---|"]
    for name, row in result["secondary"].items():
        out.append(f"| {name} | {interval(row[TRUE])} | {interval(row[CAP_WEIGHTED])} |")
    p = result["predictions"]
    out += ["", "## 解析预测（描述性）", "", "| 设定 | 漂绿占比 I − F | 漂绿占比 F − B | I 中幅度顶到上限的漂绿 |", "|---|---|---|---|"]
    for block, row in p["T1"]["settings"].items():
        out.append(f"| {block} | {interval(row['I_minus_F'], 4)} | {interval(row['F_minus_B'], 4)} | "
                   f"{row['I_washes_at_period_cap']}/{row['I_washes']} |")
    out += ["", f"- T1：{p['T1']['verdict']}" + (f"（未满足：{', '.join(p['T1']['not_met'])}）" if p["T1"]["not_met"] else "") + "。",
            f"- T2：I 中有漂绿的种子 " + "，".join(f"{k} {v}/{p['T2']['seeds']}" for k, v in p["T2"]["seeds_with_washing_under_I"].items())
            + f"：{p['T2']['verdict']}。",
            "- T3：I 的漂绿占比 rr − rev " + "，".join(f"κ = {k}：{interval(v, 4)}" for k, v in p["T3"]["rr_minus_rev_I_wash_share"].items())
            + f"：{p['T3']['verdict']}。",
            "- T4：漂绿占比 rrw − rr " + "，".join(f"{k}：{interval(v, 4)}" for k, v in p["T4"]["rrw_minus_rr_wash_share"].items())
            + f"：{p['T4']['verdict']}。门槛高于 0.6 的企业比例（实际抽样 / 设计值）："
            + "，".join(f"κ = {k} {pct(v['drawn'])} / {pct(v['design'])}" for k, v in p["T4"]["share_of_firms_with_threshold_above_0p6"].items()) + "。",
            "", "## 描述（种子均值）", "",
            "| 场景 | 真实变化 | 漂绿占比 | 大 / 中 / 小 | 继续已有案件 | 平均虚报 | 有漂绿的种子 | 罚款（占初始市值） |",
            "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, row in result["describe"].items():
        out.append(f"| {name} | {fmt(row['true_change'])} | {pct(row['wash_share'])} | "
                   f"{' / '.join(pct(row[f'wash_share_{s}']) for s in SIZES)} | {pct(row['continuation_share_of_washing'])} | "
                   f"{row['inflation_mean']:.3f} | {row['seeds_with_washing']} | {row['fines_fraction_initial_cap']:.2e} |")
    failures = result["pipeline_checks"]["core_validation_failures"]
    out += ["", "## 核心验证未通过的运行", ""]
    out += ["| 场景 | 种子 | 截断次数 | 首次截断日 |", "|---|---:|---:|---:|"] + [
        f"| {f['scenario']} | {f['seed']} | {f['price_cap_hits']} | {f['first_cap_day']} |" for f in failures] if failures else ["无。"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory name next to this script, e.g. joint_v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pipeline-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run = HERE / args.run
    summary, rows = load(run)
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    arrays = Arrays(run)
    checks = pipeline_checks(run, summary, rows, arrays, seeds)
    if args.pipeline_only:
        (HERE / f"{args.run}_pipeline.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in checks.items() if k != "wash_cost_ratios"}, indent=2, ensure_ascii=False))
        return
    cells = {name: {seed: trial(record, arrays.get(record)) for seed, record in by_seed.items() if complete(record)}
             for name, by_seed in rows.items()}
    primary = primary_questions(summary, rows, seeds)
    result = {"experiment_id": frozen["experiment_id"], "purpose": frozen["spec"]["purpose"], "pipeline_checks": checks,
              "primary": primary, "ranking": ranking(primary), "secondary": secondary(summary),
              "predictions": predictions(cells, rows, arrays, seeds), "describe": describe(cells, seeds),
              "inference": "Paired 95% t intervals across seeds; the eight primary contrasts use the runner's Holm "
                           "adjustment. All completed runs are used; core-validation failures are kept and reported "
                           "separately."}
    lines = tables(result)
    if args.smoke:
        print(f"full analysis ran: {len(result['primary'])} primary, {len(result['secondary'])} secondary, "
              f"{len(result['predictions'])} predictions, {len(result['describe'])} scenarios, {len(lines)} table lines; "
              "nothing written")
        return
    (HERE / f"{args.run}_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8", newline="\n")
    (HERE / f"{args.run}_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.run}_analysis.json and {args.run}_tables.md")


if __name__ == "__main__":
    main()
