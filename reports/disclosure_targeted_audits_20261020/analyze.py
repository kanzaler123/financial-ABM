"""Targeted-audit experiment analysis (research contract section 21). Read-only on the archive.

  python analyze.py audits_v1      # confirmatory: claim records, predictions, descriptives, tables
  python analyze.py check_v1       # development: runs the same code; its estimates are not for reading

Writes <run>_analysis.json and <run>_tables.md next to this script. Contrasts are per-seed paired differences;
the primary questions E1-E5 take the runner's Holm-adjusted p values (contract 21.5), everything else is a paired
95% t interval without multiplicity adjustment. Decision-level statistics cover decisions on days 1260-3480,
the decision days of the policy window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from _no_wmi import prefill_uname

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
prefill_uname()  # importing scipy.stats can block on a hung WMI service (see _no_wmi.py)
from abm.disclosure.experiments import paired_statistics  # noqa: E402

PERIOD = 60
WINDOW = slice(21, 59)        # decision (and audit) periods on days 1260..3480
INFLATE = 1
SIZES = ("large", "mid", "small")
DIRECTION = {"E1_V_minus_B": None, "E2_J_minus_B": None, "E3_J_minus_J_hidden": -1, "E4_V_minus_B_size": 1,
             "E5_I_minus_VF": None}
DESIGN = json.loads((HERE / "design_numbers.json").read_text(encoding="utf-8"))
STOP_RATE = DESIGN["factors"]["rate_at_which_tau_star_equals"]["0.1"]   # 10.49%: tau*(q) = 0.1 (T1)
TARGETED = ("V", "V_hidden", "J", "J_hidden", "VF", "decay0_V", "decay0_J", "decay0_J_hidden", "decay0_VF")


# -- loading ------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows: dict = {}
    for record in summary["records"]:
        rows.setdefault(record["scenario"], {})[record["seed"]] = record
    return summary, rows


def complete(record: dict | None) -> bool:
    return bool(record) and record["status"] == "complete"


def core_valid(record: dict) -> bool:
    return complete(record) and all(record["validation"][gate]["status"] == "PASS" for gate in ("mechanical", "stylized"))


def metric(record: dict, key: str) -> float:
    value = record["metrics"]
    for part in key.split("."):
        value = value[part]
    return float(value)


class Arrays:
    """Result arrays of completed trials, loaded once (compact archives keep everything used here)."""

    NAMES = ("actions", "amounts", "inflation", "detections", "audit_selected", "audit_probability", "audit_flags",
             "market_size_class", "market_price_cap_hits")

    def __init__(self, run: Path):
        self.run, self.cache = run, {}

    def get(self, record: dict) -> dict:
        key = (record["scenario"], record["seed"])
        if key not in self.cache:
            with np.load(self.run / record["path"] / "result.npz") as data:
                self.cache[key] = {name: data[name] for name in self.NAMES if name in data.files}
        return self.cache[key]


def stat(values: list[float]) -> dict:
    out = paired_statistics(values, 0.95)
    return {key: out[key] for key in ("mean", "standard_error", "ci_low", "ci_high", "p_value", "n")}


# -- per-trial quantities -----------------------------------------------------------

def trial(record: dict, data: dict) -> dict:
    """Decision-level shares and audit statistics of one trial over the policy window."""
    actions = record["metrics"]["actions"]
    out = {"wash_share": actions["wash"] / actions["decisions"],
           **{f"wash_share_{size}": float(actions["wash_share_by_size"][size]) for size in SIZES}}
    out["wash_share_large_minus_small"] = out["wash_share_large"] - out["wash_share_small"]
    size = data["market_size_class"]
    wash = data["actions"][WINDOW] == INFLATE
    # Inflation recorded on a decision day is after that day's audit, i.e. what the firm holds when deciding.
    holding = data["inflation"][::PERIOD][WINDOW] > 0
    selected = data["audit_selected"][WINDOW].astype(bool)
    caught = data["detections"][WINDOW].astype(bool)
    out["open_cases_per_decision_day"] = float(holding.sum(axis=1).mean())
    out["audits_per_audit_day"] = float(selected.sum(axis=1).mean())
    out["hit_rate"] = float(caught.sum() / selected.sum()) if selected.any() else None
    for index, name in enumerate(SIZES):
        out[f"audit_share_{name}"] = float(selected[:, size == index].sum() / selected.sum()) if selected.any() else None
    if "audit_probability" in data:
        probability = data["audit_probability"][WINDOW]
        for index, name in enumerate(SIZES):
            out[f"mean_q_{name}"] = float(probability[:, size == index].mean())
    if "audit_flags" in data:
        flags = data["audit_flags"].astype(bool)
        window_flags = flags[WINDOW]
        previous_wash = data["actions"][WINDOW.start - 1:WINDOW.stop - 1] == INFLATE
        out["flags_per_audit_day"] = float(window_flags.sum(axis=1).mean())
        out["flagged_after_washing"] = float((window_flags & previous_wash).sum() / window_flags.sum()) if window_flags.any() else None
        # Inflation decided in one period and flagged at the next audit (T4).
        decided = data["actions"][WINDOW.start:WINDOW.stop - 1] == INFLATE
        next_flags = flags[WINDOW.start + 1:WINDOW.stop]
        out["flag_rate_after_washing"] = float((decided & next_flags).sum() / decided.sum()) if decided.any() else None
    out["fines_fraction_initial_cap"] = metric(record, "costs.fine_transfer_fraction_initial_cap")
    out["continuing_wash"] = int((wash & holding).sum())
    return out


def first_cap_day(data: dict) -> int | None:
    hits = np.flatnonzero(data["market_price_cap_hits"])
    return int(hits[0]) if hits.size else None


def paired(cells: dict, plus: str, minus: str, key: str, seeds: list[int]) -> dict | None:
    values = [cells[plus][s][key] - cells[minus][s][key] for s in seeds
              if s in cells[plus] and s in cells[minus] and cells[plus][s][key] is not None and cells[minus][s][key] is not None]
    return stat(values) if len(values) >= 2 else None


# -- judgments ----------------------------------------------------------------------

def claim(name: str, row: dict) -> str:
    holm, mean, direction = row["holm_p_value"], row["mean"], DIRECTION[name]
    if holm is None:
        return "unresolved"
    if holm >= 0.05:
        return "unresolved"
    if direction is None or np.sign(mean) == direction:
        return "supported"
    return "unsupported"


def primary_questions(summary: dict, rows: dict, seeds: list[int]) -> dict:
    out = {}
    primary_metric = summary_metric = None
    for comparison in summary["comparisons"]:
        if not comparison.get("primary"):
            continue
        key = comparison.get("primary_metric", "true_change.equal_weight_sd")
        result = comparison["metrics"][key]
        row = {key2: result.get(key2) for key2 in ("mean", "ci_low", "ci_high", "p_value", "holm_p_value", "n")}
        row["metric"] = key
        row["claim"] = claim(comparison["name"], row) if result["status"] == "complete" else "unresolved"
        row["core_validation_pass"] = comparison["core_validation_pass"]
        # The same contrast without trials failing core validation (contract 21.8).
        valid = [s for s in seeds if all(core_valid(rows[name].get(s)) for name in comparison["weights"])]
        values = [sum(weight * metric(rows[name][s], key) for name, weight in comparison["weights"].items()) for s in valid]
        row["without_core_validation_failures"] = stat(values) if len(values) >= 2 else None
        out[comparison["name"]] = row
    del primary_metric, summary_metric
    return out


def predictions(cells: dict, arrays: Arrays, rows: dict, seeds: list[int]) -> dict:
    out = {}
    # T1: continuing inflation under V comes from firms below the stopping rate.
    below = above = 0
    above_at_cap = 0
    firm_periods_above = 0
    for seed in seeds:
        record = rows["V"].get(seed)
        if not complete(record):
            continue
        data = arrays.get(record)
        wash = data["actions"][WINDOW] == INFLATE
        holding = data["inflation"][::PERIOD][WINDOW] > 0
        probability = data["audit_probability"][WINDOW]
        high = probability >= STOP_RATE
        firm_periods_above += int(high.sum())
        continuing = wash & holding
        below += int((continuing & ~high).sum())
        above += int((continuing & high).sum())
        above_at_cap += int((continuing & high & (data["amounts"][WINDOW] >= 0.25 - 1e-9)).sum())
    share = below / (below + above) if below + above else None
    out["T1"] = {"stop_rate": STOP_RATE, "continuing_washing_below": below, "continuing_washing_at_or_above": above,
                 "at_or_above_with_amount_at_period_cap": above_at_cap, "share_below": share,
                 "firm_periods_at_or_above": firm_periods_above,
                 "verdict": None if share is None else ("成立" if share >= 0.99 else "不成立"),
                 "note": None if firm_periods_above else "没有企业—期达到停止所需的核查率，本项没有检验停止的一侧"}
    # T2: V lowers large firms' washing share and raises small firms'.
    large, small = paired(cells, "V", "B", "wash_share_large", seeds), paired(cells, "V", "B", "wash_share_small", seeds)
    hits = (large is not None and large["ci_high"] < 0) + (small is not None and small["ci_low"] > 0)
    out["T2"] = {"large_V_minus_B": large, "small_V_minus_B": small,
                 "verdict": {2: "成立", 1: "部分", 0: "不成立"}[hits]}
    # T3: a public jump rule raises small firms' washing share.
    small = paired(cells, "J", "J_hidden", "wash_share_small", seeds)
    out["T3"] = {"small_J_minus_J_hidden": small,
                 "verdict": None if small is None else ("成立" if small["ci_low"] > 0 else "不成立")}
    # T4: firms that know the jump rule are flagged less often after inflating.
    out["T4"] = {}
    for label, (plus, minus) in {"decay 0.2": ("J", "J_hidden"), "no decay": ("decay0_J", "decay0_J_hidden")}.items():
        flag = paired(cells, plus, minus, "flag_rate_after_washing", seeds)
        out["T4"][label] = {"flag_rate_public_minus_hidden": flag,
                            "verdict": None if flag is None else ("成立" if flag["ci_high"] < 0 else "不成立")}
    # T5: doubling random audits lowers washing in every size class; large firms still wash more than small.
    by_size = {size: paired(cells, "A2", "B", f"wash_share_{size}", seeds) for size in SIZES}
    gap = stat([cells["A2"][s]["wash_share_large_minus_small"] for s in seeds if s in cells["A2"]])
    hits = sum(row is not None and row["ci_high"] < 0 for row in by_size.values()) + (gap["ci_low"] > 0)
    out["T5"] = {"A2_minus_B_by_size": by_size, "A2_large_minus_small_share": gap,
                 "verdict": "成立" if hits == 4 else ("部分" if hits else "不成立")}
    return out


def secondary(summary: dict) -> dict:
    out = {}
    for comparison in summary["comparisons"]:
        if comparison.get("primary"):
            continue
        out[comparison["name"]] = {
            key: {field: comparison["metrics"][key].get(field) for field in ("mean", "ci_low", "ci_high", "n")}
            for key in ("true_change.equal_weight_sd", "true_change.fixed_start_cap_weight_sd", "size.large_minus_small_sd")}
    return out


def describe(cells: dict, rows: dict, arrays: Arrays, seeds: list[int]) -> dict:
    out = {}
    for name, by_seed in cells.items():
        keys = sorted({key for values in by_seed.values() for key in values})
        row = {}
        for key in keys:
            values = [by_seed[s][key] for s in seeds if s in by_seed and by_seed[s].get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        row["true_change"] = float(np.mean([metric(rows[name][s], "true_change.equal_weight_sd") for s in seeds
                                            if complete(rows[name].get(s))]))
        row["complete"] = len(by_seed)
        row["core_validation_failures"] = [
            {"seed": s, "first_cap_day": first_cap_day(arrays.get(rows[name][s]))}
            for s in seeds if complete(rows[name].get(s)) and not core_valid(rows[name][s])]
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
    out = ["# 定向核查表格", "", f"experiment_id `{result['experiment_id']}`", "",
           "## 主要问题（政策窗口第 1200–3500 日，配对 95% t 区间，Holm 校正）", "",
           "| 问题 | 指标 | 均值 [区间] | Holm p | 判定 | 剔除核心验证失败后 |", "|---|---|---|---:|---|---|"]
    for name, row in result["primary"].items():
        holm = row["holm_p_value"]
        out.append(f"| {name} | `{row['metric']}` | {interval(row)} | {'—' if holm is None else f'{holm:.3g}'} | "
                   f"{row['claim']} | {interval(row['without_core_validation_failures'])} |")
    out += ["", "## 辅助对比（未校正）", "", "| 对比 | 等权真实变化 | 市值加权真实变化 | 大减小 |", "|---|---|---|---|"]
    for name, row in result["secondary"].items():
        out.append(f"| {name} | " + " | ".join(interval(row[key]) for key in (
            "true_change.equal_weight_sd", "true_change.fixed_start_cap_weight_sd", "size.large_minus_small_sd")) + " |")
    out += ["", "## 描述（种子均值）", "",
            "| 场景 | 真实变化 | 漂绿决策 | 大 / 中 / 小 | 未查出案件 | 核查/日 | 命中率 | 核查分到大/中/小 | 标记/日 | 罚款（初始市值比例） |",
            "|---|---:|---:|---|---:|---:|---:|---|---:|---:|"]
    for name, row in result["describe"].items():
        sizes = " / ".join(pct(row.get(f"wash_share_{size}")) for size in SIZES)
        audits = " / ".join(pct(row.get(f"audit_share_{size}")) for size in SIZES)
        flags = row.get("flags_per_audit_day")
        out.append(f"| {name} | {fmt(row['true_change'])} | {pct(row['wash_share'])} | {sizes} | "
                   f"{row['open_cases_per_decision_day']:.1f} | {row['audits_per_audit_day']:.2f} | {pct(row['hit_rate'])} | "
                   f"{audits} | {'—' if flags is None else f'{flags:.1f}'} | {row['fines_fraction_initial_cap']:.2e} |")
    p = result["predictions"]
    out += ["", "## 解析预测", "",
            f"- T1：V 下继续虚报决策中来自 q < {100 * p['T1']['stop_rate']:.2f}% 的企业占 "
            f"{pct(p['T1']['share_below'])}（{p['T1']['continuing_washing_below']} 对 {p['T1']['continuing_washing_at_or_above']}），"
            f"达到该核查率的企业—期 {p['T1']['firm_periods_at_or_above']} 个：{p['T1']['verdict']}。{p['T1']['note'] or ''}",
            f"- T2：V − B 大企业漂绿占比 {interval(p['T2']['large_V_minus_B'])}，小企业 {interval(p['T2']['small_V_minus_B'])}：{p['T2']['verdict']}。",
            f"- T3：J − J_hidden 小企业漂绿占比 {interval(p['T3']['small_J_minus_J_hidden'])}：{p['T3']['verdict']}。"]
    for label, row in p["T4"].items():
        out.append(f"- T4（{label}）：虚报后被标记的比例，公布减不公布 {interval(row['flag_rate_public_minus_hidden'])}：{row['verdict']}。")
    t5 = p["T5"]
    out.append("- T5：A2 − B 漂绿占比 大 " + interval(t5["A2_minus_B_by_size"]["large"]) + "、中 "
               + interval(t5["A2_minus_B_by_size"]["mid"]) + "、小 " + interval(t5["A2_minus_B_by_size"]["small"])
               + f"；A2 中大减小 {interval(t5['A2_large_minus_small_share'])}：{t5['verdict']}。")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory name next to this script, e.g. audits_v1")
    args = parser.parse_args()
    run = HERE / args.run
    summary, rows = load(run)
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    arrays = Arrays(run)
    cells = {name: {seed: trial(record, arrays.get(record)) for seed, record in by_seed.items() if complete(record)}
             for name, by_seed in rows.items()}
    result = {"experiment_id": frozen["experiment_id"], "purpose": frozen["spec"]["purpose"],
              "complete_trials": summary["complete_trials"], "planned_trials": summary["planned_trials"],
              "trials_failing_core_validation": summary["trials_failing_core_validation"],
              "primary": primary_questions(summary, rows, seeds), "secondary": secondary(summary),
              "predictions": predictions(cells, arrays, rows, seeds), "describe": describe(cells, rows, arrays, seeds)}
    (HERE / f"{args.run}_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8", newline="\n")
    (HERE / f"{args.run}_tables.md").write_text("\n".join(tables(result)) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.run}_analysis.json and {args.run}_tables.md")


if __name__ == "__main__":
    main()
