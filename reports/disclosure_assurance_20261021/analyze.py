"""Assurance experiment analysis (research contract section 22). Read-only on the archive.

  python analyze.py assurance_v1   # confirmatory: claim records, predictions, descriptives, tables
  python analyze.py check_v1       # development: runs the same code; its estimates are not for reading

Writes <run>_analysis.json and <run>_tables.md next to this script. Contrasts are per-seed paired differences;
the primary questions C1-C5 take the runner's Holm-adjusted p values (contract 22.8), everything else is a paired
95% t interval without multiplicity adjustment. Decision-level statistics cover decisions on days 1260-3480,
the decision days of the policy window, which are also the assurance periods.
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

PERIOD, FIRST, LAST, YEAR, CAP = 60, 21, 58, 4, 0.25
WINDOW = slice(FIRST, LAST + 1)   # decision periods on days 1260..3480
INFLATE = 1
SIZES = ("large", "mid", "small")
DIRECTION = {"C1_ALL_R16_minus_B": 1, "C2_ALL_F16_minus_ALL_R16": -1, "C3_ALL_R1_minus_B": None,
             "C4_LRG_R16_minus_B_size": 1, "C5_I_minus_ALL_R16": None}
DESIGN = json.loads((HERE / "design_numbers.json").read_text(encoding="utf-8"))
STOP_QUALITY = DESIGN["factors"]["assurance_probability_at_which_tau_star_equals"]["0.1"]   # 0.0827 (T3)


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

    NAMES = ("actions", "amounts", "inflation", "detections", "market_size_class", "market_price_cap_hits",
             "assurer_assignment", "assurance_checked_inflated", "assurance_corrections", "assurer_quality",
             "assurer_inflation_share", "assurer_liability_events")

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


def firm_quality(data: dict) -> np.ndarray:
    """Each firm's check probability in each decision period: its provider's quality, or 0."""
    assignment = data["assurer_assignment"]
    quality = np.take_along_axis(data["assurer_quality"], np.maximum(assignment, 0), axis=1)
    return np.where(assignment >= 0, quality, 0.)


# -- per-trial quantities -----------------------------------------------------------

def trial(record: dict, data: dict) -> dict:
    """Decision-level shares and assurance statistics of one trial over the policy window."""
    actions = record["metrics"]["actions"]
    out = {"wash_share": actions["wash"] / actions["decisions"],
           **{f"wash_share_{size}": float(actions["wash_share_by_size"][size]) for size in SIZES}}
    out["wash_share_large_minus_small"] = out["wash_share_large"] - out["wash_share_small"]
    wash = data["actions"][WINDOW] == INFLATE
    # Inflation recorded on a decision day is after that day's audit, i.e. what the firm holds when deciding.
    holding = data["inflation"][::PERIOD][WINDOW] > 0
    out["open_cases_per_decision_day"] = float(holding.sum(axis=1).mean())
    out["wash_share_without_open_case"] = float((wash & ~holding).sum() / (~holding).sum()) if (~holding).any() else None
    out["fines_fraction_initial_cap"] = metric(record, "costs.fine_transfer_fraction_initial_cap")
    if "assurer_assignment" in data:
        assignment = data["assurer_assignment"][WINDOW]
        checked = assignment >= 0
        out["checked_firms"] = float(checked.sum(axis=1).mean())
        out["checks_with_inflation"] = float(data["assurance_checked_inflated"][WINDOW].sum(axis=1).mean())
        out["corrections"] = float(data["assurance_corrections"][WINDOW].sum(axis=1).mean())
        out["liability_events_total"] = int(data["assurer_liability_events"][WINDOW].sum())
        out["mean_quality"] = float(data["assurer_quality"][WINDOW].mean())
        out["mean_client_quality"] = float(firm_quality(data)[WINDOW][checked].mean()) if checked.any() else None
        starts = list(range(FIRST + YEAR, LAST + 1, YEAR))
        out["switches_per_year"] = float(np.mean([(data["assurer_assignment"][s] != data["assurer_assignment"][s - 1])
                                                  [data["assurer_assignment"][s - 1] >= 0].sum() for s in starts]))
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
    if holm is None or holm >= 0.05:
        return "unresolved"
    if direction is None or np.sign(mean) == direction:
        return "supported"
    return "unsupported"


def primary_questions(summary: dict, rows: dict, seeds: list[int]) -> dict:
    out = {}
    for comparison in summary["comparisons"]:
        if not comparison.get("primary"):
            continue
        key = comparison.get("primary_metric", "true_change.equal_weight_sd")
        result = comparison["metrics"][key]
        row = {key2: result.get(key2) for key2 in ("mean", "ci_low", "ci_high", "p_value", "holm_p_value", "n")}
        row["metric"] = key
        row["claim"] = claim(comparison["name"], row) if result["status"] == "complete" else "unresolved"
        row["core_validation_pass"] = comparison["core_validation_pass"]
        # The same contrast without trials failing core validation (contract 22.11).
        valid = [s for s in seeds if all(core_valid(rows[name].get(s)) for name in comparison["weights"])]
        values = [sum(weight * metric(rows[name][s], key) for name, weight in comparison["weights"].items()) for s in valid]
        row["without_core_validation_failures"] = stat(values) if len(values) >= 2 else None
        out[comparison["name"]] = row
    return out


def assured_scenarios(rows: dict) -> list[str]:
    return [name for name, by_seed in rows.items() if any(complete(r) for r in by_seed.values())
            and name not in ("B", "I", "decay0_B", "decay0_I")]


def predictions(cells: dict, arrays: Arrays, rows: dict, seeds: list[int]) -> dict:
    out = {}
    records = lambda name: [rows[name][s] for s in seeds if complete(rows[name].get(s))]  # noqa: E731
    # T1: no liability, no quality and no corrections.
    zero = [arrays.get(r) for r in records("ALL_R0")]
    out["T1"] = {"seeds": len(zero), "verdict": "成立" if zero and all(
        not d["assurer_quality"].any() and not d["assurance_corrections"].any() for d in zero) else "不成立"}
    # T2: nobody buys voluntary assurance, and the run is the baseline run.
    equal = buyers = 0
    for seed in seeds:
        voluntary, base = rows["VOL16"].get(seed), rows["B"].get(seed)
        if not (complete(voluntary) and complete(base)):
            continue
        v, b = arrays.get(voluntary), arrays.get(base)
        buyers += int((v["assurer_assignment"] >= 0).sum())
        equal += all(np.array_equal(v[name], b[name]) for name in ("actions", "amounts", "inflation")) and (
            metric(voluntary, "true_change.equal_weight_sd") == metric(base, "true_change.equal_weight_sd"))
    out["T2"] = {"buyer_firm_periods": buyers, "seeds_equal_to_B": equal,
                 "verdict": "成立" if buyers == 0 and equal == len(seeds) else "不成立"}
    # T3: firms whose provider's quality reaches the stopping level do not inflate.
    high = total = high_at_cap = 0
    exceptions = []
    for name in assured_scenarios(rows):
        for record in records(name):
            data = arrays.get(record)
            if "assurer_assignment" not in data:
                continue
            wash = data["actions"][WINDOW] == INFLATE
            strict = firm_quality(data)[WINDOW] >= STOP_QUALITY
            total += int(wash.sum())
            high += int((wash & strict).sum())
            high_at_cap += int((wash & strict & (data["amounts"][WINDOW] >= CAP - 1e-9)).sum())
            if (wash & strict).any():
                exceptions.append({"scenario": name, "seed": record["seed"], "decisions": int((wash & strict).sum())})
    share = high / total if total else None
    out["T3"] = {"stop_quality": STOP_QUALITY, "wash_decisions": total, "wash_decisions_at_or_above": high,
                 "at_or_above_with_amount_at_period_cap": high_at_cap, "share_at_or_above": share,
                 "exceptions": exceptions, "verdict": None if share is None else ("成立" if share <= 0.01 else "不成立")}
    # T4: statutory liability raises washing among firms without an open case.
    row = paired(cells, "ALL_R1", "B", "wash_share_without_open_case", seeds)
    out["T4"] = {"ALL_R1_minus_B": row, "mean_client_quality_ALL_R1": float(np.mean(
        [cells["ALL_R1"][s]["mean_client_quality"] for s in cells["ALL_R1"]])) if cells.get("ALL_R1") else None,
                 "verdict": None if row is None else ("成立" if row["ci_low"] > 0 else "相反" if row["ci_high"] < 0 else "不确定")}
    # T5: when firms pick, washing concentrates at the laxest provider.
    lax = washing = 0
    for record in records("ALL_F16"):
        data = arrays.get(record)
        wash = (data["actions"][WINDOW] == INFLATE) & (data["assurer_assignment"][WINDOW] >= 0)
        lowest = firm_quality(data)[WINDOW] == data["assurer_quality"][WINDOW].min(axis=1, keepdims=True)
        washing += int(wash.sum())
        lax += int((wash & lowest).sum())
    share = lax / washing if washing else None
    out["T5"] = {"wash_decisions_by_clients": washing, "at_laxest_provider": lax, "share": share,
                 "verdict": None if share is None else ("成立" if share >= 0.9 else "不成立")}
    # T6: under 16x liability quality cycles around the stopping level and washing persists.
    starts = list(range(FIRST, LAST + 1, YEAR))
    last_four = slice(starts[-4], LAST + 1)
    cycling = persisting = 0
    series = {}
    for record in records("ALL_R16"):
        data = arrays.get(record)
        means = [float(data["assurer_quality"][s].mean()) for s in starts]
        series[record["seed"]] = [round(m, 4) for m in means]
        above = [i for i, m in enumerate(means) if m >= STOP_QUALITY]
        cycling += bool(above) and any(m < STOP_QUALITY for m in means[above[0] + 1:])
        persisting += bool((data["actions"][last_four] == INFLATE).any())
    n = len(series)
    out["T6"] = {"seeds": n, "seeds_cycling": cycling, "seeds_washing_in_last_four_years": persisting,
                 "yearly_mean_quality_by_seed": series,
                 "verdict": "成立" if n and cycling >= n / 2 and persisting >= n / 2 else "不成立"}
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


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def tables(result: dict) -> list[str]:
    out = ["# 鉴证机构表格", "", f"experiment_id `{result['experiment_id']}`", "",
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
            "| 场景 | 真实变化 | 漂绿决策 | 大 / 中 / 小 | 无案件企业的漂绿 | 未查出案件 | 被鉴证 | 带虚报的检查/期 | 更正/期 "
            "| 平均质量 | 客户面对的质量 | 责任罚款次数 | 年换鉴证方 | 罚款（初始市值比例） |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["describe"].items():
        sizes = " / ".join(pct(row.get(f"wash_share_{size}")) for size in SIZES)
        out.append(f"| {name} | {fmt(row['true_change'])} | {pct(row['wash_share'])} | {sizes} | "
                   f"{pct(row.get('wash_share_without_open_case'))} | {row['open_cases_per_decision_day']:.1f} | "
                   f"{num(row.get('checked_firms'), 1)} | {num(row.get('checks_with_inflation'))} | {num(row.get('corrections'))} | "
                   f"{num(row.get('mean_quality'), 3)} | {num(row.get('mean_client_quality'), 3)} | "
                   f"{num(row.get('liability_events_total'), 1)} | {num(row.get('switches_per_year'), 1)} | "
                   f"{row['fines_fraction_initial_cap']:.2e} |")
    p = result["predictions"]
    t3, t4, t5, t6 = p["T3"], p["T4"], p["T5"], p["T6"]
    out += ["", "## 解析预测", "",
            f"- T1：ALL_R0 质量全为 0、没有更正（{p['T1']['seeds']} 个种子）：{p['T1']['verdict']}。",
            f"- T2：VOL16 购买的企业—期 {p['T2']['buyer_firm_periods']} 个，与 B 逐位相同的种子 {p['T2']['seeds_equal_to_B']} 个：{p['T2']['verdict']}。",
            f"- T3：鉴证方质量 ≥ {t3['stop_quality']:.4f} 的企业的漂绿决策 {t3['wash_decisions_at_or_above']} 个，占全部 "
            f"{t3['wash_decisions']} 个的 {pct(t3['share_at_or_above'])}（其中幅度顶到每期上限 {t3['at_or_above_with_amount_at_period_cap']} 个）：{t3['verdict']}。",
            f"- T4：ALL_R1 − B，没有未查出虚报的企业中选择漂绿的比例 {interval(t4['ALL_R1_minus_B'])}；"
            f"ALL_R1 客户面对的平均质量 {num(t4['mean_client_quality_ALL_R1'], 4)}：{t4['verdict']}。",
            f"- T5：ALL_F16 被鉴证企业的漂绿决策中，来自当年质量最低鉴证方客户的占 {pct(t5['share'])}"
            f"（{t5['at_laxest_provider']}/{t5['wash_decisions_by_clients']}）：{t5['verdict']}。",
            f"- T6：ALL_R16 质量先过 {STOP_QUALITY:.4f} 后又回落的种子 {t6['seeds_cycling']}/{t6['seeds']}，最后 4 个鉴证年仍有漂绿的种子 "
            f"{t6['seeds_washing_in_last_four_years']}/{t6['seeds']}：{t6['verdict']}。"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory name next to this script, e.g. assurance_v1")
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
