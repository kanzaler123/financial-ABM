"""Stage H3 analysis (research contract section 25.8). Read-only on the archive.

  python analyze.py h3_v1            # confirmatory: claim records, prediction, auxiliaries, descriptives, tables
  python analyze.py check_v1 --smoke # development: runs the same code, writes and prints counts only

The eight primary questions P1-P8 are those of the spec's "primary_questions": P1, P2, P4, P6 and P7 are paired
differences of runner metrics, P3, P5 and P8 paired differences of washing-decision shares computed here. Each is a
two-sided paired t test across seeds, and the eight are Holm-adjusted together (contract 25.8); everything else is
a paired 95% t interval without adjustment. A share is the number of washing decisions of the given kind over all
washing decisions on the 38 decision days 1260-3480; a seed whose share is undefined (no washing decision) in
either scenario is left out of that contrast and counted. Judgments follow contract 22.8.
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

PERIOD, FIRST, LAST, CAP = 60, 21, 58, 0.25
WINDOW = slice(FIRST, LAST + 1)   # decision periods on days 1260..3480
INFLATE = 1
SIZES = ("large", "mid", "small")
MAIN_METRIC = "true_change.equal_weight_sd"
AUX_METRICS = ("true_change.equal_weight_sd", "true_change.fixed_start_cap_weight_sd", "size.large_minus_small_sd")
DESIGN = json.loads((HERE / "design_numbers.json").read_text(encoding="utf-8"))


# -- loading ------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict, dict]:
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["experiment_id"] != frozen["experiment_id"]:
        raise ValueError("summary and experiment manifest disagree")
    rows: dict = {}
    for record in summary["records"]:
        rows.setdefault(record["scenario"], {})[record["seed"]] = record
    return frozen, summary, rows


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

    NAMES = ("actions", "amounts", "inflation", "market_size_class", "market_price_cap_hits",
             "assurer_assignment", "assurer_quality")

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


def ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


# -- per-trial quantities -----------------------------------------------------------

def trial(record: dict, data: dict) -> dict:
    """Washing-decision counts and shares of one trial over the policy window."""
    wash = data["actions"][WINDOW] == INFLATE
    size = data["market_size_class"]
    total = int(wash.sum())
    out = {"decisions": int(wash.size), "wash_decisions": total, "wash_share": float(wash.mean())}
    for index, name in enumerate(SIZES):
        members = size == index
        count = int(wash[:, members].sum())
        out[f"wash_decisions_{name}"] = count
        # P3, P5: this class's washing decisions over all washing decisions.
        out[f"share_of_washing_{name}"] = ratio(count, total)
        # The class's own washing rate (the abstract's "two thirds" for large firms under B).
        out[f"wash_rate_{name}"] = float(wash[:, members].mean()) if members.any() else None
    # Inflation recorded on a decision day is after that day's audit, i.e. what the firm holds when deciding.
    holding = data["inflation"][::PERIOD][WINDOW] > 0
    out["continuing_wash_share"] = ratio(int((wash & holding).sum()), total)
    out["wash_at_period_cap"] = int((wash & (data["amounts"][WINDOW] >= CAP - 1e-9)).sum())
    out["share_of_washing_laxest"] = None
    if "assurer_assignment" in data:
        assignment = data["assurer_assignment"][WINDOW]
        quality = data["assurer_quality"][WINDOW]
        assured = wash & (assignment >= 0)
        client_quality = np.take_along_axis(quality, np.maximum(assignment, 0), axis=1)
        # P8: clients of the provider with that year's lowest quality (ties count as lowest).
        laxest = (assignment >= 0) & (client_quality == quality.min(axis=1, keepdims=True))
        out["assured_wash_decisions"] = int(assured.sum())
        out["assured_wash_at_laxest"] = int((assured & laxest).sum())
        out["share_of_washing_laxest"] = ratio(out["assured_wash_at_laxest"], out["assured_wash_decisions"])
    return out


def first_cap_day(data: dict) -> int | None:
    hits = np.flatnonzero(data["market_price_cap_hits"])
    return int(hits[0]) if hits.size else None


def contrast(weights: dict, value, seeds: list[int], valid_only: bool = False) -> tuple[list[float], list[int]]:
    """Per-seed weighted contrast; seeds where any term is undefined are left out and returned."""
    values, left_out = [], []
    for seed in seeds:
        terms = [value(name, seed) for name in weights]
        if any(term is None for term in terms) or (valid_only and not all(core_valid_cell(name, seed) for name in weights)):
            left_out.append(seed)
            continue
        values.append(float(sum(weight * term for weight, term in zip(weights.values(), terms))))
    return values, left_out


# Set in main(): scenario -> seed -> record / per-trial quantities.
ROWS: dict = {}
CELLS: dict = {}


def core_valid_cell(name: str, seed: int) -> bool:
    return core_valid(ROWS.get(name, {}).get(seed))


def runner_value(key: str):
    def value(name: str, seed: int) -> float | None:
        record = ROWS.get(name, {}).get(seed)
        return metric(record, key) if complete(record) else None
    return value


def share_value(key: str):
    def value(name: str, seed: int) -> float | None:
        return CELLS.get(name, {}).get(seed, {}).get(key)
    return value


# -- primary questions --------------------------------------------------------------

def claim(row: dict, direction: int) -> str:
    holm, mean = row.get("holm_p_value"), row.get("mean")
    if holm is None or holm >= 0.05:
        return "unresolved"
    return "supported" if np.sign(mean) == direction else "unsupported"


def primary_questions(questions: list[dict], seeds: list[int]) -> dict:
    out = {}
    for q in questions:
        value = runner_value(q["metric"]) if q["kind"] == "runner_metric" else share_value(f"share_of_washing_{q['share']}")
        values, left_out = contrast(q["weights"], value, seeds)
        valid, _ = contrast(q["weights"], value, seeds, valid_only=True)
        row = {"id": q["id"], "comparison": q["comparison"], "kind": q["kind"], "direction": q["direction"],
               "measure": q.get("metric", q.get("share")), "seeds_left_out": left_out,
               **(stat(values) or {"n": len(values)}), "without_core_validation_failures": stat(valid)}
        out[q["id"]] = row
    rows = list(out.values())
    if all("p_value" in row for row in rows):
        adjusted = 0.0
        for rank, row in enumerate(sorted(rows, key=lambda row: row["p_value"])):
            adjusted = max(adjusted, min(1.0, (len(rows) - rank) * row["p_value"]))
            row["holm_p_value"] = adjusted
    else:
        for row in rows:
            row["holm_p_value"] = None
    for q in questions:
        out[q["id"]]["claim"] = claim(out[q["id"]], q["direction"])
    return out


# -- prediction, auxiliaries, descriptives --------------------------------------------

def paired(plus: str, minus: str, key: str, seeds: list[int]) -> dict | None:
    return stat(contrast({plus: 1, minus: -1}, share_value(key), seeds)[0])


def prediction(seeds: list[int]) -> dict:
    """T1 of contract 24.6 at the estimate: washing-decision share I - F and F - B below 0 (main block)."""
    out = {"design": {"main_Q": DESIGN["main_Q"], "perm": DESIGN["perm"]}}
    for prefix in ("", "perm_"):
        rows = {f"{prefix}I_minus_{prefix}F": paired(prefix + "I", prefix + "F", "wash_share", seeds),
                f"{prefix}F_minus_{prefix}B": paired(prefix + "F", prefix + "B", "wash_share", seeds)}
        out[prefix or "main"] = rows
    out["I_wash_at_period_cap"] = sum(cell["wash_at_period_cap"] for cell in CELLS.get("I", {}).values())
    out["I_wash_decisions"] = sum(cell["wash_decisions"] for cell in CELLS.get("I", {}).values())
    main = out["main"].values()
    out["verdict"] = None if any(row is None for row in main) else (
        "成立" if all(row["ci_high"] < 0 for row in main) else "不成立")
    return out


def auxiliary(summary: dict, questions: list[dict], seeds: list[int]) -> dict:
    primary = {q["comparison"] for q in questions}
    runner = {}
    for comparison in summary["comparisons"]:
        if comparison["name"] in primary:
            continue
        runner[comparison["name"]] = {
            key: {field: comparison["metrics"][key].get(field) for field in ("mean", "ci_low", "ci_high", "n")}
            for key in AUX_METRICS}
    # Market-cap weighted versions of the three core contrasts and the size gap of P6's contrast on the main metric.
    for name, weights in (("P1_I_minus_F", {"I": 1, "F": -1}), ("P2_F_minus_B", {"F": 1, "B": -1}),
                          ("I_minus_B", {"I": 1, "B": -1})):
        runner.setdefault(name, {})["true_change.fixed_start_cap_weight_sd"] = stat(
            contrast(weights, runner_value("true_change.fixed_start_cap_weight_sd"), seeds)[0])
    shares = {}
    for plus, minus in (("F", "F_2x"), ("F_8x", "F"), ("F_2x", "B"), ("F_8x", "B")):
        shares[f"{plus}_minus_{minus}"] = {key: paired(plus, minus, key, seeds)
                                           for key in ("wash_share", "share_of_washing_large")}
    for name in ("B", "I", "F"):
        shares[f"{name}_minus_u0p2_{name}"] = {key: paired(name, f"u0p2_{name}", key, seeds)
                                               for key in ("wash_share", "share_of_washing_large")}
    return {"runner": runner, "shares": shares}


def describe(seeds: list[int], arrays: Arrays) -> dict:
    out = {}
    for name, by_seed in CELLS.items():
        keys = sorted({key for values in by_seed.values() for key, value in values.items()
                       if isinstance(value, (int, float))})
        row = {}
        for key in keys:
            values = [by_seed[s][key] for s in seeds if s in by_seed and by_seed[s].get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        row["true_change"] = float(np.mean([metric(ROWS[name][s], MAIN_METRIC) for s in seeds if complete(ROWS[name].get(s))]))
        row["seeds_without_washing"] = sum(by_seed[s]["wash_decisions"] == 0 for s in by_seed)
        row["complete"] = len(by_seed)
        row["core_validation_failures"] = [
            {"seed": s, "first_cap_day": first_cap_day(arrays.get(ROWS[name][s]))}
            for s in seeds if complete(ROWS[name].get(s)) and not core_valid(ROWS[name][s])]
        out[name] = row
    return out


# -- tables --------------------------------------------------------------------------

def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−")


def interval(s: dict | None, digits: int = 3) -> str:
    if s is None or s.get("mean") is None:
        return "—"
    return f"{fmt(s['mean'], digits)} [{fmt(s['ci_low'], digits)}, {fmt(s['ci_high'], digits)}]"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def tables(result: dict) -> list[str]:
    out = ["# 阶段 H3 表格", "", f"experiment_id `{result['experiment_id']}`", "",
           "## 主要问题（政策窗口，配对 95% t 区间，八项一起 Holm 校正）", "",
           "| 问题 | 对比 | 指标 | 均值 [区间] | Holm p | 判定 | 剔除的种子 | 剔除核心验证失败后 |",
           "|---|---|---|---|---:|---|---:|---|"]
    for row in result["primary"].values():
        holm = row["holm_p_value"]
        digits = 4 if row["kind"] == "wash_decision_share" else 3
        out.append(f"| {row['id']} | {row['comparison']} | `{row['measure']}` | {interval(row, digits)} | "
                   f"{'—' if holm is None else f'{holm:.3g}'} | {row['claim']} | {len(row['seeds_left_out'])} | "
                   f"{interval(row['without_core_validation_failures'], digits)} |")
    p = result["prediction"]
    out += ["", "## 解析预测 T1（描述性，不进 Holm 族）", "",
            f"- 主块：漂绿决策占比 I − F {interval(p['main']['I_minus_F'], 4)}，F − B {interval(p['main']['F_minus_B'], 4)}："
            f"{p['verdict']}。I 中幅度顶到每期上限的漂绿决策 {p['I_wash_at_period_cap']}/{p['I_wash_decisions']}。",
            f"- perm_ 块（描述）：I − F {interval(p['perm_']['perm_I_minus_perm_F'], 4)}，"
            f"F − B {interval(p['perm_']['perm_F_minus_perm_B'], 4)}。"]
    out += ["", "## 辅助对比（未校正）", "", "| 对比 | 等权真实变化 | 市值加权真实变化 | 大减小 |", "|---|---|---|---|"]
    for name, row in result["auxiliary"]["runner"].items():
        out.append(f"| {name} | " + " | ".join(interval(row.get(key)) for key in AUX_METRICS) + " |")
    out += ["", "| 对比 | 漂绿决策占比 | 大企业在漂绿中的份额 |", "|---|---|---|"]
    for name, row in result["auxiliary"]["shares"].items():
        out.append(f"| {name} | {interval(row['wash_share'], 4)} | {interval(row['share_of_washing_large'], 4)} |")
    out += ["", "## 描述（种子均值）", "",
            "| 场景 | 真实变化 | 漂绿决策 | 大 / 中 / 小 漂绿率 | 漂绿份额 大 / 小 | 继续已有案件 | 最宽松鉴证方份额 | 无漂绿种子 |",
            "|---|---:|---:|---|---|---:|---:|---:|"]
    for name, row in result["describe"].items():
        rates = " / ".join(pct(row.get(f"wash_rate_{size}")) for size in SIZES)
        shares = " / ".join(pct(row.get(f"share_of_washing_{size}")) for size in ("large", "small"))
        out.append(f"| {name} | {fmt(row['true_change'])} | {pct(row['wash_share'])} | {rates} | {shares} | "
                   f"{pct(row.get('continuing_wash_share'))} | {pct(row.get('share_of_washing_laxest'))} | "
                   f"{row['seeds_without_washing']} |")
    return out


def counts(result: dict, frozen: dict) -> dict:
    """What --smoke writes: counts only, no estimate."""
    return {"experiment_id": result["experiment_id"], "purpose": frozen["spec"]["purpose"],
            "complete_trials": result["complete_trials"], "planned_trials": result["planned_trials"],
            "trials_failing_core_validation": result["trials_failing_core_validation"],
            "primary_questions": {key: {"n": row["n"], "seeds_left_out": len(row["seeds_left_out"]),
                                        "holm_computed": row["holm_p_value"] is not None}
                                  for key, row in result["primary"].items()},
            "auxiliary_runner_comparisons": len(result["auxiliary"]["runner"]),
            "auxiliary_share_contrasts": len(result["auxiliary"]["shares"]),
            "described_scenarios": len(result["describe"]),
            "assured_scenarios_with_laxest_share": sorted(
                name for name, by_seed in CELLS.items() if any(c["share_of_washing_laxest"] is not None for c in by_seed.values())),
            "table_lines": len(tables(result))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory name next to this script, e.g. h3_v1")
    parser.add_argument("--smoke", action="store_true", help="development archive: write and print counts only")
    args = parser.parse_args()
    run = HERE / args.run
    frozen, summary, rows = load(run)
    if frozen["spec"]["purpose"] == "development" and not args.smoke:
        raise SystemExit("development archives are analyzed with --smoke only")
    seeds, questions = frozen["spec"]["seeds"], frozen["spec"]["primary_questions"]
    arrays = Arrays(run)
    ROWS.update(rows)
    CELLS.update({name: {seed: trial(record, arrays.get(record)) for seed, record in by_seed.items() if complete(record)}
                  for name, by_seed in rows.items()})
    result = {"experiment_id": frozen["experiment_id"], "purpose": frozen["spec"]["purpose"],
              "complete_trials": summary["complete_trials"], "planned_trials": summary["planned_trials"],
              "trials_failing_core_validation": summary["trials_failing_core_validation"],
              "primary": primary_questions(questions, seeds), "prediction": prediction(seeds),
              "auxiliary": auxiliary(summary, questions, seeds), "describe": describe(seeds, arrays)}
    if args.smoke:
        out = counts(result, frozen)
        (HERE / f"{args.run}_smoke.json").write_bytes((json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    (HERE / f"{args.run}_analysis.json").write_bytes(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    (HERE / f"{args.run}_tables.md").write_bytes(("\n".join(tables(result)) + "\n").encode("utf-8"))
    print(f"wrote {args.run}_analysis.json and {args.run}_tables.md")


if __name__ == "__main__":
    main()
