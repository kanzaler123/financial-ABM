"""Stage I analysis (research contract sections 26.9 and 26.20). Read-only on the archive.

  python analyze.py stage_i_v2            # confirmatory: claim records, sensitivity curve, descriptives, tables
  python analyze.py check_v2 --smoke      # development: runs the same code, writes and prints counts only

v2 (contract 26.20): the points are rating slopes lambda instead of spreads s. The confirmatory block is at
lambda*; the sensitivity points are lambda0 = 0, lambda_lo, lambda_hi and lambda_FE. The monotonicity reading uses
lambda0, lambda_lo, lambda* and lambda_hi (26.20 (g)); lambda_FE, an exploratory point added after the estimate, is
shown on the curve and enters only a second, exploratory reading over all five points. Below, "spread" stands for
whichever the spec varies (v1: s, v2: lambda).

Primary questions (when the spec has a confirmatory block at s*): the eight of contract 25.8, as frozen in the H3
spec, on the unprefixed scenarios. P1, P2, P4, P6 and P7 are paired differences of runner metrics, P3, P5 and P8
paired differences of washing-decision shares computed here; each is a two-sided paired t test across seeds and the
eight are Holm-adjusted together. Judgments follow contract 22.8. A share is the number of washing decisions of the
given kind over all washing decisions on the 38 decision days 1260-3480; a seed whose share is undefined (no washing
decision) in either scenario is left out of that contrast and counted.

Sensitivity (contract 26.9, exploratory, no adjustment): the curve runs over the spreads s0, s_lo, s*, s_hi and s_28
that exist, in increasing order (s* from the confirmatory block's core scenarios). At each spread: the eight contrasts
with paired 95% t intervals, every core scenario's washing-decision share and the size classes' shares of washing
decisions. Reading: for each contrast the change of the point estimate between adjacent spreads; all changes of one
sign is "monotone", otherwise the segments that break it are named (contract 20.16, judgment 2). Each change also
gets a paired 95% t interval (the same seed's contrast at the two spreads). A spread within 10% of s* (s_28 here,
0.15% away) is the same calibration point (contract 26.18): it is reported against s* seed by seed and stays out of
the reading.
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

H3_SPEC = ROOT / "configs" / "disclosure_stage_h3_v1.json"
PERIOD, FIRST, LAST, CAP = 60, 21, 58, 0.25
WINDOW = slice(FIRST, LAST + 1)   # decision periods on days 1260..3480
INFLATE = 1
SIZES = ("large", "mid", "small")
MAIN_METRIC = "true_change.equal_weight_sd"
AUX_METRICS = ("true_change.equal_weight_sd", "true_change.fixed_start_cap_weight_sd", "size.large_minus_small_sd")
PREFIXES = {"s0": "s0_", "s_lo": "slo_", "s_hi": "shi_", "s_28": "s28_"}
STAR, POINTS_KEY, SYMBOL = "s_star", "financing_spreads", "s"
EXPLORATORY: set[str] = set()     # points outside the pre-registered reading (v2: lambda_fe)
CORE = ("B", "I", "F", "J", "J_hidden", "V", "ALL_R16", "ALL_F16")
REPLICATE_TOLERANCE = 0.10    # contract 26.18: within 10% of s* (the calibration tolerance) is the same point
QUESTIONS = json.loads(H3_SPEC.read_text(encoding="utf-8"))["primary_questions"]   # contract 25.8, frozen in H3


# -- loading ------------------------------------------------------------------

def configure(spec: dict) -> None:
    """v2 specs vary financing_rating_slope (contract 26.20); v1 specs vary financing_spread (26.9)."""
    global PREFIXES, STAR, POINTS_KEY, SYMBOL, EXPLORATORY
    if "financing_rating_slopes" in spec:
        PREFIXES = {"lambda0": "l0_", "lambda_lo": "llo_", "lambda_hi": "lhi_", "lambda_fe": "lfe_"}
        STAR, POINTS_KEY, SYMBOL, EXPLORATORY = "lambda_star", "financing_rating_slopes", "λ", {"lambda_fe"}


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

def trial(data: dict) -> dict:
    """Washing-decision counts and shares of one trial over the policy window (as H3's analyze.py)."""
    wash = data["actions"][WINDOW] == INFLATE
    size = data["market_size_class"]
    total = int(wash.sum())
    out = {"decisions": int(wash.size), "wash_decisions": total, "wash_share": float(wash.mean())}
    for index, name in enumerate(SIZES):
        members = size == index
        count = int(wash[:, members].sum())
        out[f"wash_decisions_{name}"] = count
        out[f"share_of_washing_{name}"] = ratio(count, total)              # P3 (large), P5 (small)
        out[f"wash_rate_{name}"] = float(wash[:, members].mean()) if members.any() else None
    holding = data["inflation"][::PERIOD][WINDOW] > 0
    out["continuing_wash_share"] = ratio(int((wash & holding).sum()), total)
    out["wash_at_period_cap"] = int((wash & (data["amounts"][WINDOW] >= CAP - 1e-9)).sum())
    out["share_of_washing_laxest"] = None
    if "assurer_assignment" in data:
        assignment = data["assurer_assignment"][WINDOW]
        quality = data["assurer_quality"][WINDOW]
        assured = wash & (assignment >= 0)
        client_quality = np.take_along_axis(quality, np.maximum(assignment, 0), axis=1)
        laxest = (assignment >= 0) & (client_quality == quality.min(axis=1, keepdims=True))   # P8, ties lowest
        out["assured_wash_decisions"] = int(assured.sum())
        out["assured_wash_at_laxest"] = int((assured & laxest).sum())
        out["share_of_washing_laxest"] = ratio(out["assured_wash_at_laxest"], out["assured_wash_decisions"])
    return out


def first_cap_day(data: dict) -> int | None:
    hits = np.flatnonzero(data["market_price_cap_hits"])
    return int(hits[0]) if hits.size else None


# Set in main(): scenario -> seed -> record / per-trial quantities.
ROWS: dict = {}
CELLS: dict = {}


def value_of(q: dict):
    """Per-seed value of a question's measure in one scenario (None when undefined)."""
    if q["kind"] == "runner_metric":
        def value(name: str, seed: int) -> float | None:
            record = ROWS.get(name, {}).get(seed)
            return metric(record, q["metric"]) if complete(record) else None
    else:
        def value(name: str, seed: int) -> float | None:
            return CELLS.get(name, {}).get(seed, {}).get(f"share_of_washing_{q['share']}")
    return value


def per_seed(q: dict, prefix: str, seeds: list[int], valid_only: bool = False) -> dict[int, float]:
    """The question's contrast in the block with this prefix, seed by seed; undefined seeds are absent."""
    value, out = value_of(q), {}
    for seed in seeds:
        names = {prefix + name: weight for name, weight in q["weights"].items()}
        terms = [value(name, seed) for name in names]
        if any(term is None for term in terms):
            continue
        if valid_only and not all(core_valid(ROWS.get(name, {}).get(seed)) for name in names):
            continue
        out[seed] = float(sum(weight * term for weight, term in zip(names.values(), terms)))
    return out


# -- primary questions --------------------------------------------------------------

def claim(row: dict, direction: int) -> str:
    holm, mean = row.get("holm_p_value"), row.get("mean")
    if holm is None or holm >= 0.05:
        return "unresolved"
    return "supported" if np.sign(mean) == direction else "unsupported"


def primary_questions(questions: list[dict], seeds: list[int]) -> dict:
    out = {}
    for q in questions:
        values = per_seed(q, "", seeds)
        valid = per_seed(q, "", seeds, valid_only=True)
        out[q["id"]] = {"id": q["id"], "comparison": q["comparison"], "kind": q["kind"], "direction": q["direction"],
                        "measure": q.get("metric", q.get("share")),
                        "seeds_left_out": [s for s in seeds if s not in values],
                        **(stat(list(values.values())) or {"n": len(values)}),
                        "without_core_validation_failures": stat(list(valid.values()))}
    rows = list(out.values())
    if rows and all("p_value" in row for row in rows):
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


# -- sensitivity curve --------------------------------------------------------------

def curve_points(spreads: dict, names: set[str]) -> list[tuple[str, float, str]]:
    """(label, spread, prefix) of every spread whose core scenarios are in the archive, by increasing spread."""
    points = [(key, spreads[key], prefix) for key, prefix in PREFIXES.items()
              if spreads.get(key) is not None and all(prefix + n in names for n in CORE)]
    if spreads.get(STAR) is not None and all(n in names for n in CORE):
        points.append((STAR, spreads[STAR], ""))
    return sorted(points, key=lambda p: p[1])


def read_segments(points: list[tuple[str, float, str]], values: dict, at: dict, seeds: list[int]) -> dict:
    """Changes between adjacent points and the monotonicity reading (contract 20.16, judgment 2)."""
    segments, signs = [], []
    for (a, sa, _), (b, sb, _) in zip(points, points[1:]):
        paired = [values[b][s] - values[a][s] for s in seeds if s in values[a] and s in values[b]]
        change = (at[b].get("mean") - at[a].get("mean")) if at[a].get("mean") is not None and at[b].get("mean") is not None else None
        signs.append(None if change is None else int(np.sign(change)))
        segments.append({"from": a, "to": b, "from_spread": sa, "to_spread": sb, "point_estimate_change": change,
                         "paired_change": stat(paired)})
    known = [s for s in signs if s is not None]
    if len(known) < len(signs) or not known:
        reading = "undetermined"
    elif all(s == known[0] != 0 for s in known):
        reading = "monotone increasing" if known[0] > 0 else "monotone decreasing"
    else:
        reading = "not monotone"
    breaking = [] if reading.startswith("monotone") else [
        f"{seg['from']}->{seg['to']}" for seg, s in zip(segments, signs) if s != (known[0] if known else None)]
    return {"segments": segments, "segment_signs": signs, "reading": reading, "segments_against_first": breaking,
            "segments_all_distinguishable": all(
                seg["paired_change"] is not None and
                (seg["paired_change"]["ci_low"] > 0 or seg["paired_change"]["ci_high"] < 0)
                for seg in segments) if segments else None}


def sensitivity(spreads: dict, seeds: list[int]) -> dict:
    """Contract 26.9 and 26.18: a spread within REPLICATE_TOLERANCE of s* is the same calibration point; it is
    reported against s* seed by seed and stays out of the monotonicity reading. Contract 26.20 (g): exploratory
    points (lambda_FE) stay out of the reading too and enter a second, exploratory reading over all points."""
    every = curve_points(spreads, set(ROWS))
    star = next((p for p in every if p[0] == STAR), None)
    replicates = [p for p in every if star and p[0] != STAR and abs(p[1] / star[1] - 1) < REPLICATE_TOLERANCE]
    exploratory = [p for p in every if p[0] in EXPLORATORY]
    points = [p for p in every if p not in replicates and p not in exploratory]
    contrasts = {}
    for q in QUESTIONS:
        values = {label: per_seed(q, prefix, seeds) for label, _, prefix in every}
        at = {label: {"spread": spread, **(stat(list(values[label].values())) or {"n": len(values[label])}),
                      "seeds_left_out": [s for s in seeds if s not in values[label]]}
              for label, spread, _ in every}
        replicate = {label: stat([values[label][s] - values[STAR][s] for s in seeds
                                  if s in values[label] and s in values[STAR]]) for label, _, _ in replicates}
        contrasts[q["id"]] = {"comparison": q["comparison"], "measure": q.get("metric", q.get("share")),
                              "direction": q["direction"], "at": at, **read_segments(points, values, at, seeds),
                              "replicate_minus_s_star": replicate}
        if exploratory:
            contrasts[q["id"]]["with_exploratory_points"] = read_segments(
                [p for p in every if p not in replicates], values, at, seeds)
    shares = {}
    for label, spread, prefix in every:
        shares[label] = {"spread": spread, **{
            name: {key: mean_cell(prefix + name, key, seeds) for key in
                   ("wash_share", "share_of_washing_large", "share_of_washing_mid", "share_of_washing_small",
                    "wash_rate_large", "wash_rate_small")} for name in CORE}}
    return {"points": [{"label": label, "spread": spread, "prefix": prefix} for label, spread, prefix in points],
            "replicates": [{"label": label, "spread": spread, "prefix": prefix} for label, spread, prefix in replicates],
            "exploratory": [{"label": label, "spread": spread, "prefix": prefix} for label, spread, prefix in exploratory],
            "contrasts": contrasts, "washing_shares": shares}


def mean_cell(name: str, key: str, seeds: list[int]) -> float | None:
    values = [CELLS[name][s][key] for s in seeds if s in CELLS.get(name, {}) and CELLS[name][s].get(key) is not None]
    return float(np.mean(values)) if values else None


# -- auxiliaries and descriptives --------------------------------------------------------

def auxiliary(summary: dict, questions: list[dict]) -> dict:
    """Runner comparisons that are not primary questions, unadjusted (s* block and sensitivity blocks)."""
    primary = {q["comparison"] for q in questions}
    return {c["name"]: {key: {field: c["metrics"][key].get(field) for field in ("mean", "ci_low", "ci_high", "n")}
                        for key in AUX_METRICS}
            for c in summary["comparisons"] if c["name"] not in primary}


def describe(seeds: list[int], arrays: Arrays) -> dict:
    out = {}
    for name, by_seed in CELLS.items():
        keys = sorted({key for values in by_seed.values() for key, value in values.items()
                       if isinstance(value, (int, float))})
        row = {key: mean_cell(name, key, seeds) for key in keys}
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


def point_text(value: float) -> str:
    return f"λ = {value * 1e4:g} bp" if SYMBOL == "λ" else f"s = {value:g}"


def tables(result: dict) -> list[str]:
    star = f"{SYMBOL}*"
    out = ["# 阶段 I 模型运行表格", "", f"experiment_id `{result['experiment_id']}`", "",
           f"{'债务利率对评级的直接反应 λ（年利率/SD）' if SYMBOL == 'λ' else '融资利差'}："
           f"{json.dumps(result['points'], ensure_ascii=False)}", ""]
    if result["primary"]:
        out += [f"## 主要问题（{star}，政策窗口，配对 95% t 区间，八项一起 Holm 校正）", "",
                "| 问题 | 对比 | 指标 | 均值 [区间] | Holm p | 判定 | 剔除的种子 | 剔除核心验证失败后 |",
                "|---|---|---|---|---:|---|---:|---|"]
        for row in result["primary"].values():
            holm = row["holm_p_value"]
            digits = 4 if row["kind"] == "wash_decision_share" else 3
            out.append(f"| {row['id']} | {row['comparison']} | `{row['measure']}` | {interval(row, digits)} | "
                       f"{'—' if holm is None else f'{holm:.3g}'} | {row['claim']} | {len(row['seeds_left_out'])} | "
                       f"{interval(row['without_core_validation_failures'], digits)} |")
    sens = result["sensitivity"]
    labels = [p["label"] for p in sens["points"]]
    shown = sorted(sens["points"] + sens["replicates"] + sens["exploratory"], key=lambda p: p["spread"])
    repeat = {p["label"] for p in sens["replicates"]}
    extra = {p["label"] for p in sens["exploratory"]}
    tag = lambda label: "（重复）" if label in repeat else "（探索性）" if label in extra else ""  # noqa: E731
    out += ["", f"## 敏感性（探索性，未校正）：各{'λ' if SYMBOL == 'λ' else '利差'}下的对比", "",
            "| 问题 | " + " | ".join(f"{p['label']}{tag(p['label'])} ({point_text(p['spread'])})" for p in shown)
            + " | 读法（事先写明的点） |" + (" 读法（含探索性点） |" if extra else ""),
            "|---|" + "---|" * len(shown) + "---|" + ("---|" if extra else "")]
    for key, row in sens["contrasts"].items():
        digits = 4 if row["measure"] in ("large", "small", "laxest") else 3
        reading = row["reading"] + (f"（{', '.join(row['segments_against_first'])}）" if row["segments_against_first"] else "")
        cells = f"| {key} | " + " | ".join(interval(row["at"][p["label"]], digits) for p in shown) + f" | {reading} |"
        if extra:
            other = row["with_exploratory_points"]
            cells += f" {other['reading']}" + (f"（{', '.join(other['segments_against_first'])}）"
                                               if other["segments_against_first"] else "") + " |"
        out.append(cells)
    if repeat:
        out += ["", f"重复点与 {star} 的逐种子配对差（第 26.18 条，不进入单调性判断）：", "",
                "| 问题 | " + " | ".join(f"{label} − {star}" for label in sorted(repeat)) + " |", "|---|" + "---|" * len(repeat)]
        for key, row in sens["contrasts"].items():
            digits = 4 if row["measure"] in ("large", "small", "laxest") else 3
            out.append(f"| {key} | " + " | ".join(interval(row["replicate_minus_s_star"][label], digits)
                                                  for label in sorted(repeat)) + " |")
    out += ["", "相邻两点之差（配对 95% t 区间，事先写明的点）：", "", "| 问题 | " + " | ".join(
        f"{a}→{b}" for a, b in zip(labels, labels[1:])) + " |", "|---|" + "---|" * max(len(labels) - 1, 0)]
    for key, row in sens["contrasts"].items():
        digits = 4 if row["measure"] in ("large", "small", "laxest") else 3
        out.append(f"| {key} | " + " | ".join(interval(seg["paired_change"], digits) for seg in row["segments"]) + " |")
    if extra:
        first = next(iter(sens["contrasts"].values()))["with_exploratory_points"]["segments"]
        out += ["", "相邻两点之差（含探索性点，第 26.20 条 (g)）：", "", "| 问题 | " + " | ".join(
            f"{seg['from']}→{seg['to']}" for seg in first) + " |", "|---|" + "---|" * len(first)]
        for key, row in sens["contrasts"].items():
            digits = 4 if row["measure"] in ("large", "small", "laxest") else 3
            out.append(f"| {key} | " + " | ".join(interval(seg["paired_change"], digits)
                                                  for seg in row["with_exploratory_points"]["segments"]) + " |")
    out += ["", "## 漂绿决策占比与按规模的份额（种子均值）", "",
            f"| {'λ' if SYMBOL == 'λ' else '利差'} | 场景 | 漂绿决策占比 | 份额 大 / 中 / 小 | 漂绿率 大 / 小 |", "|---|---|---:|---|---|"]
    for label, block in sens["washing_shares"].items():
        for name in CORE:
            row = block[name]
            out.append(f"| {label} | {name} | {pct(row['wash_share'])} | " + " / ".join(
                pct(row[f"share_of_washing_{s}"]) for s in SIZES) + f" | {pct(row['wash_rate_large'])} / {pct(row['wash_rate_small'])} |")
    out += ["", "## 辅助对比（未校正）", "", "| 对比 | 等权真实变化 | 市值加权真实变化 | 大减小 |", "|---|---|---|---|"]
    for name, row in result["auxiliary"].items():
        out.append(f"| {name} | " + " | ".join(interval(row.get(key)) for key in AUX_METRICS) + " |")
    out += ["", "## 描述（种子均值）", "",
            "| 场景 | 真实变化 | 漂绿决策 | 大 / 中 / 小 漂绿率 | 漂绿份额 大 / 小 | 继续已有案件 | 最宽松鉴证方份额 | 无漂绿种子 | 核心验证失败 |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for name, row in result["describe"].items():
        rates = " / ".join(pct(row.get(f"wash_rate_{size}")) for size in SIZES)
        shares = " / ".join(pct(row.get(f"share_of_washing_{size}")) for size in ("large", "small"))
        out.append(f"| {name} | {fmt(row['true_change'])} | {pct(row['wash_share'])} | {rates} | {shares} | "
                   f"{pct(row.get('continuing_wash_share'))} | {pct(row.get('share_of_washing_laxest'))} | "
                   f"{row['seeds_without_washing']} | {len(row['core_validation_failures'])} |")
    return out


def counts(result: dict, frozen: dict) -> dict:
    """What --smoke writes: counts only, no estimate."""
    sens = result["sensitivity"]
    return {"experiment_id": result["experiment_id"], "purpose": frozen["spec"]["purpose"],
            "complete_trials": result["complete_trials"], "planned_trials": result["planned_trials"],
            "trials_failing_core_validation": result["trials_failing_core_validation"],
            "primary_questions": {key: {"n": row["n"], "seeds_left_out": len(row["seeds_left_out"]),
                                        "holm_computed": row["holm_p_value"] is not None}
                                  for key, row in result["primary"].items()},
            "sensitivity_points": [p["label"] for p in sens["points"]],
            "sensitivity_replicates": [p["label"] for p in sens["replicates"]],
            "sensitivity_exploratory": [p["label"] for p in sens["exploratory"]],
            "sensitivity_contrasts": {key: {"n_by_point": {label: at["n"] for label, at in row["at"].items()},
                                            "segments": len(row["segments"]),
                                            "segments_with_paired_interval": sum(seg["paired_change"] is not None
                                                                                 for seg in row["segments"]),
                                            "reading_computed": row["reading"] != "undetermined"}
                                      for key, row in sens["contrasts"].items()},
            "auxiliary_runner_comparisons": len(result["auxiliary"]),
            "described_scenarios": len(result["describe"]),
            "assured_scenarios_with_laxest_share": sorted(
                name for name, by_seed in CELLS.items() if any(c["share_of_washing_laxest"] is not None for c in by_seed.values())),
            "table_lines": len(tables(result))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory name next to this script, e.g. stage_i_v2")
    parser.add_argument("--smoke", action="store_true", help="development archive: write and print counts only")
    args = parser.parse_args()
    run = HERE / args.run
    frozen, summary, rows = load(run)
    if frozen["spec"]["purpose"] == "development" and not args.smoke:
        raise SystemExit("development archives are analyzed with --smoke only")
    spec = frozen["spec"]
    configure(spec)
    seeds, questions = spec["seeds"], spec["primary_questions"]
    if questions and questions != QUESTIONS:
        raise SystemExit("the spec's primary questions differ from H3's (contracts 26.9, 26.20)")
    arrays = Arrays(run)
    ROWS.update(rows)
    CELLS.update({name: {seed: trial(arrays.get(record)) for seed, record in by_seed.items() if complete(record)}
                  for name, by_seed in rows.items()})
    result = {"experiment_id": frozen["experiment_id"], "purpose": spec["purpose"],
              "points_key": POINTS_KEY, "points": spec[POINTS_KEY],
              "complete_trials": summary["complete_trials"], "planned_trials": summary["planned_trials"],
              "trials_failing_core_validation": summary["trials_failing_core_validation"],
              "primary": primary_questions(questions, seeds), "sensitivity": sensitivity(spec[POINTS_KEY], seeds),
              "auxiliary": auxiliary(summary, questions), "describe": describe(seeds, arrays)}
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
