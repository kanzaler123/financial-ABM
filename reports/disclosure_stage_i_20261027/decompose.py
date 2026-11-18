"""Decomposition of the displacement shares (research contract section 26.22 (c)). Read-only on the archive.

  python decompose.py ../disclosure_stage_h3_20261027/h3_v1     # post-hoc: P3, P5, P8 already seen
  python decompose.py stage_i_v2                                # after analyze.py stage_i_v2 (contract 26.22 (a))
  python decompose.py check_v2 --smoke                          # development: writes and prints counts only

Supplementary analysis: outside the Holm family, no supported/unsupported judgment. Conventions are those of
analyze.py's trial(): decision periods 21-58 (days 1260-3480, 38 decision days), a washing decision is action 1,
ties for the year's lowest-quality assurer all count as lowest; quantities are computed per seed, then paired
differences get a 95% t interval without adjustment; a seed where a quantity is undefined in either scenario
(zero denominator) is left out of that quantity and counted.

  P3 (F against B) and P5 (V against B), for every size class c: the share s_c = n_c / N (the primary quantity,
  unchanged), the class's washing decisions n_c and all washing decisions N, the class rate r_c = n_c / D_c over
  its D_c decisions, the mean amount of a washing decision, the fraction at the period cap 0.25, and the class's
  mean inflation stock on decision days (SD units). Per seed, exactly, log(s1/s0) = log(n1/n0) - log(N1/N0).

  P8 (ALL_F16 against ALL_R16), over assured firms' decisions: A assured decisions, L those of clients of the
  laxest provider, client share c = L / A, washing rates r_L (laxest clients) and r_O (other clients), so the
  share is s = c r_L / (c r_L + (1 - c) r_O). The change in s is split exactly into "client allocation" (c alone)
  and "within-client behaviour" (r_L and r_O) by the two-factor Shapley rule (the mean over both orders).

The blocks are the prefixes p ("" for the main block) for which pB, pF, pV, pALL_R16 and pALL_F16 all exist.
Comparisons across worlds are distributions in two worlds, not the same firms moving.
"""

from __future__ import annotations

import argparse
import json
import math
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
NAMES = ("actions", "amounts", "inflation", "market_size_class", "assurer_assignment", "assurer_quality")
SIZE_CONTRASTS = (("P3", "F", "B", "large"), ("P5", "V", "B", "small"))
ASSURANCE_CONTRAST = ("P8", "ALL_F16", "ALL_R16")
BLOCK_SCENARIOS = ("B", "F", "V", "ALL_R16", "ALL_F16")


def ratio(part: float, whole: float) -> float | None:
    return part / whole if whole else None


def stat(values: list[float]) -> dict | None:
    if len(values) < 2:
        return None
    out = paired_statistics(values, 0.95)
    return {key: out[key] for key in ("mean", "standard_error", "ci_low", "ci_high", "n")}


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# -- per-trial quantities -----------------------------------------------------------

def size_cells(data: dict) -> dict:
    wash = data["actions"][WINDOW] == INFLATE
    amounts = data["amounts"][WINDOW]
    stock = data["inflation"][::PERIOD][WINDOW]     # after that day's audit: what the firm holds when deciding
    size = data["market_size_class"]
    total = int(wash.sum())
    out = {"N": total}
    for index, name in enumerate(SIZES):
        members = size == index
        cell = wash[:, members]
        count = int(cell.sum())
        out[name] = {
            "n": count, "D": int(cell.size), "share": ratio(count, total), "rate": ratio(count, cell.size),
            "mean_amount": float(amounts[:, members][cell].mean()) if count else None,
            "at_cap": ratio(int((cell & (amounts[:, members] >= CAP - 1e-9)).sum()), count),
            "inflation_stock": float(stock[:, members].mean()) if members.any() else None,
        }
    return out


def assurance_cell(data: dict) -> dict | None:
    if "assurer_assignment" not in data:
        return None
    wash = data["actions"][WINDOW] == INFLATE
    assignment = data["assurer_assignment"][WINDOW]
    quality = data["assurer_quality"][WINDOW]
    assured = assignment >= 0
    client_quality = np.take_along_axis(quality, np.maximum(assignment, 0), axis=1)
    laxest = assured & (client_quality == quality.min(axis=1, keepdims=True))
    A, L = int(assured.sum()), int(laxest.sum())
    wash_laxest, wash_other = int((wash & laxest).sum()), int((wash & assured & ~laxest).sum())
    out = {"A": A, "L": L, "c": ratio(L, A), "r_L": ratio(wash_laxest, L), "r_O": ratio(wash_other, A - L),
           "assured_wash": wash_laxest + wash_other, "assured_wash_at_laxest": wash_laxest, "N": int(wash.sum()),
           "share": ratio(wash_laxest, wash_laxest + wash_other)}
    if None not in (out["c"], out["r_L"], out["r_O"]) and out["share"] is not None:
        assert abs(share_of(out["c"], out["r_L"], out["r_O"]) - out["share"]) < 1e-12
    return out


def share_of(c: float, r_l: float, r_o: float) -> float:
    return c * r_l / (c * r_l + (1 - c) * r_o)


def shapley(plus: dict, minus: dict) -> tuple[float, float] | None:
    """Client allocation and within-client behaviour parts of share(plus) - share(minus); None if undefined."""
    try:
        c1, c0 = plus["c"], minus["c"]
        r1, r0 = (plus["r_L"], plus["r_O"]), (minus["r_L"], minus["r_O"])
        f = {(i, j): share_of((c0, c1)[i], *((r0, r1)[j])) for i in (0, 1) for j in (0, 1)}
    except (TypeError, ZeroDivisionError):
        return None
    allocation = ((f[1, 0] - f[0, 0]) + (f[1, 1] - f[0, 1])) / 2
    behaviour = ((f[0, 1] - f[0, 0]) + (f[1, 1] - f[1, 0])) / 2
    assert abs(allocation + behaviour - (f[1, 1] - f[0, 0])) < 1e-12
    return allocation, behaviour


# -- contrasts ------------------------------------------------------------------------

def paired(plus: dict, minus: dict, value, seeds: list[int]) -> dict:
    """Levels of both scenarios and their paired difference, over the seeds where the value is defined in both."""
    pairs, left_out = [], []
    for seed in seeds:
        a = value(plus[seed]) if seed in plus else None
        b = value(minus[seed]) if seed in minus else None
        if a is None or b is None:
            left_out.append(seed)
        else:
            pairs.append((a, b))
    return {"plus": mean([a for a, _ in pairs]), "minus": mean([b for _, b in pairs]),
            "difference": stat([a - b for a, b in pairs]) or {"n": len(pairs)}, "seeds_left_out": left_out}


def per_seed(plus: dict, minus: dict, value, seeds: list[int]) -> dict:
    """A per-seed quantity of the pair (already a contrast), over the seeds where it is defined."""
    values, left_out = [], []
    for seed in seeds:
        v = value(plus[seed], minus[seed]) if seed in plus and seed in minus else None
        (left_out.append(seed) if v is None else values.append(v))
    return {"value": stat(values) or {"n": len(values)}, "seeds_left_out": left_out}


def log_ratio(key: str, size: str | None = None):
    def value(a: dict, b: dict) -> float | None:
        x, y = (a[size][key], b[size][key]) if size else (a[key], b[key])
        return math.log(x / y) if x and y else None
    return value


def size_contrast(cells: dict, plus: str, minus: str, seeds: list[int]) -> dict:
    p, m = cells[plus], cells[minus]
    out = {"N": paired(p, m, lambda cell: cell["N"], seeds), "log_N_ratio": per_seed(p, m, log_ratio("N"), seeds)}
    for size in SIZES:
        row = {key: paired(p, m, lambda cell, key=key: cell[size][key], seeds)
               for key in ("share", "n", "rate", "mean_amount", "at_cap", "inflation_stock")}
        row["log_share_ratio"] = per_seed(p, m, log_ratio("share", size), seeds)
        row["log_n_ratio"] = per_seed(p, m, log_ratio("n", size), seeds)
        out[size] = row
    return out


def assurance_contrast(cells: dict, plus: str, minus: str, seeds: list[int]) -> dict:
    p, m = cells[plus], cells[minus]
    out = {key: paired(p, m, lambda cell, key=key: cell[key], seeds)
           for key in ("share", "c", "r_L", "r_O", "A", "L", "assured_wash", "assured_wash_at_laxest", "N")}
    out["allocation"] = per_seed(p, m, lambda a, b: (shapley(a, b) or (None,))[0], seeds)
    out["behaviour"] = per_seed(p, m, lambda a, b: None if shapley(a, b) is None else shapley(a, b)[1], seeds)
    return out


# -- loading, tables, main ------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict]:
    frozen = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["experiment_id"] != frozen["experiment_id"]:
        raise ValueError("summary and experiment manifest disagree")
    return frozen, summary


def blocks(names: set[str]) -> list[str]:
    prefixes = sorted({name[: -len("ALL_F16")] for name in names if name.endswith("ALL_F16")})
    return [p for p in prefixes if all(p + s in names for s in BLOCK_SCENARIOS)]


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def diff(row: dict, scale: float = 100, digits: int = 1) -> str:
    d = row["difference"] if "difference" in row else row["value"]
    if d.get("mean") is None:
        return "—"
    f = lambda v: f"{scale * v:+.{digits}f}".replace("-", "−")
    return f"{f(d['mean'])} [{f(d['ci_low'])}, {f(d['ci_high'])}]"


def tables(result: dict) -> list[str]:
    out = [f"# 转移的分解（契约第 26.22 条 (c)）：`{result['archive']}`", "",
           f"experiment_id `{result['experiment_id']}`；{result['status_note']}", "",
           "补充分析，不进 Holm 族，不作判定。窗口为第 1260–3480 日的 38 个决策日；先逐种子计算，再取配对差的 95% t 区间。"
           "不同世界之间的差异是两个世界中的分布，不是同一企业的迁移。", ""]
    for prefix, block in result["blocks"].items():
        label = prefix or "主块"
        for pid, plus, minus, target in SIZE_CONTRASTS:
            row = block[pid]
            out += [f"## {label} {pid}：{prefix}{plus} 对 {prefix}{minus}（目标组：{target}）", "",
                    f"全部漂绿决策 N（每个世界）：{num(row['N']['minus'])} → {num(row['N']['plus'])}，"
                    f"差 {diff(row['N'], 1)}；log(N¹/N⁰) {diff(row['log_N_ratio'], 1, 3)}", "",
                    "| 组 | 份额 | 份额差（百分点） | 漂绿次数 | 次数差 | 组内率 | 组内率差（百分点） | log 份额比 | log 次数比 | 平均幅度 | 顶到上限 | 虚报存量（SD） |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for size in SIZES:
                r = row[size]
                out.append(
                    f"| {size} | {pct(r['share']['minus'])} → {pct(r['share']['plus'])} | {diff(r['share'])} | "
                    f"{num(r['n']['minus'])} → {num(r['n']['plus'])} | {diff(r['n'], 1)} | "
                    f"{pct(r['rate']['minus'], 2)} → {pct(r['rate']['plus'], 2)} | {diff(r['rate'], 100, 2)} | "
                    f"{diff(r['log_share_ratio'], 1, 3)} | {diff(r['log_n_ratio'], 1, 3)} | "
                    f"{num(r['mean_amount']['minus'], 3)} → {num(r['mean_amount']['plus'], 3)} | "
                    f"{pct(r['at_cap']['minus'])} → {pct(r['at_cap']['plus'])} | "
                    f"{num(r['inflation_stock']['minus'], 3)} → {num(r['inflation_stock']['plus'], 3)} |")
            out.append("")
        pid, plus, minus = ASSURANCE_CONTRAST
        row = block[pid]
        out += [f"## {label} {pid}：{prefix}{plus} 对 {prefix}{minus}", "",
                "| 量 | " + f"{prefix}{minus}" + " | " + f"{prefix}{plus}" + " | 配对差 [95% 区间] |", "|---|---|---|---|"]
        for key, label_key, as_pct in (("share", "最宽松方客户在受鉴证漂绿中的份额 s", True),
                                       ("c", "最宽松方的客户份额 c", True),
                                       ("r_L", "最宽松方客户的漂绿率 r_L", True),
                                       ("r_O", "其他客户的漂绿率 r_O", True),
                                       ("assured_wash", "受鉴证漂绿决策数", False),
                                       ("assured_wash_at_laxest", "其中最宽松方客户", False),
                                       ("N", "全部漂绿决策数", False)):
            r = row[key]
            lv = (lambda v: pct(v, 2)) if as_pct else num
            out.append(f"| {label_key} | {lv(r['minus'])} | {lv(r['plus'])} | "
                       f"{diff(r, 100 if as_pct else 1, 2 if as_pct else 1)} |")
        out += [f"| Δs 中的客户分配（Shapley） | | | {diff(row['allocation'], 100, 2)} |",
                f"| Δs 中的客户内行为（Shapley） | | | {diff(row['behaviour'], 100, 2)} |", ""]
    return out


def counts(result: dict) -> dict:
    def n(row: dict) -> int:
        return (row.get("difference") or row.get("value"))["n"]
    return {"archive": result["archive"], "experiment_id": result["experiment_id"], "purpose": result["purpose"],
            "complete_trials": result["complete_trials"], "blocks": {
                prefix or "main": {pid: {"n_share": n(block[pid][target]["share"]) if target else n(block[pid]["share"]),
                                         "n_split": None if target else n(block[pid]["allocation"]),
                                         "seeds_left_out": len((block[pid][target]["share"] if target else block[pid]["share"])["seeds_left_out"])}
                                   for pid, target in (("P3", "large"), ("P5", "small"), ("P8", None))}
                for prefix, block in result["blocks"].items()},
            "table_lines": len(tables(result))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="archive directory, relative to this script, e.g. stage_i_v2")
    parser.add_argument("--smoke", action="store_true", help="development archive: write and print counts only")
    args = parser.parse_args()
    run = (HERE / args.run).resolve()
    frozen, summary = load(run)
    purpose = frozen["spec"]["purpose"]
    if purpose == "development" and not args.smoke:
        raise SystemExit("development archives are decomposed with --smoke only")
    seeds = frozen["spec"]["seeds"]
    names = {record["scenario"] for record in summary["records"]}
    prefixes = blocks(names)
    wanted = {p + s for p in prefixes for s in BLOCK_SCENARIOS}
    size, assurance = {}, {}
    for record in summary["records"]:
        if record["scenario"] not in wanted or record["status"] != "complete":
            continue
        with np.load(run / record["path"] / "result.npz") as saved:
            data = {name: saved[name] for name in NAMES if name in saved.files}
        size.setdefault(record["scenario"], {})[record["seed"]] = size_cells(data)
        cell = assurance_cell(data)
        if cell is not None:
            assurance.setdefault(record["scenario"], {})[record["seed"]] = cell
    result = {"archive": run.relative_to(ROOT).as_posix(), "experiment_id": frozen["experiment_id"], "purpose": purpose,
              "complete_trials": summary["complete_trials"], "planned_trials": summary["planned_trials"],
              "status_note": ("事后分析：P3、P5、P8 的结果已在本分解之前看过（契约第 26.22 条 (c)）"
                              if run.name == "h3_v1" else "补充分析，写于分析之前（契约第 26.22 条 (c)）"),
              "blocks": {}}
    for p in prefixes:
        block = {pid: size_contrast(size, p + plus, p + minus, seeds) for pid, plus, minus, _ in SIZE_CONTRASTS}
        pid, plus, minus = ASSURANCE_CONTRAST
        block[pid] = assurance_contrast(assurance, p + plus, p + minus, seeds)
        result["blocks"][p] = block
    stem = "decomposition_" + run.name
    if args.smoke:
        out = counts(result)
        (HERE / f"{stem}_smoke.json").write_bytes((json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    (HERE / f"{stem}.json").write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    (HERE / f"{stem}.md").write_bytes(("\n".join(tables(result)) + "\n").encode("utf-8"))
    print(f"wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
