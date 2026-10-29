"""Stage K1 estimates: disclosure-enforcement incidence, its size pattern, fines and composition (contract 28.2).

    python -I -X utf8 k1_estimate.py --final
    python -I -X utf8 k1_estimate.py --check     # the full path on permuted events; writes only counts

--check reassigns every event to a random firm-year of the same year (stream [28901, 99]) and writes
k1_check.json with sample sizes and event counts only, so the code path is tested before the one final run.

Reads the panels written by k1_build.py (hashes checked against PANELS) and the CNRDS monthly returns (market value
in the month before an announcement). Runs once: refuses when k1_estimates.json exists. All intervals are percentile
intervals of a bootstrap over firms (1,000 draws, seed 28901, contract 28.8). Nothing here touches ratings.

Mapping to the model (contract 28.2): the incidences are lower bounds on the yearly detection probability of general
disclosure violations among violating firms; they are not estimates of the model's audit probability q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT / "data/processed/stage_k_20261029"
CNRDS = ROOT / "data/processed/cnrds_20261026"
PANELS = {"k1_events.csv.gz": "9e18ade4478d9a7036a9bba2e7056b0c600115157a9ef947da7219e8cb7670a4",
          "k1_firm_year.csv.gz": "502873098264df5cea39d996778f1a817976ccf07f724cdbadd2be50a021d3a9"}
OUTPUT = HERE / "k1_estimates.json"
SEED, DRAWS = 28901, 1000
MAIN_YEARS = (2016, 2025)
LAW_DATE = "2020-03-01"                       # revised Securities Law in force
FK_WINDOW = ("2020-03-01", "2025-12-31")
STATUTORY_FINE_YUAN = 1.0e7
MEAN_CAP_YUAN = 2.1923e11                     # contract 16.2 (2.19226573732597e11)
MEAN_CAP_YUAN_EXACT = 219226573732.59695
MODEL_AUDIT_Q, AUDITS_PER_YEAR = 0.01, 252 / 60


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Boot:
    """Firm-level bootstrap: every statistic is a function of per-firm sums, so a draw is a weighted sum."""

    def __init__(self, firms: pd.Index):
        self.firms = firms
        rng = np.random.default_rng(SEED)
        self.w = rng.multinomial(len(firms), np.full(len(firms), 1 / len(firms)), size=DRAWS).astype(np.float64)

    def sums(self, df: pd.DataFrame, col: str) -> tuple[float, np.ndarray]:
        s = df.groupby("Scode")[col].sum().reindex(self.firms, fill_value=0.0).to_numpy(np.float64)
        return float(s.sum()), self.w @ s


def ci(draws: np.ndarray) -> list[float]:
    d = draws[np.isfinite(draws)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [None, None]


def rate(boot: Boot, df: pd.DataFrame, col: str) -> dict:
    df = df.assign(_one=1.0, _x=df[col].astype(float))
    num, num_b = boot.sums(df, "_x")
    den, den_b = boot.sums(df, "_one")
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = num_b / den_b
    return {"rate": num / den if den else None, "ci95": ci(draws), "events_firm_years": int(num),
            "firm_years": int(den), "_draws": draws}


def log_ratio(a: dict, b: dict) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.log(a["_draws"] / b["_draws"])
    return {"log_ratio": float(np.log(a["rate"] / b["rate"])), "ratio": float(a["rate"] / b["rate"]), "ci95_log": ci(draws),
            "draws_undefined": int((~np.isfinite(draws)).sum())}


def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [clean(v) for v in d]
    return d


def size_groups(fy: pd.DataFrame) -> pd.Series:
    """Per year: top 10% of previous year-end market value large, next 30% mid, rest small (model shares)."""
    out = pd.Series(pd.NA, index=fy.index, dtype="object")
    for _, g in fy[fy.mcap_prev_dec > 0].groupby("year"):
        r = g.mcap_prev_dec.rank(ascending=False, method="first") / len(g)
        out[g.index] = np.where(r <= 0.10, "large", np.where(r <= 0.40, "mid", "small"))
    return out


def firm_year_flags(fy: pd.DataFrame, e: pd.DataFrame, tiers: dict[str, list[str]], event_filter=None) -> pd.DataFrame:
    ev = e if event_filter is None else e[event_filter(e)]
    out = fy.copy()
    for name, ts in tiers.items():
        hit = ev[ev.tier.isin(ts)].groupby(["Scode", "year"]).size().rename(name).reset_index()
        out = out.merge(hit, on=["Scode", "year"], how="left")
        out[name] = out[name].fillna(0) > 0
    return out


TIERS = {"T1": ["T1"], "T1_T2": ["T1", "T2"], "T1_T2_T3": ["T1", "T2", "T3"]}


def incidence_block(fy: pd.DataFrame, e: pd.DataFrame, boot: Boot, years=MAIN_YEARS, event_filter=None,
                    with_size=True, tiers=TIERS) -> dict:
    sub = fy[fy.year.between(*years)]
    flags = firm_year_flags(sub, e, tiers, event_filter)
    out = {"pooled": {t: rate(boot, flags, t) for t in tiers}}
    if with_size:
        groups = {g: flags[flags["size_group"] == g] for g in ("large", "mid", "small")}
        out["by_size"] = {g: {t: rate(boot, d, t) for t in tiers} for g, d in groups.items()}
        out["log_ratio_small_over_large"] = {t: log_ratio(out["by_size"]["small"][t], out["by_size"]["large"][t])
                                             for t in tiers}
        out["firm_years_without_size_group"] = int(flags.size_group.isna().sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.final == args.check:
        raise SystemExit("run with --final (once) or --check")
    if args.final and OUTPUT.exists():
        raise SystemExit(f"{OUTPUT.name} exists; the final run is done")
    for name, digest in PANELS.items():
        if sha(PANEL / name) != digest:
            raise SystemExit(f"{name}: hash differs from the recorded panel")

    e = pd.read_csv(PANEL / "k1_events.csv.gz", dtype={"Scode": str, "Itemid": str})
    fy = pd.read_csv(PANEL / "k1_firm_year.csv.gz", dtype={"Scode": str})
    fy["size_group"] = size_groups(fy)
    if args.check:
        rng = np.random.default_rng([SEED, 99])
        codes = fy.groupby("year").Scode.apply(list).to_dict()
        e = e[e.year.isin(codes)].copy()
        e["Scode"] = [codes[y][i] for y, i in zip(e.year, rng.integers(0, 10**9, len(e)) % e.year.map(lambda y: len(codes[y])))]
    nonfin = fy[~fy.financial].copy()
    boot = Boot(pd.Index(sorted(fy.Scode.unique())))
    res: dict = {"panels": PANELS, "seed": SEED, "draws": DRAWS, "main_years": list(MAIN_YEARS)}

    # (i) and (ii): main population, 2016-2025.
    res["main"] = incidence_block(nonfin, e, boot)
    res["by_year"] = {}
    for y in range(2008, 2026):
        flags = firm_year_flags(nonfin[nonfin.year == y], e, TIERS)
        res["by_year"][str(y)] = {t: {"rate": float(flags[t].mean()), "events_firm_years": int(flags[t].sum()),
                                      "firm_years": int(len(flags))} for t in TIERS}
    res["by_law_period"] = {
        "2016-2019": incidence_block(nonfin, e, boot, years=(2016, 2019)),
        "2020-2025": incidence_block(nonfin, e, boot, years=(2020, 2025)),
    }
    res["pre_2016_descriptive"] = incidence_block(nonfin, e, boot, years=(2008, 2015))

    # (iii) fines: T1 events of main-population firms, the company's own fine (10,000 yuan).
    pop_keys = nonfin[["Scode", "year"]].assign(_in=True)
    ev = e.merge(pop_keys, on=["Scode", "year"], how="left")
    ev = ev[ev._in.fillna(False).astype(bool)].drop(columns="_in")
    mr = pd.read_csv(CNRDS / "monthly_returns.csv.gz", dtype=str, usecols=["Scode", "Trdmt", "Mmkecap"])
    mr = mr[mr.Trdmt != "2017-03-10"]
    mcap = pd.Series(pd.to_numeric(mr.Mmkecap, errors="coerce").to_numpy(),
                     index=pd.MultiIndex.from_arrays([mr.Scode, pd.PeriodIndex(mr.Trdmt.str[:7], freq="M")]))
    ev["mcap_prev_month"] = mcap.reindex(pd.MultiIndex.from_arrays(
        [ev.Scode, pd.PeriodIndex(ev.date.str[:7], freq="M") - 1])).to_numpy()

    def fines(d: pd.DataFrame) -> dict:
        f = d[(d.tier == "T1") & (d.fine_wan > 0)]
        share = (f.fine_wan * 1e4 / f.mcap_prev_month).dropna()
        q = f.fine_wan.quantile([0.1, 0.5, 0.9]).tolist() if len(f) else [None] * 3
        return {"t1_events": int((d.tier == "T1").sum()), "t1_events_with_fine": int(len(f)),
                "fine_wan_mean": float(f.fine_wan.mean()) if len(f) else None,
                "fine_wan_p10_p50_p90": [float(x) for x in q],
                "fine_share_of_market_value": {"n": int(len(share)), "median": float(share.median()) if len(share) else None,
                                               "p90": float(share.quantile(0.9)) if len(share) else None}}

    main_ev = ev[ev.year.between(*MAIN_YEARS)]
    res["fines"] = {"2016-2025": fines(main_ev),
                    "before_law": fines(main_ev[main_ev.date < LAW_DATE]),
                    "from_law": fines(main_ev[main_ev.date >= LAW_DATE])}

    # F_K: mean company fine over T1 and T2 events (T2 counts 0), 2020-03-01 to 2025-12-31.
    win = ev[(ev.date >= FK_WINDOW[0]) & (ev.date <= FK_WINDOW[1]) & ev.tier.isin(["T1", "T2"])].copy()
    win["_one"] = 1.0
    win["_fine"] = win.fine_wan.where(win.tier == "T1", 0.0).fillna(0.0)
    tot, tot_b = boot.sums(win, "_fine")
    n, n_b = boot.sums(win, "_one")
    fk_yuan, fk_b = tot / n * 1e4, tot_b / n_b * 1e4
    rel = fk_yuan / STATUTORY_FINE_YUAN - 1
    res["F_K"] = {"events": int(n), "t1_events": int((win.tier == "T1").sum()), "mean_fine_yuan": fk_yuan,
                  "ci95_yuan": ci(fk_b), "model_units": fk_yuan / MEAN_CAP_YUAN_EXACT,
                  "relative_to_statutory_cap": rel,
                  "rule": "use F_K unless within 10% of 10 million yuan (contract 28.2)",
                  "model_fine_used_in_K3_model_units": (STATUTORY_FINE_YUAN / MEAN_CAP_YUAN_EXACT if abs(rel) < 0.10
                                                        else fk_yuan / MEAN_CAP_YUAN_EXACT)}

    # (iv) composition among firm-years with an event.
    sub = firm_year_flags(nonfin[nonfin.year.between(*MAIN_YEARS)], e, TIERS)
    any_ev = sub[sub.T1_T2_T3]
    res["composition_highest_tier"] = {"firm_years_with_event": int(len(any_ev)),
                                       "T1": float(any_ev.T1.mean()),
                                       "T2": float((any_ev.T1_T2 & ~any_ev.T1).mean()),
                                       "T3": float((~any_ev.T1_T2).mean())}
    res["composition_events"] = {t: int(n) for t, n in main_ev.tier.value_counts().sort_index().items()}

    # (v) ESG-specific disclosure events: the list.
    res["esg_events"] = e[e.esg][["Itemid", "Scode", "date", "tier", "fine_wan"]].to_dict("records")

    # Model reference (contract 28.2).
    res["model_reference"] = {"q_per_decision_period": MODEL_AUDIT_Q,
                              "yearly_audit_probability": 1 - (1 - MODEL_AUDIT_Q) ** AUDITS_PER_YEAR}

    # Robustness (descriptive).
    top300 = nonfin.copy()
    top300["size_group"] = pd.NA
    keep300 = []
    for _, g in nonfin[nonfin.mcap_prev_dec > 0].groupby("year"):
        keep300.extend(g.nlargest(300, "mcap_prev_dec").index)
    res["robustness"] = {
        "including_financials": incidence_block(fy, e, boot),
        "top_300_by_market_value": incidence_block(nonfin.loc[keep300], e, boot, with_size=False),
        "without_keyword_定期报告": incidence_block(nonfin, e, boot, event_filter=lambda d: d.no_periodic),
        "t1_by_amount_only": incidence_block(nonfin, e.assign(tier=np.where(e.tier == "T1", np.where(e.t1_amount, "T1", np.where(e.t2, "T2", "T3")), e.tier)),
                                             boot, tiers={"T1": ["T1"], "T1_T2": ["T1", "T2"]}),
    }
    dec = nonfin[nonfin.year.between(*MAIN_YEARS) & (nonfin.mcap_prev_dec > 0)].copy()
    dec["decile"] = dec.groupby("year").mcap_prev_dec.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    dflags = firm_year_flags(dec, e, TIERS)
    res["robustness"]["by_size_decile_1_smallest"] = {
        str(int(d)): {t: float(g[t].mean()) for t in TIERS} | {"firm_years": int(len(g))} for d, g in dflags.groupby("decile")}

    if args.check:
        drop = {"rate", "ci95", "log_ratio", "ratio", "ci95_log", "mean_fine_yuan", "ci95_yuan", "model_units",
                "relative_to_statutory_cap", "model_fine_used_in_K3_model_units", "fine_wan_mean",
                "fine_wan_p10_p50_p90", "median", "p90", "T1", "T2", "T3", "T1_T2", "T1_T2_T3", "esg_events",
                "composition_events", "by_size_decile_1_smallest"}

        def counts(d):
            if isinstance(d, dict):
                return {k: counts(v) for k, v in d.items() if not k.startswith("_") and k not in drop}
            return d
        out = counts(res)
        (HERE / "k1_check.json").write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print("check done; counts only written to k1_check.json")
        return
    text = json.dumps(clean(res), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUTPUT.write_bytes(text.encode("utf-8"))
    m = res["main"]
    print(json.dumps(clean({"pooled": m["pooled"], "log_ratio": m["log_ratio_small_over_large"], "F_K": res["F_K"],
                            "fines": res["fines"], "composition": res["composition_highest_tier"]}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
