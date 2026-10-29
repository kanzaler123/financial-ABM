"""Stage J, model counterparts (contract 27.10, option M-a): replay of the H3 B runs with a read-only recorder.

    .venv/Scripts/python.exe replay.py run [--workers 16]     # 30 replays, then the model moments
    .venv/Scripts/python.exe replay.py moments                # the model moments from the saved arrays

Scenario B of configs/disclosure_stage_h3_v1.json (frozen identity 12fe871e...d8eb, contract 25.17), seeds
53001-53030, the whole 3500 days. A wrapper on _record_attributes (called once a day after trading; it reads and
returns nothing) keeps, on the sampling days t_j = 1260 + 63 j (j = 0..35, contract 25.3): the public signal, the
price, the intrinsic value, and the shares held by impact investors and by all investors (fractions of each firm's
shares). Each replay's result fingerprint must equal the manifest of the same seed in the h3_v1 archive; when any
differs, no moment is computed. Arrays go to replay_h3_b/ (local, not committed); writes replay_h3_b.json and,
when every fingerprint matches, model_moments.json.

Moments (contract 27.10; the cross-sectional SD convention of 27.15):
  price response    per seed and quarter [t_j, t_j+1], the cross-sectional OLS slope of the change in log(P / V)
                    on the change in the public signal z, times the quarter's cross-sectional SD of z (mean of the
                    SDs at t_j and t_j+1); averaged over quarters, then over seeds; percentage points per SD;
  holding response  the same with the change in the impact investors' share of the firm's shares (percentage points);
  impact share      median of impact_wealth_share over days 1200-3500 (and 60-1200);
  yield slope       pricing_slope (dividend yield on the public signal, contract 16.3) on the decision days 60..1200
                    and 1260..3480, median over seeds and days, basis points per signal unit, and per cross-sectional
                    SD (times that day's SD of the public signal, recorded daily by the wrapper).
Descriptive variants: the raw log return instead of log(P / V), and rank-normal signals instead of z times SD.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from _no_wmi import prefill_uname

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
prefill_uname()  # importing scipy can block on a hung WMI service (see _no_wmi.py)

from scipy.special import ndtri  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from abm.disclosure.experiments import _inputs, freeze_spec  # noqa: E402
from abm.disclosure.game import IMPACT_INVESTOR, DisclosureMarket  # noqa: E402
from abm.platform_tuning import disable_power_throttling  # noqa: E402

H3_SPEC = ROOT / "configs" / "disclosure_stage_h3_v1.json"
H3_ARCHIVE = ROOT / "reports" / "disclosure_stage_h3_20261027" / "h3_v1" / "B"
SCENARIO = "B"
SEEDS = tuple(range(53001, 53031))
SAMPLE_DAYS = tuple(1260 + 63 * j for j in range(36))
ARRAYS = HERE / "replay_h3_b"


def archived_fingerprint(seed: int) -> str:
    done = []
    for manifest in sorted((H3_ARCHIVE / f"seed_{seed}").glob("attempt_*/manifest.json")):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m["status"] == "complete":
            done.append(m["result_fingerprint"])
    if len(done) != 1:
        raise RuntimeError(f"seed {seed}: {len(done)} complete attempts in h3_v1")
    return done[0]


def run_one(scenario: dict, seed: int) -> dict:
    config, domain, policy = _inputs(scenario, seed)
    market = DisclosureMarket(config, domain, policy)
    rows = {k: [] for k in ("day", "signal", "price", "value", "impact_shares", "investor_shares")}
    signal_sd = np.full(config.trading_days + 1, np.nan)
    original = market._record_attributes
    sample = set(SAMPLE_DAYS)

    def recorder(day: int) -> None:
        original(day)
        signal_sd[day] = market.public_signal_history[day].std()
        if day in sample:
            inv = market.investors
            impact = inv.investor_type == IMPACT_INVESTOR
            rows["day"].append(day)
            rows["signal"].append(market.public_signal_history[day].copy())
            rows["price"].append(market.prices.copy())
            rows["value"].append(market.firms.intrinsic_value.copy())
            rows["impact_shares"].append(inv.holdings[impact].sum(axis=0))
            rows["investor_shares"].append(inv.holdings.sum(axis=0))

    market._record_attributes = recorder
    started = time.perf_counter()
    result = market.run()
    fingerprint = result.fingerprint()
    ARRAYS.mkdir(exist_ok=True)
    path = ARRAYS / f"seed_{seed}.npz"
    np.savez_compressed(path, **{k: np.array(v) for k, v in rows.items()},
                        impact_wealth_share=result.impact_wealth_share, impact_share=result.impact_share,
                        pricing_slope=result.pricing_slope, signal_sd=signal_sd)
    expected = archived_fingerprint(seed)
    return {"seed": seed, "fingerprint": fingerprint, "archived_fingerprint": expected,
            "fingerprint_matches": fingerprint == expected, "sampled_days": len(rows["day"]),
            "price_cap_hits": int(result.market.price_cap_hits.sum()), "seconds": time.perf_counter() - started,
            "arrays": path.relative_to(HERE).as_posix()}


def slope(y: np.ndarray, x: np.ndarray) -> float:
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def rank_normal(z: np.ndarray) -> np.ndarray:
    return ndtri((rankdata(z) - 0.5) / z.size)


def moments(runs: list[dict]) -> dict:
    per_seed = {k: [] for k in ("price", "price_raw", "price_rank", "holding", "holding_rank", "sd_z")}
    wealth_late, wealth_pre, ys_pre, ys_late, ys_pre_sd, ys_late_sd = [], [], [], [], [], []
    for run in runs:
        with np.load(HERE / run["arrays"]) as a:
            z, p, v, h = a["signal"], a["price"], a["value"], a["impact_shares"]
            iws, ps, sdd = a["impact_wealth_share"], a["pricing_slope"], a["signal_sd"]
        q = {k: [] for k in per_seed}
        for j in range(len(z) - 1):
            dz = z[j + 1] - z[j]
            sd = 0.5 * (z[j].std() + z[j + 1].std())
            dr = rank_normal(z[j + 1]) - rank_normal(z[j])
            dlogpv = np.log(p[j + 1] / v[j + 1]) - np.log(p[j] / v[j])
            dlogp = np.log(p[j + 1] / p[j])
            dh = 100 * (h[j + 1] - h[j])
            q["price"].append(100 * slope(dlogpv, dz) * sd)
            q["price_raw"].append(100 * slope(dlogp, dz) * sd)
            q["price_rank"].append(100 * slope(dlogpv, dr))
            q["holding"].append(slope(dh, dz) * sd)
            q["holding_rank"].append(slope(dh, dr))
            q["sd_z"].append(sd)
        for k in per_seed:
            per_seed[k].append(float(np.mean(q[k])))
        wealth_late.append(iws[1200:3501])
        wealth_pre.append(iws[60:1201])
        pre_days, late_days = np.arange(60, 1201, 60), np.arange(1260, 3481, 60)
        ys_pre.append(ps[pre_days] * 1e4)
        ys_late.append(ps[late_days] * 1e4)
        ys_pre_sd.append(ps[pre_days] * 1e4 * sdd[pre_days])
        ys_late_sd.append(ps[late_days] * 1e4 * sdd[late_days])

    def across(values: list[float]) -> dict:
        x = np.array(values)
        se = x.std(ddof=1) / np.sqrt(x.size)
        return {"mean": float(x.mean()), "se_across_seeds": float(se), "ci95": [float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)],
                "seeds": int(x.size)}

    return {
        "price_response_pp_per_sd": across(per_seed["price"]),
        "price_response_raw_return_pp_per_sd": across(per_seed["price_raw"]),
        "price_response_rank_normal_pp_per_sd": across(per_seed["price_rank"]),
        "impact_holding_response_pp_per_sd": across(per_seed["holding"]),
        "impact_holding_response_rank_normal_pp_per_sd": across(per_seed["holding_rank"]),
        "signal_cross_sectional_sd_mean": across(per_seed["sd_z"]),
        "impact_wealth_share_median": {"days_1200_3500": float(np.median(np.concatenate(wealth_late))),
                                       "days_60_1200": float(np.median(np.concatenate(wealth_pre)))},
        "yield_slope_bp_per_signal_unit_median": {"decision_days_60_1200": float(np.median(np.concatenate(ys_pre))),
                                                  "decision_days_1260_3480": float(np.median(np.concatenate(ys_late)))},
        "yield_slope_bp_per_cross_sectional_sd_median": {"decision_days_60_1200": float(np.median(np.concatenate(ys_pre_sd))),
                                                         "decision_days_1260_3480": float(np.median(np.concatenate(ys_late_sd)))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "moments"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.mode == "run":
        frozen = freeze_spec(H3_SPEC)
        scenario = {row["name"]: row for row in frozen["scenarios"]}[SCENARIO]
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[name] = "1"
        disable_power_throttling()
        runs = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=disable_power_throttling) as pool:
            futures = {pool.submit(run_one, scenario, seed): seed for seed in SEEDS}
            for future in as_completed(futures):
                out = future.result()
                runs.append(out)
                print(out["seed"], out["fingerprint_matches"], round(out["seconds"], 1), flush=True)
        runs.sort(key=lambda r: r["seed"])
        record = {"spec": H3_SPEC.relative_to(ROOT).as_posix(), "experiment_id_current_source": frozen["experiment_id"],
                  "scenario": SCENARIO, "seeds": [SEEDS[0], SEEDS[-1]], "sample_days": [SAMPLE_DAYS[0], SAMPLE_DAYS[-1], len(SAMPLE_DAYS)],
                  "all_fingerprints_match": all(r["fingerprint_matches"] for r in runs), "runs": runs}
        (HERE / "replay_h3_b.json").write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    record = json.loads((HERE / "replay_h3_b.json").read_text(encoding="utf-8"))
    if not record["all_fingerprints_match"]:
        raise SystemExit("a replay differs from its archived fingerprint: no moments (contract 27.10)")
    out = moments(record["runs"])
    (HERE / "model_moments.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
