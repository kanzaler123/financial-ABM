"""Stage I calibration: the firms' local financing-rate response rho against financing_spread (contract 26.7).

The pre-policy window of the H3 main setting, form Q (contract 25.16-25.17): scenario B of
configs/disclosure_stage_h3_v1.json run for 1200 days (the main runs switch policy on day 1201), contract scale,
calibration seeds 56001-56003, financing_spread s in {0.05, 0.1, 0.2, 0.4, 0.8}. A read-only wrapper on
marginal_financing_saving records, on each of the 20 pre-policy decision days (60, ..., 1200) and for every firm:

  rho          the local financing-rate response firms act on, s E P / (E + P)^2 m (contract 26.7), obtained as the
               benefit B = (E / P) rho times P / E; basis points per SD of the signal per year;
  pass-through s E P / (E + P)^2, so that m = rho / pass-through is the one-rebalance quote response per SD;
  E / P        the financing exposure over market value;

and per decision day the dividend-yield slope (yV/P on the public signal, the taste calibration's statistic of
contract 16.3; diagnostic only) and the cross-sectional slope of log(P / V) on the signal. Per run: price cap hits.
Only pre-policy pricing and financing are read; firm actions, inflation and policy outcomes are not.

  python calibrate.py grid [--workers 16]
      the five grid points, an instrumentation check (an uninstrumented rerun reproduces the fingerprint) and a
      horizon check (the first 1200 days of the 3500-day B run equal the 1200-day run); writes calibration_grid.json
  python calibrate.py target --pi PI --lo LO --hi HI [--workers 3]
      after the empirical estimate (basis points per SD): s*, s_lo, s_hi and s_28 by log-log interpolation of the
      median rho on s, doubling the grid outward while a target lies outside it, then the verification of s* on the
      same seeds (median rho within 10% of PI, otherwise one re-interpolation with the new point; both recorded);
      writes calibration_target.json

Per-firm arrays go to calibration_v1/ (local, not committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from _no_wmi import prefill_uname

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
prefill_uname()  # importing scipy can block on a hung WMI service (see _no_wmi.py)

from abm.disclosure.experiments import _inputs, freeze_spec  # noqa: E402
from abm.disclosure.game import DisclosureMarket, pricing_slope  # noqa: E402
from abm.platform_tuning import disable_power_throttling  # noqa: E402

H3_SPEC = ROOT / "configs" / "disclosure_stage_h3_v1.json"
H3_EXPERIMENT_ID = "12fe871ed3a81d1b7f3ace185fa8c01385612bd0e908b9330b792050612cd8eb"   # contract 25.17
SCENARIO = "B"
SEEDS = (56001, 56002, 56003)            # contract 26.7 and 26.10: calibration only, never used before
GRID = (0.05, 0.1, 0.2, 0.4, 0.8)
CURRENT = 0.05                           # s0, the spread of every run so far
PRE_POLICY_DAYS = 1200                   # policy switch of the main runs on day 1201
BURN_IN = 600                            # metric window only; does not enter the simulated path
TOLERANCE = 0.10                         # verification: median rho within 10% of the target
TARGET_28 = 28.0                         # contract 15.4, basis points per SD
ARRAYS = HERE / "calibration_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_scenario() -> tuple[dict, str]:
    frozen = freeze_spec(H3_SPEC)
    rows = {row["name"]: row for row in frozen["scenarios"]}
    return rows[SCENARIO], frozen["experiment_id"]


def setting(scenario: dict, spread: float, seed: int, full: bool = False):
    """The scenario at the given spread: the pre-policy window, or the whole run of the main setting (full)."""
    config, domain, policy = _inputs(scenario, seed)
    if not full:
        config = replace(config, trading_days=PRE_POLICY_DAYS, burn_in_days=BURN_IN)
    return replace(config, financing_spread=spread), domain, policy


def ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    centered = x - x.mean()
    return float(centered @ (y - y.mean()) / (centered @ centered))


def run_one(scenario: dict, spread: float, seed: int) -> dict:
    """One instrumented pre-policy run; per-firm arrays are written to ARRAYS."""
    config, domain, policy = setting(scenario, spread, seed)
    market = DisclosureMarket(config, domain, policy)
    days, rho, passthrough, exposure_share, yield_slope, premium_slope = [], [], [], [], [], []
    original = market.marginal_financing_saving

    def wrapped():
        benefit = original()
        day = market.signal_day
        signal = market.signal_z(day)
        prices, exposure = market.prices, market.financing_exposure
        days.append(day)
        rho.append(benefit * prices / exposure)
        passthrough.append(config.financing_spread * exposure * prices / (exposure + prices) ** 2)
        exposure_share.append(exposure / prices)
        yield_slope.append(pricing_slope(market.cost_of_capital(), signal))
        premium_slope.append(ols_slope(np.log(prices / market.firms.intrinsic_value), signal))
        return benefit

    market.marginal_financing_saving = wrapped
    started = time.perf_counter()
    try:
        result = market.run()
    except Exception as error:  # a failed world is recorded, never dropped
        return {"spread": spread, "seed": seed, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    ARRAYS.mkdir(exist_ok=True)
    path = ARRAYS / f"s{spread:g}_seed{seed}.npz"
    np.savez_compressed(path, days=np.array(days), rho=np.array(rho), passthrough=np.array(passthrough),
                        exposure_share=np.array(exposure_share), yield_slope=np.array(yield_slope),
                        premium_slope=np.array(premium_slope))
    return {"spread": spread, "seed": seed, "status": "complete", "fingerprint": result.fingerprint(),
            "seconds": time.perf_counter() - started, "decision_days": days,
            "price_cap_hits": int(result.market.price_cap_hits.sum()), "arrays": path.relative_to(HERE).as_posix()}


def check_one(scenario: dict, seed: int) -> dict:
    """Uninstrumented runs at s0: the 1200-day run (its fingerprint must equal the instrumented one) and the
    3500-day B run of the main setting, whose first 1200 days must be bit-identical to the 1200-day run."""
    short = DisclosureMarket(*setting(scenario, CURRENT, seed)).run()
    full = DisclosureMarket(*setting(scenario, CURRENT, seed, full=True)).run()
    n, periods = PRE_POLICY_DAYS + 1, short.actions.shape[0]
    pairs = {"prices": (short.market.prices, full.market.prices[:n]),
             "intrinsic_values": (short.market.intrinsic_values, full.market.intrinsic_values[:n]),
             "true_goodness": (short.true_goodness, full.true_goodness[:n]),
             "inflation": (short.inflation, full.inflation[:n]),
             "financing_rate": (short.financing_rate, full.financing_rate[:n]),
             "public_signal": (short.public_signal, full.public_signal[:n]),
             "actions": (short.actions, full.actions[:periods]),
             "amounts": (short.amounts, full.amounts[:periods])}
    return {"seed": seed, "spread": CURRENT, "fingerprint_uninstrumented": short.fingerprint(),
            "decision_periods_compared": int(periods), "days_compared": int(n),
            "identical_to_full_run": {name: bool(a.shape == b.shape and np.array_equal(a, b)) for name, (a, b) in pairs.items()}}


def pool(workers: int) -> ProcessPoolExecutor:
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"  # inherited by the workers
    disable_power_throttling()
    return ProcessPoolExecutor(max_workers=workers, initializer=disable_power_throttling)


def run_jobs(scenario: dict, jobs: list[tuple[float, int]], workers: int, checks: tuple[int, ...] = ()) -> tuple:
    runs, checked = [], []
    with pool(workers) as executor:
        futures = {executor.submit(check_one, scenario, seed): ("check", seed) for seed in checks}
        futures.update({executor.submit(run_one, scenario, s, seed): (s, seed) for s, seed in jobs})
        for future in as_completed(futures):
            out = future.result()
            (checked if futures[future][0] == "check" else runs).append(out)
            print(futures[future], out.get("status", "check"), round(out.get("seconds", 0), 1), flush=True)
    runs.sort(key=lambda r: (r["spread"], r["seed"]))
    return runs, checked


def load_arrays(run: dict) -> dict:
    with np.load(HERE / run["arrays"]) as data:
        return {name: data[name] for name in data.files}


def summarize(runs: list[dict]) -> list[dict]:
    """Per spread: the median rho over seeds x firms x decision days (contract 26.7) and the diagnostics."""
    table = []
    for spread in sorted({r["spread"] for r in runs}):
        group = [r for r in runs if r["spread"] == spread]
        done = [r for r in group if r["status"] == "complete"]
        arrays = [load_arrays(r) for r in done]
        row = {"spread": spread, "worlds": len(group), "failed_worlds": len(group) - len(done),
               "price_cap_hits": sum(r["price_cap_hits"] for r in done)}
        if arrays:
            rho = np.concatenate([a["rho"].ravel() for a in arrays]) * 1e4
            late = np.concatenate([a["rho"][a["days"] >= 600].ravel() for a in arrays]) * 1e4
            m = np.concatenate([(a["rho"] / a["passthrough"]).ravel() for a in arrays])
            benefit = np.concatenate([(a["exposure_share"] * a["rho"]).ravel() for a in arrays]) * 1e4
            row.update({
                "observations": int(rho.size),
                "rho_bp": {"median": float(np.median(rho)), "p5": float(np.quantile(rho, .05)),
                           "p25": float(np.quantile(rho, .25)), "p75": float(np.quantile(rho, .75)),
                           "p95": float(np.quantile(rho, .95)), "median_days_600_on": float(np.median(late)),
                           "median_by_seed": [float(np.median(a["rho"]) * 1e4) for a in arrays]},
                "passthrough_median": float(np.median(np.concatenate([a["passthrough"].ravel() for a in arrays]))),
                "quote_response_m_median": float(np.median(m)),
                "exposure_share_median": float(np.median(np.concatenate([a["exposure_share"].ravel() for a in arrays]))),
                "benefit_B_bp_of_value_median": float(np.median(benefit)),
                "yield_slope_bp_median": float(np.median(np.concatenate([a["yield_slope"] for a in arrays])) * 1e4),
                "log_price_premium_slope_median": float(np.median(np.concatenate([a["premium_slope"] for a in arrays]))),
            })
        table.append(row)
    return table


def interpolate(table: list[dict], target_bp: float) -> float | None:
    """The spread at which the median rho reaches the target, by linear interpolation of log rho on log s."""
    s = np.array([row["spread"] for row in table if row.get("rho_bp")])
    rho = np.array([row["rho_bp"]["median"] for row in table if row.get("rho_bp")])
    if s.size < 2 or not np.all(np.diff(rho) > 0) or not rho[0] <= target_bp <= rho[-1]:
        return None
    return float(np.exp(np.interp(np.log(target_bp), np.log(rho), np.log(s))))


def round_spread(value: float) -> float:
    return float(f"{value:.4g}")


def grid(args: argparse.Namespace) -> None:
    scenario, experiment_id = frozen_scenario()
    if experiment_id != H3_EXPERIMENT_ID:
        raise SystemExit(f"the H3 spec no longer freezes to {H3_EXPERIMENT_ID[:8]}: source or spec changed")
    jobs = [(s, seed) for s in GRID for seed in SEEDS]
    started = time.strftime("%Y-%m-%d %H:%M:%S %z")
    runs, checks = run_jobs(scenario, jobs, args.workers, checks=(SEEDS[0],))
    check = checks[0]
    instrumented = next(r for r in runs if r["spread"] == CURRENT and r["seed"] == SEEDS[0])
    check["fingerprint_matches_instrumented"] = check["fingerprint_uninstrumented"] == instrumented.get("fingerprint")
    if not check["fingerprint_matches_instrumented"]:
        raise RuntimeError("instrumentation changed the simulated path")
    table = summarize(runs)
    payload = {
        "purpose": "calibration of financing_spread (research contract section 26.7); calibration seeds only",
        "script_sha256": sha(Path(__file__)), "setting": {
            "spec": H3_SPEC.relative_to(ROOT).as_posix(), "spec_sha256_lf": sha(H3_SPEC), "experiment_id": experiment_id,
            "scenario": SCENARIO, "overrides": {"trading_days": PRE_POLICY_DAYS, "burn_in_days": BURN_IN,
                                                "financing_spread": "grid"},
            "domain": scenario["domain"], "market": scenario["config"]},
        "seeds": list(SEEDS), "grid": list(GRID), "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "checks": check, "summary": table,
        "runs": [{k: v for k, v in r.items() if k != "decision_days"} | {"decision_days": len(r.get("decision_days", []))}
                 for r in runs],
        "scope": "Pre-policy pricing and financing only; firm actions, inflation and policy outcomes were not read.",
    }
    (HERE / "calibration_grid.json").write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"checks": check, "summary": [{k: row[k] for k in ("spread", "failed_worlds", "price_cap_hits")}
                      | {"rho_median_bp": row.get("rho_bp", {}).get("median"),
                         "yield_slope_bp": row.get("yield_slope_bp_median")} for row in table]}, indent=1))


def target(args: argparse.Namespace) -> None:
    scenario, experiment_id = frozen_scenario()
    if experiment_id != H3_EXPERIMENT_ID:
        raise SystemExit("the H3 spec no longer freezes to the recorded identity")
    base = json.loads((HERE / "calibration_grid.json").read_text(encoding="utf-8"))
    runs = list(base["runs"])
    wanted = {"s_star": args.pi, "s_lo": args.lo, "s_hi": args.hi, "s_28": TARGET_28}
    wanted = {name: value for name, value in wanted.items() if value is not None and value > 0}
    extensions = []
    while True:  # contract 26.7: double the grid outward while a target lies outside its range
        table = summarize(runs)
        rho = [row["rho_bp"]["median"] for row in table]
        spreads = [row["spread"] for row in table]
        low = [v for v in wanted.values() if v < rho[0]]
        high = [v for v in wanted.values() if v > rho[-1]]
        if not low and not high:
            break
        if len(extensions) >= 6:
            raise SystemExit("targets still outside the grid after six extensions")
        new = [round_spread(spreads[0] / 2)] * bool(low) + [round_spread(spreads[-1] * 2)] * bool(high)
        extra, _ = run_jobs(scenario, [(s, seed) for s in new for seed in SEEDS], args.workers)
        extensions += extra
        runs += extra
    points = {name: interpolate(table, value) for name, value in wanted.items()}
    out = {"targets_bp": wanted, "interpolated": points, "extensions": extensions, "summary_before_verification": table}
    if "s_star" in points:
        first = round_spread(points["s_star"])
        check, _ = run_jobs(scenario, [(first, seed) for seed in SEEDS], args.workers)
        verified = summarize(check)[0]
        median = verified["rho_bp"]["median"] if verified.get("rho_bp") else None
        within = median is not None and abs(median / args.pi - 1) <= TOLERANCE
        out["verification"] = {"spread": first, "rho_median_bp": median, "within_10pct": within, "runs": check,
                               "summary": verified}
        adopted = first
        if not within:  # one re-interpolation with the new point; both recorded
            again = summarize(runs + check)
            second = interpolate(again, args.pi)
            out["reinterpolation"] = {"summary": again, "spread": second}
            adopted = round_spread(second) if second is not None else None
        out["s_star_adopted"] = adopted
        if adopted is None:
            out["rerun_rule_26_9"] = "undecided: the re-interpolation failed (see reinterpolation)"
        elif abs(adopted / CURRENT - 1) >= 0.10:
            out["rerun_rule_26_9"] = "confirmatory rerun at s*"
        else:
            out["rerun_rule_26_9"] = "no rerun: s* within 10% of 0.05; sensitivity block only"
    else:
        out["s_star_adopted"] = None
        out["rerun_rule_26_9"] = "no rerun: the premium estimate is not positive (contract 26.9)"
    out["sensitivity_spreads"] = {"s0": CURRENT, **{name: round_spread(v) for name, v in points.items()
                                                    if name != "s_star" and v is not None}}
    out["script_sha256"] = sha(Path(__file__))
    out["grid_file_sha256"] = sha(HERE / "calibration_grid.json")
    (HERE / "calibration_target.json").write_bytes((json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("targets_bp", "interpolated", "s_star_adopted", "rerun_rule_26_9",
                                          "sensitivity_spreads")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--workers", type=int, default=16)
    t = sub.add_parser("target")
    t.add_argument("--pi", type=float, required=True, help="premium estimate, basis points per SD")
    t.add_argument("--lo", type=float, required=True, help="lower end of its 95%% interval")
    t.add_argument("--hi", type=float, required=True, help="upper end of its 95%% interval")
    t.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    {"grid": grid, "target": target}[args.command](args)


if __name__ == "__main__":
    main()
