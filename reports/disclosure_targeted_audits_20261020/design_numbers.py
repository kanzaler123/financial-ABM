"""Design numbers for contract section 21 (stage E.1, targeted audits), fixed before any implementation or run.

  python design_numbers.py [--stage-f-development <stage F development archive>]

Two sources, nothing else:

1. The firm's financing factors (src/abm/disclosure/financing.py) in the main timing of contract 20.1: the
   real annuity A_r, the wash-signal annuity A_w and the detection discount D at a given audit probability,
   and the common threshold tau*(q) = 1 - sqrt(w) A_r / A_w(q) of proposition 2. A flagged path (the first
   audit after disclosure at q1, later audits at q2) comes from `two_rate_factors`, which reproduces the
   library when q1 = q2 (checked below).
2. Report jumps under random audits in the stage F development archive (seeds 33001-33003, scenarios B and
   decay0_B, policy window): how many firms a jump flag of h = 0.125 SD would mark per audit, and how many of
   them inflated in the period before. The archive is compact, so true values are rebuilt from landed
   improvements and the uniform decay (bounds ignored).

Writes design_numbers.json (LF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist, mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import _no_wmi  # noqa: E402

_no_wmi.prefill_uname()

import numpy as np  # noqa: E402

from abm.disclosure.config import TRADING_DAYS_PER_YEAR  # noqa: E402
from abm.disclosure.financing import financing_signal_factors  # noqa: E402

# Main timing and parameters (contract 20.1, domain csi300_..._fixed_fine_saturated, market of stage F).
DAY, HORIZON, REPRICE, PERIOD, IMPLEMENT, DISCLOSE = 1260, 1260, 60, 60, 120, 20
RATE, W, TAU, Q_BAR, N_FIRMS, CAP = 0.05, 0.2, 0.1, 0.01, 200, 0.25
VALUE_LOG_SD = 0.9151544344205937
H = 0.125  # jump flag: half the per-period cap
DECAY_PER_PERIOD = 0.2 * PERIOD / TRADING_DAYS_PER_YEAR
IMPROVE, INFLATE = 0, 1


def library_factors(q: float) -> tuple[float, float, float]:
    real, _, wash, fine = financing_signal_factors(
        np.array([RATE]), decision_day=DAY, horizon_days=HORIZON, repricing_interval=REPRICE, period_days=PERIOD,
        implementation_delay=IMPLEMENT, disclosure_delay=DISCLOSE, audit_probability=q, transparency=TAU)
    return float(real[0]), float(wash[0]), float(fine[0])


def two_rate_factors(q1: float, q2: float) -> tuple[float, float]:
    """A_w and D when the first audit after disclosure has probability q1 and later audits q2."""
    times = np.arange(1, HORIZON + 1)
    quoted_day = (DAY + times) // REPRICE * REPRICE - 1
    wash_due = DAY + DISCLOSE
    first_audit = ((wash_due + PERIOD - 1) // PERIOD) * PERIOD
    audits = np.maximum(0, (quoted_day - 1 - first_audit) // PERIOD + 1)
    survival = np.where(audits > 0, (1.0 - q1) * (1.0 - q2) ** np.maximum(audits - 1, 0), 1.0)
    survival = survival * (quoted_day >= wash_due)
    discount = np.exp(-times * RATE / TRADING_DAYS_PER_YEAR)
    wash = float(survival @ discount / TRADING_DAYS_PER_YEAR)
    first = np.exp(-RATE * (first_audit - DAY) / TRADING_DAYS_PER_YEAR)
    repeat = np.exp(-RATE * PERIOD / TRADING_DAYS_PER_YEAR)
    fine = float(q1 * first + (1.0 - q1) * first * repeat * q2 / (1.0 - (1.0 - q2) * repeat))
    return wash, fine


def threshold(a_r: float, a_w: float) -> float:
    return 1.0 - np.sqrt(W) * a_r / a_w


def bisect(function, low: float, high: float, steps: int = 80) -> float:
    f_low = function(low)
    for _ in range(steps):
        middle = 0.5 * (low + high)
        f_middle = function(middle)
        if (f_middle > 0) == (f_low > 0):
            low, f_low = middle, f_middle
        else:
            high = middle
    return 0.5 * (low + high)


def factor_tables() -> dict:
    a_r, a_w_bar, d_bar = library_factors(Q_BAR)
    for q in (0.0, 0.004, 0.01, 0.055, 0.3):
        _, a_w, d = library_factors(q)
        a_w2, d2 = two_rate_factors(q, q)
        assert abs(a_w - a_w2) < 1e-12 and abs(d - d2) < 1e-12, (q, a_w, a_w2, d, d2)
    by_rate = []
    for q in (0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.105, 0.2, 0.5, 1.0):
        _, a_w, d = library_factors(q)
        by_rate.append({"q": q, "A_w": round(a_w, 4), "D": round(d, 4), "tau_star": round(threshold(a_r, a_w), 4)})
    stop = {f"{tau}": round(bisect(lambda q: threshold(a_r, library_factors(q)[1]) - tau, 1e-6, 1.0), 4)
            for tau in (0.1, 0.3, 0.6)}

    # Value rule: q_i = q_bar * P_i / mean(P), at quantiles of the initial value distribution (mean 1).
    normal = NormalDist()
    blom = lambda rank: normal.inv_cdf((rank - 0.375) / (N_FIRMS + 0.25))  # noqa: E731
    points = {"smallest of 200": blom(1), "small-class median (30th pct)": normal.inv_cdf(0.30),
              "small/mid cut (60th pct)": normal.inv_cdf(0.60), "mid/large cut (90th pct)": normal.inv_cdf(0.90),
              "large-class median (95th pct)": normal.inv_cdf(0.95), "largest of 200": blom(N_FIRMS)}
    value_rule = []
    for name, z in points.items():
        relative = float(np.exp(VALUE_LOG_SD * z - 0.5 * VALUE_LOG_SD**2))
        q = min(1.0, Q_BAR * relative)
        _, a_w, d = library_factors(q)
        value_rule.append({
            "point": name, "value_over_mean": round(relative, 3), "q": round(q, 5), "A_w": round(a_w, 4),
            "D": round(d, 4), "tau_star": round(threshold(a_r, a_w), 4),
            # Expected fixed-fine term (F/P) D relative to an average-value firm under random audits.
            "fine_term_random": round(1.0 / relative, 3),
            "fine_term_value_rule": round(d / d_bar / relative, 3),
        })

    # Jump rule: half the expected audits at random (q_bar / 2 each), half shared by the flagged firms.
    q_rest = Q_BAR / 2
    a_w_rest, d_rest = two_rate_factors(q_rest, q_rest)
    jump_rule = []
    for flagged in (1, 2, 5, 10, 20, 40):
        q_flag = min(1.0, q_rest + N_FIRMS * Q_BAR / 2 / flagged)
        a_w, d = two_rate_factors(q_flag, q_rest)
        jump_rule.append({"flagged_firms": flagged, "q_flagged": round(q_flag, 4), "A_w_flagged": round(a_w, 4),
                          "D_flagged": round(d, 4), "annuity_ratio_g": round(a_w / a_w_rest, 4),
                          "tau_star_flagged": round(threshold(a_r, a_w), 4)})
    # A firm whose unconstrained inflation x (at the unflagged rate) exceeds the room under the flag, x_h,
    # keeps under it rather than accept the flag iff 2 x x_h - x_h^2 >= (g x)^2 (interior, no new fixed fine),
    # i.e. iff g <= sqrt(2 x x_h - x_h^2) / x. Shown for a cap-bound firm (x = 0.25).
    smoothing = [{"x": CAP, "x_h": x_h, "g_critical": round(float(np.sqrt(2 * CAP * x_h - x_h**2) / CAP), 4)}
                 for x_h in (0.10, 0.15, 0.17)]

    _, a_w_two, d_two = library_factors(2 * Q_BAR)
    return {
        "A_r": round(a_r, 4), "A_w_random": round(a_w_bar, 4), "D_random": round(d_bar, 4),
        "tau_star_random": round(threshold(a_r, a_w_bar), 4),
        "by_audit_rate": by_rate, "rate_at_which_tau_star_equals": stop,
        "value_rule": value_rule,
        "jump_rule": {"q_unflagged": q_rest, "A_w_unflagged": round(a_w_rest, 4), "D_unflagged": round(d_rest, 4),
                      "tau_star_unflagged": round(threshold(a_r, a_w_rest), 4), "by_flagged_count": jump_rule,
                      "smoothing_condition": smoothing},
        "double_random": {"q": 2 * Q_BAR, "A_w": round(a_w_two, 4), "D": round(d_two, 4),
                          "tau_star": round(threshold(a_r, a_w_two), 4)},
    }


def archive_jumps(archive: Path) -> dict:
    out = {}
    for scenario, decay in (("B", DECAY_PER_PERIOD), ("decay0_B", 0.0)):
        rows, inflations, improvements = [], [], []
        for seed in (33001, 33002, 33003):
            with np.load(archive / scenario / f"seed_{seed}" / "attempt_001" / "result.npz") as z:
                actions, amounts, inflation = z["actions"], z["amounts"], z["inflation"]
                size = z["market_size_class"]
            for k in range(21, 59):  # audits on days 1260..3480
                day = PERIOD * k
                landed = np.where(actions[k - 2] == IMPROVE, amounts[k - 2], 0.0)  # decided 120 days before
                jump = landed - decay + inflation[day - 1] - inflation[day - PERIOD]
                inflated = actions[k - 1] == INFLATE
                flag = jump > H
                open_case = inflation[day - 1] > 0
                rows.append({"flagged": int(flag.sum()), "flagged_after_inflating": int((flag & inflated).sum()),
                             "inflated_last_period": int(inflated.sum()),
                             "inflated_unflagged": int((inflated & ~flag).sum()),
                             "improvement_landed": int((landed > 0).sum()), "open_cases": int(open_case.sum()),
                             **{f"open_cases_{name}": int((open_case & (size == index)).sum())
                                for index, name in enumerate(("large", "mid", "small"))}})
            window = np.arange(21, 59)
            inflations.append(amounts[window][actions[window] == INFLATE])
            improvements.append(amounts[window][actions[window] == IMPROVE])
        inflate_amounts, improve_amounts = np.concatenate(inflations), np.concatenate(improvements)
        out[scenario] = {
            "per_audit_mean": {key: round(mean(row[key] for row in rows), 2) for key in rows[0]},
            "inflation_decisions": int(inflate_amounts.size),
            "inflation_amount_mean": round(float(inflate_amounts.mean()), 4),
            "inflation_share_at_cap": round(float(np.mean(inflate_amounts >= CAP - 1e-4)), 3),
            "inflation_share_above_h_plus_decay": round(float(np.mean(inflate_amounts > H + decay)), 3),
            "improvement_decisions": int(improve_amounts.size),
            "improvement_amount_mean": round(float(improve_amounts.mean()), 4),
            "improvement_share_above_h": round(float(np.mean(improve_amounts > H)), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f-development", type=Path,
                        default=ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1")
    args = parser.parse_args()
    payload = {
        "_note": "Contract section 21 design numbers; main timing of contract 20.1; jump flag h = 0.125 SD.",
        "factors": factor_tables(),
        "random_audit_jumps_stage_f_development": archive_jumps(args.stage_f_development),
    }
    (HERE / "design_numbers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
