"""Design numbers for contract section 22 (stage E.2, assurance providers), fixed before any implementation or run.

  python design_numbers.py [--stage-f-development <stage F development archive>]

Three sources, nothing else:

1. The firm's financing factors in the main timing of contract 20.1 with a pre-publication assurance check:
   every period, on the disclosure day (decision day + 20), the assurer checks the report about to be published
   (old undetected inflation plus the new amount) and with probability a has all of it corrected before
   publication, without a fine. `assured_factors` is a standalone implementation of the real annuity A_r, the
   wash-signal annuity A_w and the regulator detection discount D under audit probability q and assurance
   probability a; at a = 0 it reproduces the library (checked below).
2. Proposition 1 and 2 thresholds with assurance: tau*(a) = 1 - sqrt(w) A_r / A_w(a), and the fixed-fine size
   threshold P*(a), proportional to R(a) = D(a) / [(1 - tau)^2 A_w(a)^2 / w - A_r^2].
3. Inflation present on engagement days under the baseline in the stage F development archive (seeds
   33001-33003, scenarios B and decay0_B): the share s of firms whose report would carry inflation when checked,
   which enters the assurers' best response a* = min(1, L q s / (2 kappa)).

Money: the statutory fixed fine CNY 10 million is 4.5614908e-05 model units (domain baseline_fixed_fine).

Writes design_numbers.json (LF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

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
RATE, W, TAU, Q_BAR = 0.05, 0.2, 0.1, 0.01
TEN_MILLION = 4.5614908036639594e-05  # CNY 10 million in model units
WAN = TEN_MILLION / 1000.0  # CNY 10,000
FEE_PER_YEAR = 50 * WAN  # assumed assurance fee, CNY 0.5 million a year
FEE = FEE_PER_YEAR * PERIOD / TRADING_DAYS_PER_YEAR  # per engagement (one per 60-day period)
REFERENCE_QUALITY = 0.5  # the fee covers the effort cost of a = 0.5
KAPPA = FEE / REFERENCE_QUALITY**2  # effort cost kappa a^2 per engagement
STATUTORY_LIABILITY = 500 * WAN  # Securities Law art. 213: up to 10x business income of CNY 0.5 million
LIABILITY_DOSES = (0, 1, 4, 10, 16, 100)
IMPROVE, INFLATE = 0, 1


def library_factors(q: float) -> tuple[float, float, float, float]:
    real, displaced, wash, fine = financing_signal_factors(
        np.array([RATE]), decision_day=DAY, horizon_days=HORIZON, repricing_interval=REPRICE, period_days=PERIOD,
        implementation_delay=IMPLEMENT, disclosure_delay=DISCLOSE, audit_probability=q, transparency=TAU)
    return float(real[0]), float(displaced[0]), float(wash[0]), float(fine[0])


def assured_factors(q: float, a: float) -> tuple[float, float, float, float]:
    """A_r, displaced annuity, A_w and D with audits at probability q and assurance checks at probability a.

    Audits fall on multiples of PERIOD after trading; a quote precedes the audit on its own day. Assurance
    checks fall on disclosure days (DAY + DISCLOSE + k PERIOD) before publication, so a quote on a check day
    already sees the corrected report. The first check is the new report's own pre-publication check.
    """
    times = np.arange(1, HORIZON + 1)
    quoted_day = (DAY + times) // REPRICE * REPRICE - 1
    real_due, wash_due = DAY + max(1, IMPLEMENT), DAY + max(1, DISCLOSE)
    first_audit = ((wash_due + PERIOD - 1) // PERIOD) * PERIOD
    audits = np.maximum(0, (quoted_day - 1 - first_audit) // PERIOD + 1)
    checks = np.where(quoted_day >= wash_due, (quoted_day - wash_due) // PERIOD + 1, 0)
    survival = (quoted_day >= wash_due) * (1.0 - q) ** audits * (1.0 - a) ** checks
    discount = np.exp(-times * RATE / TRADING_DAYS_PER_YEAR)
    real_visible = quoted_day >= real_due
    real = float(real_visible @ discount / TRADING_DAYS_PER_YEAR)
    wash = float(survival @ discount / TRADING_DAYS_PER_YEAR)
    # Old stock displaced by a real action: audits after the decision day, checks from the first disclosure day.
    old_audits = np.maximum(0, (quoted_day - 1) // PERIOD - DAY // PERIOD)
    old = (1.0 - q) ** old_audits * (1.0 - a) ** checks
    displaced = real - (1.0 - TAU) * float((real_visible * old) @ discount / TRADING_DAYS_PER_YEAR)
    # Regulator detection: audit j (day first_audit + j PERIOD) finds the report if it survived j audits and
    # every check before that day (checks on wash_due + k PERIOD < audit day).
    fine = 0.0
    for j in range(100000):
        day = first_audit + j * PERIOD
        alive = (1.0 - q) ** j * (1.0 - a) ** ((day - 1 - wash_due) // PERIOD + 1)
        term = q * alive * np.exp(-RATE * (day - DAY) / TRADING_DAYS_PER_YEAR)
        fine += term
        if term < 1e-16 and j > 10:
            break
    return real, displaced, wash, fine


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
    for q in (0.0, 0.01, 0.05, 0.3):
        library, own = library_factors(q), assured_factors(q, 0.0)
        assert all(abs(x - y) < 1e-12 for x, y in zip(library, own)), (q, library, own)
    a_r, _, a_w0, d0 = assured_factors(Q_BAR, 0.0)
    margin0 = (1 - TAU) ** 2 * a_w0**2 / W - a_r**2
    rows = []
    for a in (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        _, displaced, a_w, d = assured_factors(Q_BAR, a)
        margin = (1 - TAU) ** 2 * a_w**2 / W - a_r**2
        rows.append({
            "a": a, "A_w": round(a_w, 4), "D": round(d, 4), "displaced_annuity": round(displaced, 4),
            "tau_star": round(threshold(a_r, a_w), 4) if a_w > 0 else None,
            # Fixed-fine size threshold P*(a) / P*(0); None once no firm inflates at tau = 0.1.
            "P_star_ratio": round(d / margin * margin0 / d0, 4) if margin > 0 else None,
        })
    stop = {f"{tau}": round(bisect(lambda a: threshold(a_r, assured_factors(Q_BAR, a)[2]) - tau, 0.0, 1.0), 4)
            for tau in (0.1, 0.3)}
    # Where P*(a) / P*(0) is lowest, i.e. where assurance most lowers the size above which fined firms inflate.
    grid = np.linspace(0.0, stop["0.1"] - 1e-4, 400)
    ratios = []
    for a in grid:
        _, _, a_w, d = assured_factors(Q_BAR, float(a))
        ratios.append(d / ((1 - TAU) ** 2 * a_w**2 / W - a_r**2) * margin0 / d0)
    lowest = int(np.argmin(ratios))

    def ratio_minus_one(a: float) -> float:
        _, _, a_w, d = assured_factors(Q_BAR, a)
        return d / ((1 - TAU) ** 2 * a_w**2 / W - a_r**2) * margin0 / d0 - 1.0
    return {
        "A_r": round(a_r, 4), "A_w_unassured": round(a_w0, 4), "D_unassured": round(d0, 4),
        "tau_star_unassured": round(threshold(a_r, a_w0), 4),
        "by_assurance_probability": rows,
        "assurance_probability_at_which_tau_star_equals": stop,
        "P_star_ratio_minimum": {"a": round(float(grid[lowest]), 4), "ratio": round(float(ratios[lowest]), 4)},
        "P_star_ratio_rises_monotonically": bool(np.all(np.diff(ratios) >= -1e-12)),
        "P_star_ratio_back_to_one_at": round(bisect(ratio_minus_one, float(grid[lowest]), stop["0.1"] - 1e-4), 4),
    }


def best_response_table() -> dict:
    lam = {f"{dose}x": round(dose * STATUTORY_LIABILITY * Q_BAR / (2 * KAPPA), 4) for dose in LIABILITY_DOSES}
    return {
        "fee_per_engagement": FEE, "fee_per_engagement_cny_10k": round(FEE / WAN, 3),
        "kappa": KAPPA, "kappa_cny_10k": round(KAPPA / WAN, 3),
        "statutory_liability": STATUTORY_LIABILITY,
        "slope_lambda_by_dose": lam,
        "_note": "a* = min(1, lambda s): s is the share of the assurer's engagements whose report carried inflation.",
    }


def archive_shares(archive: Path) -> dict:
    out = {}
    names = ("large", "mid", "small")
    for scenario in ("B", "decay0_B"):
        engagement, switch, washing = [], [], []
        by_size = {name: [] for name in names}
        for seed in (33001, 33002, 33003):
            with np.load(archive / scenario / f"seed_{seed}" / "attempt_001" / "result.npz") as z:
                inflation, actions, size = z["inflation"], z["actions"], z["market_size_class"]
            switch.append(float(np.mean(inflation[1200] > 0)))
            for k in range(21, 58):  # decisions on days 1260..3420, checked on day + 20
                day = PERIOD * k + DISCLOSE
                present = inflation[day] > 0  # after that day's disclosure: old stock plus new amount
                engagement.append(float(present.mean()))
                washing.append(float(np.mean(actions[k] == INFLATE)))
                for index, name in enumerate(names):
                    by_size[name].append(float(present[size == index].mean()))
        out[scenario] = {
            "share_with_inflation_at_switch_day_1200": round(mean(switch), 4),
            "share_with_inflation_on_engagement_days": round(mean(engagement), 4),
            "share_with_inflation_on_engagement_days_by_size": {k: round(mean(v), 4) for k, v in by_size.items()},
            "share_of_decisions_inflating": round(mean(washing), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-f-development", type=Path,
                        default=ROOT / "reports" / "disclosure_stage_f_20261014" / "development_v1")
    args = parser.parse_args()
    payload = {
        "_note": "Contract section 22 design numbers; main timing of contract 20.1; pre-publication assurance checks.",
        "factors": factor_tables(),
        "assurer_best_response": best_response_table(),
        "baseline_inflation_shares_stage_f_development": archive_shares(args.stage_f_development),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (HERE / "design_numbers.json").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
