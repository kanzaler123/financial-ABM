"""Design numbers for the analytic prediction of contract 25.8 (T1 of contract 24.6 at the stage H estimate).

  python design_numbers.py

In the main timing of contract 20.1 (decision day 1260, horizon 1260 days, repricing every 60 days, audits every
60 days at q = 0.01, implementation delay 120, disclosure delay 20, discount rate 0.05) the library's financing
factors give the wash-signal annuity A_w and the real annuity A_r, the latter with the reverting share of a real
change removed (real_persistence). Under Q improvements revert at kappa_T, so A_r(Q) = A_r(kappa_T); the permanent
level does not enter. Under the perm_ block improvements stay, so A_r is the non-reverting one.

With w = 0.2: tau* = 1 - sqrt(w) A_r / A_w (proposition 1), and with X = A_w^2 / w and a = A_r^2 the size threshold
ratio P*_F / P*_I = 4 (0.16 X - a) / (0.81 X - a) (contract 24.6, the same new-case probability in I and F). The
contract 24.6 values of A_r at kappa = 0.32 and 0.50 are reproduced as a check.

Writes design_numbers.json (LF).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from abm.disclosure.domain import load_domain  # noqa: E402
from abm.disclosure.financing import financing_signal_factors  # noqa: E402
from abm.disclosure.game import real_persistence  # noqa: E402

DAY, HORIZON, REPRICE, PERIOD, IMPLEMENT, DISCLOSE = 1260, 1260, 60, 60, 120, 20
RATE, W, Q_BAR, FIRMS = 0.05, 0.2, 0.01, 200
DOMAIN = load_domain(ROOT / "configs" / "domains" / "csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json")
KAPPA_T, SHARE = 0.505, 0.423   # contract 25.16


def factors(kappa: float | None) -> tuple[float, float]:
    persistence = None
    if kappa is not None:
        persistence = real_persistence(replace(DOMAIN, natural_decay=0.0, natural_reversion=kappa), FIRMS, HORIZON)
    real, _, wash, _ = financing_signal_factors(
        np.array([RATE]), decision_day=DAY, horizon_days=HORIZON, repricing_interval=REPRICE, period_days=PERIOD,
        implementation_delay=IMPLEMENT, disclosure_delay=DISCLOSE, audit_probability=Q_BAR, transparency=0.1,
        real_persistence=persistence)
    return float(real[0]), float(wash[0])


def thresholds(real: float, wash: float) -> dict:
    x, a = wash**2 / W, real**2
    # With tau = 0.6 at or above tau*, no firm inflates under I at an interior solution and the ratio is undefined.
    ratio = 4.0 * (0.16 * x - a) / (0.81 * x - a) if 0.16 * x > a else None
    return {"A_r": real, "tau_star": 1.0 - math.sqrt(W) * real / wash, "I_tau_below_tau_star": 0.16 * x > a,
            "size_threshold_ratio_F_over_I": ratio}


def main() -> None:
    plain_real, wash = factors(None)
    q_real, q_wash = factors(KAPPA_T)
    assert q_wash == wash
    checks = {f"{kappa:g}": round(factors(kappa)[0], 4) for kappa in (0.32, 0.5)}
    assert (round(plain_real, 4), round(wash, 4), checks) == (3.7258, 3.8305, {"0.32": 2.0077, "0.5": 1.51}), checks
    # The permanent share does not change a firm's improvement persistence.
    q_domain = replace(DOMAIN, natural_decay=0.0, natural_reversion=KAPPA_T, reversion_permanent_share=SHARE)
    assert np.array_equal(real_persistence(q_domain, FIRMS, HORIZON),
                          real_persistence(replace(q_domain, reversion_permanent_share=0.0), FIRMS, HORIZON))
    assert real_persistence(replace(q_domain, reversion_improvement="permanent"), FIRMS, HORIZON) is None
    out = {
        "timing": {"decision_day": DAY, "horizon_days": HORIZON, "repricing_interval": REPRICE, "period_days": PERIOD,
                   "implementation_delay": IMPLEMENT, "disclosure_delay": DISCLOSE, "discount_rate": RATE,
                   "audit_probability": Q_BAR, "n_firms": FIRMS, "wash_cost_ratio": W},
        "A_w": wash,
        "main_Q": {"kappa_T": KAPPA_T, "reversion_permanent_share": SHARE, **thresholds(q_real, wash)},
        "perm": {"note": "improvements stay; A_r without reversion", **thresholds(plain_real, wash)},
        "check_contract_24_6_A_r": checks,
        "T1": "In the same state and at an interior solution, a firm that inflates under I (tau 0.6, fine f) also "
              "inflates under F (tau 0.1, fine 4f), and one that inflates under F also inflates under B, for any A_r. "
              "Judged on the main block: the paired intervals of the policy-window washing-decision share for I - F "
              "and F - B both lie below 0.",
    }
    (HERE / "design_numbers.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
