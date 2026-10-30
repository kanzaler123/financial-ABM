"""Supplement to estimate_k3.py stage1 (contract 28.21): J quantiles over the points with finite J, the ten best
design points, and the per-moment fit at theta_1. It reads the same archive and changes no estimate.

    python -I -X utf8 stage1_report_k3.py <stage 1 archive> <stage 1 spec>

Writes stage1_report_k3.json here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimate_k3 as ek  # noqa: E402


def main() -> None:
    archive, spec_path = Path(sys.argv[1]), Path(sys.argv[2])
    m_data, Omega_data, W, _ = ek.data_target()
    rows, U, M = ek.design_moments(archive, spec_path)
    names = ek.identified_names()
    J = ek.criterion(m_data, W, M)                          # (points, sigmas)
    Jmin = J.min(1)
    finite = np.isfinite(Jmin)
    order = np.argsort(np.where(np.isnan(Jmin), np.inf, Jmin))
    i, s = np.unravel_index(int(np.nanargmin(J)), J.shape)
    se = np.sqrt(np.diag(Omega_data))
    g = m_data - M[i, s]
    out = {"contract": "28.21", "points": len(rows), "points_J_finite": int(finite.sum()),
           "points_J_inf": [rows[j] for j in np.flatnonzero(np.isinf(Jmin))],
           "points_J_nan": [rows[j] for j in np.flatnonzero(np.isnan(Jmin))],
           "J_quantiles_finite_0_10_25_50": np.quantile(Jmin[finite], [0, 0.1, 0.25, 0.5]).tolist(),
           "best10": [{"point": rows[j], "u": U[j].tolist(), "theta": ek.to_theta(U[j], names),
                       "sigma_m": float(ek.SIGMAS[int(np.nanargmin(J[j]))]), "J": float(Jmin[j])} for j in order[:10]],
           "best_point": rows[i], "sigma_m_best": float(ek.SIGMAS[s]), "J_best_by_sigma": J[i].tolist(),
           "moments_data": m_data.tolist(), "moments_best": M[i, s].tolist(), "data_se": se.tolist(),
           "g_over_data_se": (g / se).tolist(), "J_share_by_moment": (g ** 2 * np.diag(W) / J[i, s]).tolist(),
           # reach of the design: per moment, range over all points at sigma_m = 0 and where the data value falls
           "design_range_sigma0": {"min": np.nanmin(np.where(np.isfinite(M[:, 0]), M[:, 0], np.nan), 0).tolist(),
                                   "max": np.nanmax(np.where(np.isfinite(M[:, 0]), M[:, 0], np.nan), 0).tolist(),
                                   "share_points_above_data": (M[:, 0] > m_data).mean(0).tolist()},
           "m9_by_point_sigma0": dict(zip(rows, M[:, 0, 8].tolist()))}
    (HERE / "stage1_report_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("points_J_finite", "points_J_inf", "points_J_nan",
                                          "J_quantiles_finite_0_10_25_50", "g_over_data_se", "J_share_by_moment")},
                     indent=1))
    for b in out["best10"]:
        print(b["point"], round(b["J"], 1), b["sigma_m"], np.round(b["u"], 3).tolist())


if __name__ == "__main__":
    main()
