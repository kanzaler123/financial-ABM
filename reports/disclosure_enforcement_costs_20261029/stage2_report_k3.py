"""Supplement to estimate_k3.py stage2 (contract 28.24): the surrogate check of contract 28.6 (b) and the fit at
theta_hat. It reads the same archive and changes no estimate.

The check: per moment, the five-fold cross-validated RMSE of the quadratic (stage2_k3.json, at the sigma_m grid
value nearest sigma_m_hat) against 1.5 times the simulated SD of a 10-seed mean. That SD is the square root of the
seed-bootstrap variance of the pooled moments (moments_k3.seed_bootstrap_cov, as in K3a), averaged over the design
points at the same sigma_m.

    python -I -X utf8 stage2_report_k3.py <stage 2 archive> <stage 2 spec> <stage2_k3.json>

Writes <stage2_k3 name>_report.json here.
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
    archive, spec_path, est_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    est = json.loads(est_path.read_text(encoding="utf-8"))
    mk = ek._load("moments_k3")
    m_data, Omega_data, W, cuts = ek.data_target()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    sigma = est["cv_rmse_sigma_grid_value"]
    names = [sc["name"] for sc in spec["scenarios"] if sc["name"] not in est["points_dropped_undefined"]]
    sds, M = [], []
    for name in names:
        stats = ek.point_stats(archive, name, spec["seeds"], sigma, mk, cuts)
        M.append(mk.pooled_model_moments(stats))
        sds.append(np.sqrt(np.diag(mk.seed_bootstrap_cov(stats))))
    M, sim_sd = np.array(M), np.array(sds).mean(0)
    cv = np.array(est["cv_rmse"])
    se = np.sqrt(np.diag(Omega_data))
    rows, U, Mall = ek.design_moments(archive, spec_path)
    keep = np.isfinite(Mall).all(axis=(1, 2))
    surface = ek.Surface(U[keep], Mall[keep], ek.SIGMAS)
    m_hat = surface(np.array(est["u_hat"]), est["sigma_m_hat"])
    i = names.index(est["best_design_point"])
    g_hat, g_best = m_data - m_hat, m_data - M[i]
    out = {"contract": "28.24", "estimate": est_path.name, "sigma_m_grid_value": sigma,
           "sim_sd_10_seed_mean": sim_sd.tolist(), "cv_rmse": cv.tolist(), "cv_over_sim_sd": (cv / sim_sd).tolist(),
           "surrogate_ok": bool((cv <= 1.5 * sim_sd).all()),
           "moments_data": m_data.tolist(), "data_se": se.tolist(),
           "surrogate_at_theta_hat": m_hat.tolist(), "g_over_data_se_theta_hat": (g_hat / se).tolist(),
           "J_share_theta_hat": (g_hat ** 2 * np.diag(W) / float(g_hat @ W @ g_hat)).tolist(),
           "best_design_point": est["best_design_point"], "moments_best_design": M[i].tolist(),
           "g_over_data_se_best_design": (g_best / se).tolist(), "J_best_design_at_sigma": float(g_best @ W @ g_best)}
    (HERE / f"{est_path.stem}_report.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("cv_over_sim_sd", "surrogate_ok", "g_over_data_se_theta_hat",
                                          "J_share_theta_hat", "g_over_data_se_best_design",
                                          "J_best_design_at_sigma")}, indent=1))


if __name__ == "__main__":
    main()
