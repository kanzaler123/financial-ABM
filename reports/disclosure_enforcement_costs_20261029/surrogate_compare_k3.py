"""Diagnostic of contract 28.27: the recentred quadratic surrogate at the first local estimate, against the first
surrogate's J there and the nearest recentred design point. Reads the archive; changes no estimate.

    python -I -X utf8 surrogate_compare_k3.py <stage2r archive> <stage2r spec>
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
    m_data, _, W, _ = ek.data_target()
    e1 = json.loads((HERE / "stage2_k3.json").read_text(encoding="utf-8"))
    e2 = json.loads((HERE / "stage2r_k3.json").read_text(encoding="utf-8"))
    _, U, M = ek.design_moments(archive, spec_path)
    surface = ek.Surface(U, M, ek.SIGMAS)
    u1, u2 = np.array(e1["u_hat"]), np.array(e2["u_hat"])
    J = ek.criterion(m_data, W, M)[:, 0]
    d = np.abs(U - u1).max(1)
    i = int(np.argmin(d))
    print(json.dumps({"J_surrogate1_at_u1": e1["J_hat"],
                      "J_surrogate2_at_u1": float(ek.criterion(m_data, W, surface(u1, e1["sigma_m_hat"]))),
                      "J_surrogate2_at_u2": float(ek.criterion(m_data, W, surface(u2, e2["sigma_m_hat"]))),
                      "nearest_design_point_max_abs_du": float(d[i]), "nearest_design_point_J_sigma0": float(J[i])},
                     indent=1))


if __name__ == "__main__":
    main()
