"""Self-test of the K3 estimator on a statistical stand-in for the model (contract 28.7).

    python -I -X utf8 selftest_k3.py

The stand-in maps six parameters in unit coordinates to nine moments with a known smooth function: linear and
quadratic terms and one interaction; parameter 6 has zero derivative for every moment, and parameters 3 and 4 move
the moments in nearly the same direction (column correlation about 0.97). Data standard errors are those of
data_moments_k3.json; a simulated "seed" adds noise 4.5 times the data SE (200 firms against about 4,000), with a
seed component common to every parameter point (common random numbers) and a point component. The test runs the
identification rule of 28.5 at u0 = 0.5, then 28.6 (a)-(d) on the identified parameters, with the same functions and
design sizes (32k and 16k Sobol points, 5 and 10 seeds, 30 final seeds, three recovery groups of 10).

Passing (contract 28.7): the rule drops parameter 6, and the 95% interval of every kept parameter covers its true
value. Every random number comes from fixed integers (seed 28903 and streams derived from it); no hash().
Writes selftest_k3.json (LF).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("estimate_k3", HERE / "estimate_k3.py")
ek = importlib.util.module_from_spec(spec)
sys.modules["estimate_k3"] = ek
spec.loader.exec_module(ek)

SEED = 28903
K, NM = 6, 9
NOISE_RATIO = 4.5


class StandIn:
    def __init__(self, se: np.ndarray):
        rng = np.random.default_rng([SEED, 1])
        self.se = se
        scale = 40 * se                                     # effect over the full range: about 40 data SEs
        B = rng.normal(0, 1, (NM, K))
        B[:, 3] = 0.95 * B[:, 2] + 0.3 * rng.normal(0, 1, NM)
        B[:, 5] = 0.0
        C = 0.3 * rng.normal(0, 1, (NM, K))
        C[:, 5] = 0.0
        self.B, self.C = B * scale[:, None], C * scale[:, None]
        self.D = 0.2 * rng.normal(0, 1, NM) * scale
        self.a = rng.normal(0, 1, NM) * scale
        self.column_corr_3_4 = float(np.corrcoef(B[:, 2], B[:, 3])[0, 1])

    def mean(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        return self.a + u @ self.B.T + ((u - 0.5) ** 2) @ self.C.T + np.outer(u[:, 0] * u[:, 1], self.D)

    def seeds(self, u: np.ndarray, seeds: list[int], point: int) -> np.ndarray:
        """Per-seed moments at one point, shape (seeds, moments)."""
        out = []
        for s in seeds:
            common = np.random.default_rng([SEED, 2, s]).normal(0, 1, NM)
            own = np.random.default_rng([SEED, 3, s, point]).normal(0, 1, NM)
            out.append(self.mean(u)[0] + NOISE_RATIO * self.se * (0.8 * common + 0.6 * own))
        return np.array(out)


def full_u(u_kept: np.ndarray, kept: list[int], u0: np.ndarray) -> np.ndarray:
    u = u0.copy()
    u[kept] = u_kept
    return u


def main() -> None:
    d = json.loads((HERE / "data_moments_k3.json").read_text(encoding="utf-8"))
    se = np.array(d["bootstrap_se"])
    Omega_data = np.diag(se ** 2)
    W = np.diag(1 / se ** 2)
    model = StandIn(se)
    rng = np.random.default_rng([SEED, 4])
    u_true = np.round(rng.uniform(0.3, 0.7, K), 3)
    m_data = model.mean(u_true)[0] + se * rng.normal(0, 1, NM)
    u0 = np.full(K, 0.5)
    names = [f"p{j + 1}" for j in range(K)]
    out = {"seed": SEED, "u_true": u_true.tolist(), "column_corr_p3_p4": model.column_corr_3_4}
    point = iter(range(10**6))

    # 28.5: Jacobian at u0, 10 seeds, central differences of 0.1.
    seeds10 = list(range(1, 11))
    base = model.seeds(u0, seeds10, next(point))
    cov_sim = np.cov(base, rowvar=False)
    Omega = Omega_data + cov_sim / 30

    def at(j, sign):
        u = u0.copy()
        u[j] += sign * ek.FD_STEP
        return model.seeds(u, seeds10, next(point)).mean(0)

    G = ek.central_jacobian(at, u0)
    rule = ek.identification_rule(G, W, Omega, names)
    out["identification"] = {k: rule[k] for k in ("identified", "dropped", "condition_number", "pairs_abs_corr_above_0p9")}
    kept = [names.index(n) for n in rule["identified"]]
    k = len(kept)

    # 28.6 (a): global design.
    U1 = ek.sobol(k, 32 * k, 28904)
    M1 = np.array([model.seeds(full_u(u, kept, u0), list(range(101, 106)), next(point)).mean(0) for u in U1])[:, None, :]
    i1, _, J1 = ek.stage1_pick(U1, M1, m_data, W)
    centre = U1[i1]

    # 28.6 (b): local design, quadratic surface, at most one recentring.
    recentred = False
    for attempt in range(2):
        lo, hi = ek.local_box(centre)
        U2 = ek.sobol(k, 16 * k, 28905, lo, hi)
        per = np.array([model.seeds(full_u(u, kept, u0), list(range(201, 211)), next(point)) for u in U2])
        M2 = per.mean(1)
        surface = ek.Surface(U2, M2[:, None, :], np.array([0.0]))
        J2 = ek.criterion(m_data, W, M2)
        starts = np.column_stack([U2[np.argsort(J2)[:10]], np.zeros(10)])
        u_hat, _, J_hat = ek.minimise_surface(surface, starts, lo, hi, m_data, W)
        edge = ek.on_box_edge(u_hat, lo, hi)
        if edge["local_edge"] and attempt == 0:
            centre, recentred = u_hat, True
            continue
        break
    cv = ek.Quadratic.cv_rmse(U2, M2)
    sim_sd = per.std(1, ddof=1).mean(0) / np.sqrt(per.shape[1])
    out["local"] = {"recentred": recentred, "edge": edge, "J_stage1": J1, "J_hat": J_hat,
                    "cv_rmse_over_sim_sd": (cv / sim_sd).tolist(), "surrogate_ok": bool((cv <= 1.5 * sim_sd).all())}

    # 28.6 (c): final stage, 30 seeds.
    final = model.seeds(full_u(u_hat, kept, u0), list(range(301, 331)), next(point))
    Omega_f = Omega_data + np.cov(final, rowvar=False) / 30
    inf = ek.inference(surface, u_hat, 0.0, np.zeros(k), np.ones(k), m_data, W, Omega_f, final.mean(0),
                       with_sigma=False, mc_draws=200_000)
    se_u = np.array(inf["se_unit"])
    cover = [bool(abs(u_hat[j] - u_true[kept[j]]) <= 1.96 * se_u[j]) for j in range(k)]
    out["final"] = {"u_hat": u_hat.tolist(), "se_unit": se_u.tolist(), "covers_true": cover, "T": inf["T"],
                    "p_value": inf["p_value"]}

    # 28.6 (d): recovery, three pseudo-datasets of 10 seeds at u_hat, same local design and surface.
    rec = []
    for g in range(3):
        pseudo = model.seeds(full_u(u_hat, kept, u0), list(range(401 + 10 * g, 411 + 10 * g)), next(point))
        W_g = np.diag(1 / (pseudo.var(0, ddof=1) / 10))
        u_g, _, _ = ek.minimise_surface(surface, starts, lo, hi, pseudo.mean(0), W_g)
        rec.append({"u": u_g.tolist(), "inside_main_ci": bool((np.abs(u_g - u_hat) <= 1.96 * se_u).all())})
    out["recovery"] = rec
    out["passed"] = bool("p6" in rule["dropped"] and all(cover))
    (HERE / "selftest_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k_: out[k_] for k_ in ("identification", "final", "passed")}, indent=1))
    print("recovery inside:", [r["inside_main_ci"] for r in rec], "surrogate ok:", out["local"]["surrogate_ok"])


if __name__ == "__main__":
    main()
