"""Stage K3: identification diagnostics and simulated minimum distance estimation of the cost parameters (contract 28.5-28.6).

The estimator works in unit coordinates u in [0, 1]^k: each parameter's range of contract 28.5 maps to [0, 1],
linearly or (for c) on the log scale, so "half the range" in the identification rule is 0.5 in every coordinate.
The model reader adds measurement noise sigma_m to the sampled public signal in post-processing (stream
SeedSequence([seed, 2504]), the same as stage H) and computes m1-m9 with moments_k3.py.

  python estimate_k3.py identify <K3a archive>      # contract 28.5: Jacobian at theta_0 and the identification rule

The later steps (28.6 (a)-(e)) add subcommands here before their specs are frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- parameters ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    name: str
    theta0: float
    low: float
    high: float
    log: bool = False

    def to_unit(self, x: float) -> float:
        if self.log:
            return (math.log(x) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (x - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        if self.log:
            return math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        return self.low + u * (self.high - self.low)

    def scale(self, u: float) -> float:
        """d theta / d u at u (for the delta method)."""
        if self.log:
            return self.from_unit(u) * (math.log(self.high) - math.log(self.low))
        return self.high - self.low


PARAMS = (                                      # contract 28.5
    Param("kappa_T", 0.505, 0.1, 2.0),
    Param("s_P", 0.423, 0.1, 0.8),
    Param("c", 0.05, 0.0125, 0.2, log=True),
    Param("eta", 1.5, 0.0, 3.0),
    Param("psi", 1.0, 0.0, 2.0),
    Param("w", 0.2, 0.05, 0.8),
)
SIZE_MULTIPLIERS_0 = (0.6, 1.0, 1.6)
SIGMAS = np.round(np.arange(0.0, 0.6001, 0.05), 2)          # sigma_m grid, contract 25.2
NOISE_STREAM = 2504
FD_STEP = 0.10                                              # contract 28.5: 10% of each range
CI_RULE = 0.5                                               # 95% interval at most half the range
FINAL_SEEDS = 30


def domain_changes(theta: dict[str, float], *, q: float, fine: float) -> dict:
    """Scenario domain overrides for one parameter point (H3 B otherwise)."""
    psi = theta["psi"]
    return {"natural_decay": 0.0, "natural_reversion": round(theta["kappa_T"], 6),
            "reversion_permanent_share": round(theta["s_P"], 6), "real_cost_scale": round(theta["c"], 8),
            "cost_level_elasticity": round(theta["eta"], 6),
            "size_cost_multipliers": [round(m ** psi, 8) for m in SIZE_MULTIPLIERS_0],
            "wash_cost_ratio": round(theta["w"], 6), "baseline_audit_probability": q, "baseline_fixed_fine": fine}


# -- inference ----------------------------------------------------------------------------------------------------

def sandwich(G: np.ndarray, W: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """(G'WG)^-1 G'W Omega W G (G'WG)^-1 (contract 25.5)."""
    bread = np.linalg.pinv(G.T @ W @ G)
    return bread @ G.T @ W @ Omega @ W @ G @ bread


def identification_rule(G: np.ndarray, W: np.ndarray, Omega: np.ndarray, names: list[str]) -> dict:
    """Contract 28.5: drop the parameter with the largest SE / range while any 95% interval exceeds half its range."""
    keep = list(range(G.shape[1]))
    steps = []
    while keep:
        cov = sandwich(G[:, keep], W, Omega)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        width = 2 * 1.96 * se
        steps.append({"parameters": [names[j] for j in keep], "se_unit": se.tolist(), "ci_width_unit": width.tolist()})
        if (width <= CI_RULE).all():
            break
        keep.pop(int(np.argmax(se)))
    s = np.linalg.svd(np.sqrt(W) @ G, compute_uv=False)
    full = sandwich(G, W, Omega)
    sd = np.sqrt(np.clip(np.diag(full), 1e-300, None))
    corr = full / np.outer(sd, sd)
    pairs = [[names[i], names[j], float(corr[i, j])] for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(corr[i, j]) > 0.9]
    return {"identified": [names[j] for j in keep], "dropped": [n for n in names if n not in [names[j] for j in keep]],
            "steps": steps, "singular_values_weighted": s.tolist(),
            "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else math.inf, "pairs_abs_corr_above_0p9": pairs}


def weighted_chi2_pvalue(T: float, G: np.ndarray, W: np.ndarray, Omega: np.ndarray, draws: int = 1_000_000,
                         seed: int = 28907) -> float:
    """p-value of T = g'Wg under a weighted chi-square with weights eig(Omega^1/2 P'WP Omega^1/2) (contract 25.5)."""
    P = np.eye(G.shape[0]) - G @ np.linalg.pinv(G.T @ W @ G) @ G.T @ W
    vals, vecs = np.linalg.eigh(Omega)
    root = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    lam = np.clip(np.linalg.eigvalsh(root @ P.T @ W @ P @ root), 0, None)
    rng = np.random.default_rng(seed)
    total = np.zeros(draws)
    for value in lam:
        total += value * rng.chisquare(1, draws)
    return float((total >= T).mean())


# -- quadratic surrogate (contract 28.6 (b)) -----------------------------------------------------------------------

def quad_features(U: np.ndarray) -> np.ndarray:
    U = np.atleast_2d(U)
    k = U.shape[1]
    cols = [np.ones(len(U))] + [U[:, i] for i in range(k)]
    cols += [U[:, i] * U[:, j] for i in range(k) for j in range(i, k)]
    return np.column_stack(cols)


class Quadratic:
    """Per moment, a full second-order polynomial in u fitted by least squares to seed-mean moments."""

    def __init__(self, U: np.ndarray, Y: np.ndarray):
        self.coef, *_ = np.linalg.lstsq(quad_features(U), Y, rcond=None)

    def __call__(self, U: np.ndarray) -> np.ndarray:
        return quad_features(U) @ self.coef

    @staticmethod
    def cv_rmse(U: np.ndarray, Y: np.ndarray, folds: int = 5, seed: int = 28905) -> np.ndarray:
        order = np.random.default_rng([seed, 5]).permutation(len(U))
        err = np.empty_like(Y)
        for f in range(folds):
            test = order[f::folds]
            train = np.setdiff1d(order, test)
            err[test] = Quadratic(U[train], Y[train])(U[test]) - Y[test]
        return np.sqrt((err ** 2).mean(0))


def sobol(k: int, n: int, seed: int, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> np.ndarray:
    """Scrambled Sobol points in the box [lo, hi] of unit coordinates (contract 28.6 (a), (b))."""
    import warnings

    from scipy.stats import qmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # n need not be a power of two
        U = qmc.Sobol(d=k, scramble=True, seed=seed).random(n)
    lo = np.zeros(k) if lo is None else lo
    hi = np.ones(k) if hi is None else hi
    return lo + U * (hi - lo)


def criterion(m_data: np.ndarray, W: np.ndarray, m: np.ndarray) -> np.ndarray:
    g = m_data - m
    return np.einsum("...i,ij,...j->...", g, W, g)


def stage1_pick(U: np.ndarray, M: np.ndarray, m_data: np.ndarray, W: np.ndarray) -> tuple[int, int, float]:
    """M: (points, sigmas, moments) seed-mean moments. The design point and sigma_m grid value with the smallest J."""
    J = criterion(m_data, W, M)
    i, s = np.unravel_index(int(np.nanargmin(J)), J.shape)
    return int(i), int(s), float(J[i, s])


def local_box(u: np.ndarray, half: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(u - half, 0.0, 1.0), np.clip(u + half, 0.0, 1.0)


class Surface:
    """Quadratic in u for every sigma_m grid value, cubic spline across sigma_m (contract 28.6 (b))."""

    def __init__(self, U: np.ndarray, M: np.ndarray, sigmas: np.ndarray):
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.fits = [Quadratic(U, M[:, s]) for s in range(len(self.sigmas))]

    def __call__(self, u: np.ndarray, sigma: float) -> np.ndarray:
        values = np.array([f(u[None, :])[0] for f in self.fits])
        if len(self.sigmas) == 1:
            return values[0]
        from scipy.interpolate import CubicSpline
        return CubicSpline(self.sigmas, values, axis=0)(sigma)

    def jacobian(self, u: np.ndarray, sigma: float, lo: np.ndarray, hi: np.ndarray, with_sigma: bool) -> np.ndarray:
        cols = []
        for j in range(len(u)):
            h = 1e-3
            up, dn = u.copy(), u.copy()
            up[j], dn[j] = min(u[j] + h, hi[j]), max(u[j] - h, lo[j])
            cols.append((self(up, sigma) - self(dn, sigma)) / (up[j] - dn[j]))
        if with_sigma:
            h = 1e-3 * (self.sigmas[-1] - self.sigmas[0])
            s_up, s_dn = min(sigma + h, self.sigmas[-1]), max(sigma - h, self.sigmas[0])
            cols.append((self(u, s_up) - self(u, s_dn)) / (s_up - s_dn))
        return np.column_stack(cols)


def minimise_surface(surface: Surface, starts: np.ndarray, lo: np.ndarray, hi: np.ndarray, m_data: np.ndarray,
                     W: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Multi-start L-BFGS-B over (u, sigma_m) inside the local box (contract 28.6 (b)). starts: rows (u, sigma)."""
    from scipy.optimize import minimize
    k = len(lo)
    bounds = list(zip(lo, hi)) + [(surface.sigmas[0], surface.sigmas[-1])]
    best = None
    for x0 in starts:
        res = minimize(lambda x: float(criterion(m_data, W, surface(x[:k], x[k]))), x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x[:k], float(best.x[k]), float(best.fun)


def on_box_edge(u: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float = 1e-4) -> dict[str, list[int]]:
    """Coordinates on the local box's edge but inside the range, and those on the range's edge."""
    at = (np.abs(u - lo) < tol) | (np.abs(u - hi) < tol)
    at_range = (u < tol) | (u > 1 - tol)
    return {"local_edge": np.flatnonzero(at & ~at_range).tolist(), "range_edge": np.flatnonzero(at_range).tolist()}


def inference(surface: Surface, u: np.ndarray, sigma: float, lo: np.ndarray, hi: np.ndarray, m_data: np.ndarray,
              W: np.ndarray, Omega: np.ndarray, m_final: np.ndarray, with_sigma: bool, mc_draws: int = 1_000_000) -> dict:
    """Contract 28.6 (c): sandwich SEs in unit coordinates, fit test with the final-stage moments."""
    G = surface.jacobian(u, sigma, lo, hi, with_sigma)
    cov = sandwich(G, W, Omega)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    g = m_data - m_final
    T = float(g @ W @ g)
    band = 1.96 * np.sqrt(np.diag(Omega))
    return {"G": G.tolist(), "cov_unit": cov.tolist(), "se_unit": se.tolist(), "T": T,
            "p_value": weighted_chi2_pvalue(T, G, W, Omega, draws=mc_draws), "g": g.tolist(), "band": band.tolist(),
            "in_band": (np.abs(g) <= band).tolist()}


def central_jacobian(moments_at, u0: np.ndarray, step: float = FD_STEP) -> np.ndarray:
    """Columns d m / d u_j by central differences; moments_at(j, sign) returns the moments at u0 +- step e_j."""
    return np.column_stack([(moments_at(j, +1) - moments_at(j, -1)) / (2 * step) for j in range(len(u0))])


# -- model archives ---------------------------------------------------------------------------------------------

def finished(archive: Path, scenario: str, seed: int) -> dict[str, np.ndarray]:
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*"), reverse=True):
        manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] == "complete":
            with np.load(attempt / "result.npz") as saved:
                return {"signal": saved["sampled_public_signal"].copy(), "size_class": saved["market_size_class"].copy(),
                        "detections": saved["detections"].copy(), "price_cap_hits": saved["market_price_cap_hits"].copy()}
    raise FileNotFoundError(f"no completed attempt for {scenario} seed {seed}")


def noise(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, NOISE_STREAM])).standard_normal(shape)


def point_stats(archive: Path, scenario: str, seeds: list[int], sigma: float, mk, cuts) -> list:
    out = []
    for seed in seeds:
        run = finished(archive, scenario, seed)
        signal = run["signal"] + sigma * noise(seed, run["signal"].shape)
        out.append(mk.model_stats(signal, run["size_class"], run["detections"], cuts))
    return out


def data_target():
    d = json.loads((HERE / "data_moments_k3.json").read_text(encoding="utf-8"))
    m = np.array(d["moments"])
    Omega = np.array(d["bootstrap_cov"])
    W = np.diag(1 / np.diag(Omega))
    return m, Omega, W, np.array(d["inputs"]["notch_cuts"])


def identify(archive: Path) -> dict:
    """Contract 28.5 at theta_0 with sigma_m = 0."""
    mk = _load("moments_k3")
    spec_mod = _load("build_spec_k3")
    m_data, Omega_data, W, cuts = data_target()
    seeds = spec_mod.SEEDS["k3a"]
    base = point_stats(archive, "theta0", seeds, 0.0, mk, cuts)
    m0 = mk.pooled_model_moments(base)
    cov10 = mk.seed_bootstrap_cov(base)                       # Omega_sim / 10
    Omega = Omega_data + cov10 * len(seeds) / FINAL_SEEDS     # Omega_data + Omega_sim / 30
    names = [p.name for p in PARAMS]

    def at(j, sign):
        name = spec_mod.fd_name(names[j], sign)
        return mk.pooled_model_moments(point_stats(archive, name, seeds, 0.0, mk, cuts))

    G = central_jacobian(at, np.array([p.to_unit(p.theta0) for p in PARAMS]))
    rule = identification_rule(G, W, Omega, names)
    return {"contract": "28.5", "archive": archive.as_posix(), "seeds": seeds, "moments_theta0": m0.tolist(),
            "jacobian_unit": G.tolist(), "jacobian_over_data_se": (G / np.sqrt(np.diag(Omega_data))[:, None]).tolist(),
            "omega_sim_over_30_diag": (cov10 * len(seeds) / FINAL_SEEDS).diagonal().tolist(), **rule}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identify")
    p.add_argument("archive", type=Path)
    args = ap.parse_args()
    if args.cmd == "identify":
        out = identify(args.archive)
        (HERE / "identification_k3a.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("identified", "dropped", "condition_number", "pairs_abs_corr_above_0p9")}))


if __name__ == "__main__":
    main()
