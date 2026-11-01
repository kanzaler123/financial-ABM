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


# -- 28.6 (a)-(e) on the archives ----------------------------------------------------------------------------

def design_moments(archive: Path, spec_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Scenario names, their unit coordinates over the identified parameters, and moments (points, sigmas, 9)."""
    mk = _load("moments_k3")
    names = identified_names()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _, _, _, cuts = data_target()
    rows, U, M = [], [], []
    for sc in spec["scenarios"]:
        theta = theta_from_domain(sc["domain"])
        U.append([next(p for p in PARAMS if p.name == n).to_unit(theta[n]) for n in names])
        M.append([mk.pooled_model_moments(point_stats(archive, sc["name"], spec["seeds"], float(sig), mk, cuts))
                  for sig in SIGMAS])
        rows.append(sc["name"])
    return rows, np.array(U), np.array(M)


def identified_names() -> list[str]:
    return json.loads((HERE / "identification_k3a.json").read_text(encoding="utf-8"))["identified"]


def theta_from_domain(d: dict) -> dict[str, float]:
    psi = math.log(d["size_cost_multipliers"][2]) / math.log(SIZE_MULTIPLIERS_0[2])
    return {"kappa_T": d["natural_reversion"], "s_P": d["reversion_permanent_share"], "c": d["real_cost_scale"],
            "eta": d["cost_level_elasticity"], "psi": psi, "w": d["wash_cost_ratio"]}


def to_theta(u: np.ndarray, names: list[str]) -> dict[str, float]:
    theta = {p.name: p.theta0 for p in PARAMS}
    for n, x in zip(names, u):
        theta[n] = next(p for p in PARAMS if p.name == n).from_unit(float(x))
    return theta


def stage1(archive: Path, spec_path: Path) -> dict:
    m_data, _, W, _ = data_target()
    rows, U, M = design_moments(archive, spec_path)
    i, s, J = stage1_pick(U, M, m_data, W)
    names = identified_names()
    return {"contract": "28.6 (a)", "parameters": names, "points": len(rows), "best_point": rows[i],
            "u_best": U[i].tolist(), "theta_best": to_theta(U[i], names), "sigma_m_best": float(SIGMAS[s]), "J_best": J,
            "J_quantiles_0_10_50": np.quantile(criterion(m_data, W, M).min(1), [0, 0.1, 0.5]).tolist()}


def stage2(archive: Path, spec_path: Path, centre: np.ndarray) -> dict:
    m_data, _, W, _ = data_target()
    rows, U, M = design_moments(archive, spec_path)
    names = identified_names()
    lo, hi = local_box(centre)
    # contract 28.23: points with any moment undefined (m9 when a size group has no detection) leave the fit
    keep = np.isfinite(M).all(axis=(1, 2))
    dropped = [r for r, k in zip(rows, keep) if not k]
    rows, U, M = [r for r, k in zip(rows, keep) if k], U[keep], M[keep]
    surface = Surface(U, M, SIGMAS)
    J = criterion(m_data, W, M)
    best = np.argsort(J.min(1))[:10]
    starts = np.column_stack([U[best], SIGMAS[J[best].argmin(1)]])
    u, sigma, J_hat = minimise_surface(surface, starts, lo, hi, m_data, W)
    s_near = int(np.argmin(np.abs(SIGMAS - sigma)))
    cv = Quadratic.cv_rmse(U, M[:, s_near])
    return {"contract": "28.6 (b)", "parameters": names, "points": len(rows), "points_dropped_undefined": dropped,
            "centre": centre.tolist(),
            "box_lo": lo.tolist(), "box_hi": hi.tolist(), "u_hat": u.tolist(), "theta_hat": to_theta(u, names),
            "sigma_m_hat": sigma, "J_hat": J_hat, "edge": on_box_edge(u, lo, hi),
            "sigma_m_on_range_edge": bool(sigma <= SIGMAS[0] + 1e-4 or sigma >= SIGMAS[-1] - 1e-4),
            "cv_rmse_sigma_grid_value": float(SIGMAS[s_near]), "cv_rmse": cv.tolist(),
            "best_design_point": rows[int(best[0])], "best_design_J": float(J[best[0]].min())}


def local_surface(local_archive: Path, local_spec: Path, est: dict) -> tuple[Surface, np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray]:
    """The quadratic surrogate of the last local estimate (same points, same drop rule as stage2), its box, and the
    design's unit coordinates and moments."""
    rows, U, M = design_moments(local_archive, local_spec)
    keep = np.isfinite(M).all(axis=(1, 2))
    U, M = U[keep], M[keep]
    return Surface(U, M, SIGMAS), np.array(est["box_lo"]), np.array(est["box_hi"]), U, M


def run_point(spec_path: Path) -> tuple[dict, np.ndarray, list[int]]:
    """theta and unit coordinates of the single scenario of a final or recovery spec (theta_hat to three decimals)."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    (sc,) = spec["scenarios"]
    theta = theta_from_domain(sc["domain"])
    u = np.array([next(p for p in PARAMS if p.name == n).to_unit(theta[n]) for n in identified_names()])
    return theta, u, spec["seeds"]


def profile(surface: Surface, j: int, lo: np.ndarray, hi: np.ndarray, sigma_free: bool, sigma: float,
            m_data: np.ndarray, Omega_inv: np.ndarray, starts: np.ndarray, points: int = 61) -> dict:
    """Contract 25.5 / 28.6 (c): J* = g' Omega^-1 g profiled over parameter j on 61 points of the local box, the
    others (and sigma_m if free) minimised inside the box; the range with J*_prof - min J* <= 3.84."""
    from scipy.optimize import minimize
    k = len(lo)
    grid = np.linspace(lo[j], hi[j], points)
    others = [i for i in range(k) if i != j]
    on_grid = np.flatnonzero(np.isclose(surface.sigmas, sigma))
    # sigma_m fixed on a grid value: the spline across sigma_m passes through that grid value's quadratic exactly
    fixed = surface.fits[int(on_grid[0])] if (not sigma_free and len(on_grid)) else None
    out = []
    for x in grid:
        def f(z):
            u = np.empty(k)
            u[j] = x
            u[others] = z[:k - 1]
            m = fixed(u[None, :])[0] if fixed is not None else surface(u, z[k - 1] if sigma_free else sigma)
            return float(criterion(m_data, Omega_inv, m))
        bounds = [(lo[i], hi[i]) for i in others] + ([(surface.sigmas[0], surface.sigmas[-1])] if sigma_free else [])
        best = math.inf
        for s in starts:
            z0 = np.append(s[others], s[k]) if sigma_free else s[others]
            best = min(best, minimize(f, z0, method="L-BFGS-B", bounds=bounds).fun)
        out.append(best)
    out = np.array(out)
    inside = grid[out - out.min() <= 3.84]
    return {"grid_unit": grid.tolist(), "J_star": out.tolist(), "range_unit": [float(inside.min()), float(inside.max())],
            "touches_box_edge": bool(inside.min() <= lo[j] + 1e-9 or inside.max() >= hi[j] - 1e-9)}


def unit_to_theta_interval(name: str, lo_u: float, hi_u: float) -> list[float]:
    p = next(p for p in PARAMS if p.name == name)
    return [p.from_unit(min(max(lo_u, 0.0), 1.0)), p.from_unit(min(max(hi_u, 0.0), 1.0))]


def final(local_archive: Path, local_spec: Path, est_path: Path, final_archive: Path, final_spec: Path,
          mc_draws: int = 1_000_000) -> dict:
    """Contract 28.6 (c) at theta_hat (three decimals): Omega = Omega_data + Omega_sim/30 (seed bootstrap of the 30
    seeds, as in K3a); G from the local surrogate; sigma_m enters G only when its estimate is inside its grid
    (contract 28.16: at sigma_m = 0 its derivative is 0); identification re-checked by the rule of 28.5; profile
    intervals for every parameter."""
    mk = _load("moments_k3")
    est = json.loads(est_path.read_text(encoding="utf-8"))
    names = identified_names()
    m_data, Omega_data, W, cuts = data_target()
    surface, lo, hi, U, M = local_surface(local_archive, local_spec, est)
    theta, u, seeds = run_point(final_spec)
    sigma = float(est["sigma_m_hat"])
    with_sigma = not est["sigma_m_on_range_edge"]
    stats = point_stats(final_archive, "theta_hat", seeds, sigma, mk, cuts)
    m_final = mk.pooled_model_moments(stats)
    cov_sim = mk.seed_bootstrap_cov(stats)                    # Omega_sim / 30
    Omega = Omega_data + cov_sim
    inf = inference(surface, u, sigma, lo, hi, m_data, W, Omega, m_final, with_sigma, mc_draws=mc_draws)
    G = np.array(inf["G"])[:, :len(names)]
    se_u = np.array(inf["se_unit"])[:len(names)]
    params = {}
    for j, n in enumerate(names):
        p = next(p for p in PARAMS if p.name == n)
        se = float(se_u[j] * p.scale(u[j]))
        params[n] = {"estimate": theta[n], "u": float(u[j]), "se": se, "ci95": [theta[n] - 1.96 * se, theta[n] + 1.96 * se],
                     "se_unit": float(se_u[j]), "ci_width_unit": float(2 * 1.96 * se_u[j])}
    rule = identification_rule(G, W, Omega, names)
    J_all = criterion(m_data, W, M)
    best = np.argsort(np.nanmin(J_all, 1))[:10]
    starts = np.vstack([np.append(u, sigma), np.column_stack([U[best], SIGMAS[np.nanargmin(J_all[best], 1)]])])
    Omega_inv = np.linalg.inv(Omega)
    profiles = {}
    for j, n in enumerate(names):
        pr = profile(surface, j, lo, hi, with_sigma, sigma, m_data, Omega_inv, starts)
        pr["range_theta"] = unit_to_theta_interval(n, *pr["range_unit"])
        profiles[n] = pr
    weak = rule["dropped"]
    return {"contract": "28.6 (c)", "estimate_from": est_path.name, "theta_hat": theta, "u_hat": u.tolist(),
            "sigma_m": sigma, "sigma_m_in_G": with_sigma, "seeds": seeds, "moments_final": m_final.tolist(),
            "moments_data": m_data.tolist(), "omega_sim_over_30_diag": np.diag(cov_sim).tolist(),
            "parameters": params, "T": inf["T"], "p_value": inf["p_value"], "g": inf["g"], "band": inf["band"],
            "in_band": inf["in_band"], "g_over_data_se": (np.array(inf["g"]) / np.sqrt(np.diag(Omega_data))).tolist(),
            "G": inf["G"], "cov_unit": inf["cov_unit"], "identification_at_theta_hat": rule, "weakly_identified": weak,
            "profiles": profiles, "local_box": {"lo": lo.tolist(), "hi": hi.tolist()}}


def recovery(local_archive: Path, local_spec: Path, est_path: Path, rec_archive: Path, rec_spec: Path,
             final_path: Path) -> dict:
    """Contract 28.6 (d): three groups of 10 seeds at theta_hat as pseudo-data; Omega_data replaced by the group's
    seed-bootstrap covariance (W its inverse diagonal); the same local surrogate and estimator. "Recovered" when every
    parameter of every group lies inside the main 95% interval."""
    mk = _load("moments_k3")
    est = json.loads(est_path.read_text(encoding="utf-8"))
    fin = json.loads(final_path.read_text(encoding="utf-8"))
    names = identified_names()
    _, _, _, cuts = data_target()
    surface, lo, hi, U, M = local_surface(local_archive, local_spec, est)
    theta, u, seeds = run_point(rec_spec)
    sigma = float(est["sigma_m_hat"])
    groups = []
    for g in range(3):
        stats = point_stats(rec_archive, "theta_hat", seeds[10 * g:10 * g + 10], sigma, mk, cuts)
        m_pseudo = mk.pooled_model_moments(stats)
        W_g = np.diag(1 / np.diag(mk.seed_bootstrap_cov(stats)))
        J = criterion(m_pseudo, W_g, M)
        best = np.argsort(np.nanmin(J, 1))[:10]
        starts = np.column_stack([U[best], SIGMAS[np.nanargmin(J[best], 1)]])
        u_g, s_g, J_g = minimise_surface(surface, starts, lo, hi, m_pseudo, W_g)
        th = to_theta(u_g, names)
        inside = {n: bool(fin["parameters"][n]["ci95"][0] <= th[n] <= fin["parameters"][n]["ci95"][1]) for n in names}
        groups.append({"seeds": seeds[10 * g:10 * g + 10], "moments": m_pseudo.tolist(), "theta": th,
                       "u": u_g.tolist(), "sigma_m": s_g, "J": J_g, "inside_main_ci": inside,
                       "bias": {n: th[n] - fin["theta_hat"][n] for n in names}})
    return {"contract": "28.6 (d)", "theta_hat": theta, "groups": groups,
            "recovered": all(all(g["inside_main_ci"].values()) for g in groups)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identify")
    p.add_argument("archive", type=Path)
    p = sub.add_parser("stage1")
    p.add_argument("archive", type=Path)
    p.add_argument("spec", type=Path)
    p = sub.add_parser("stage2")
    p.add_argument("archive", type=Path)
    p.add_argument("spec", type=Path)
    p.add_argument("--centre-from", type=Path, required=True, help="stage1_k3.json (u_best) or stage2_k3.json (u_hat)")
    for cmd, extra in (("final", ("final_archive", "final_spec")), ("recovery", ("recovery_archive", "recovery_spec"))):
        p = sub.add_parser(cmd)
        p.add_argument("local_archive", type=Path)
        p.add_argument("local_spec", type=Path)
        p.add_argument("estimate", type=Path, help="stage2r_k3.json (or stage2_k3.json without recentring)")
        for name in extra:
            p.add_argument(name, type=Path)
    args = ap.parse_args()
    if args.cmd == "final":
        out = final(args.local_archive, args.local_spec, args.estimate, args.final_archive, args.final_spec)
        (HERE / "final_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("theta_hat", "T", "p_value", "in_band", "weakly_identified")}, indent=1))
        return
    if args.cmd == "recovery":
        out = recovery(args.local_archive, args.local_spec, args.estimate, args.recovery_archive, args.recovery_spec,
                       HERE / "final_k3.json")
        (HERE / "recovery_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({"recovered": out["recovered"], "groups": [g["theta"] for g in out["groups"]]}, indent=1))
        return
    if args.cmd == "stage1":
        out = stage1(args.archive, args.spec)
        (HERE / "stage1_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(out, indent=1))
        return
    if args.cmd == "stage2":
        prev = json.loads(args.centre_from.read_text(encoding="utf-8"))
        out = stage2(args.archive, args.spec, np.array(prev.get("u_hat", prev.get("u_best"))))
        name = "stage2_k3.json" if args.centre_from.name == "stage1_k3.json" else "stage2r_k3.json"
        (HERE / name).write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("theta_hat", "sigma_m_hat", "J_hat", "edge", "best_design_J")}, indent=1))
        return
    if args.cmd == "identify":
        out = identify(args.archive)
        (HERE / "identification_k3a.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("identified", "dropped", "condition_number", "pairs_abs_corr_above_0p9")}))


if __name__ == "__main__":
    main()
