"""Stage K3: identification diagnostics and simulated minimum distance estimation of the cost parameters (contract 28.5-28.6).

The estimator works in unit coordinates u in [0, 1]^k: each parameter's range of contract 28.5 maps to [0, 1],
linearly or (for c) on the log scale, so "half the range" in the identification rule is 0.5 in every coordinate.
The model reader adds measurement noise sigma_m to the sampled public signal in post-processing (stream
SeedSequence([seed, 2504]), the same as stage H) and computes m1-m9 with moments_k3.py.

  python estimate_k3.py identify <K3a archive>      # contract 28.5: Jacobian at theta_0 and the identification rule

The later steps (28.6 (a)-(e)) add subcommands here before their specs are frozen; contract 28.37 adds

  python estimate_k3.py sizeaudit <archive> <spec> [--check]   # the size-dependent audit scan at theta_hat
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


def yearly_rates(archive: Path, scenario: str, seeds: list[int], mk) -> np.ndarray:
    return np.array([mk.yearly_detection_rate(finished(archive, scenario, s)["detections"]) for s in seeds])


def qw_set(qw_archive: Path, qw_spec: Path, final_archive: Path, final_spec: Path) -> dict:
    """Contract 28.6 (e): the model's yearly detection rate (all firms, eight blocks; seed mean, equal firm counts)
    on the (q, w) grid at theta_hat, against K1's main T1-or-T2 rate and its 95% interval; (0.01, w_hat) from the
    final-stage runs. Also the ESG-specific bound of 28.2 (v): the one-sided 95% Clopper-Pearson upper limit of the
    ESG-specific T1-or-T2 firm-year rate over the main years."""
    from scipy.stats import beta
    mk = _load("moments_k3")
    k1 = json.loads((HERE / "k1_estimates.json").read_text(encoding="utf-8"))
    main = k1["main"]["pooled"]["T1_T2"]
    y0, y1 = k1["main_years"]
    esg = {(e["Scode"], e["date"][:4]) for e in k1["esg_events"]
           if e["tier"] in ("T1", "T2") and y0 <= int(e["date"][:4]) <= y1}
    n = main["firm_years"]
    esg_upper = float(beta.ppf(0.95, len(esg) + 1, n - len(esg)))
    spec = json.loads(qw_spec.read_text(encoding="utf-8"))
    rows = []
    for sc in spec["scenarios"]:
        d = sc["domain"]
        r = yearly_rates(qw_archive, sc["name"], spec["seeds"], mk)
        rows.append({"q": d["baseline_audit_probability"], "w": d["wash_cost_ratio"], "rate": float(r.mean()),
                     "rate_se": float(r.std(ddof=1) / math.sqrt(len(r))), "per_seed": r.tolist(),
                     "inside_k1_ci": bool(main["ci95"][0] <= r.mean() <= main["ci95"][1]),
                     "below_esg_upper": bool(r.mean() <= esg_upper)})
    theta, _, seeds = run_point(final_spec)
    r = yearly_rates(final_archive, "theta_hat", seeds, mk)
    hat = {"q": 0.01, "w": theta["w"], "rate": float(r.mean()), "rate_se": float(r.std(ddof=1) / math.sqrt(len(r))),
           "inside_k1_ci": bool(main["ci95"][0] <= r.mean() <= main["ci95"][1])}
    return {"contract": "28.6 (e)", "theta_hat": theta, "k1_rate": main["rate"], "k1_ci95": main["ci95"],
            "esg_events_main_years": sorted(map(list, esg)), "firm_years": n, "esg_upper_95": esg_upper,
            "grid": rows, "set_inside_k1_ci": [[x["q"], x["w"]] for x in rows if x["inside_k1_ci"]],
            "set_below_esg_upper": [[x["q"], x["w"]] for x in rows if x["below_esg_upper"]],
            "theta_hat_point": hat, "theta_hat_point_in_set": hat["inside_k1_ci"]}


def load_arrays(archive: Path, scenario: str, seed: int, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Named arrays of a run's completed attempt (finished() keeps its own fixed set)."""
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*"), reverse=True):
        manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] == "complete":
            with np.load(attempt / "result.npz") as saved:
                return {k: saved[k].copy() for k in keys}
    raise FileNotFoundError(f"no completed attempt for {scenario} seed {seed}")


def qscan(archive: Path, spec_path: Path) -> dict:
    """Contract 28.34: the model's yearly detection rate (as in 28.6 (e)) along q at fixed w_hat; per q a 95% t
    interval over the 30 seeds, classed against K1's main T1-or-T2 interval; monotonicity; interpolated crossings on
    rising steps only; the yearly audit probability; a cross-check of q = 0.010 against the final stage. Descriptive
    by-products: size-group rates and m9; the share of firm-days with inflation; the share of audits that found
    inflation; m1-m9 at sigma_m = 0 and J."""
    from scipy.stats import t as student_t
    mk = _load("moments_k3")
    k1 = json.loads((HERE / "k1_estimates.json").read_text(encoding="utf-8"))["main"]["pooled"]["T1_T2"]
    lo_k1, hi_k1 = k1["ci95"]
    m_data, _, W, cuts = data_target()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    seeds = spec["seeds"]
    tq = float(student_t.ppf(0.975, len(seeds) - 1))
    groups = {"large": 0, "mid": 1, "small": 2}
    points, without_audit = [], 0
    for sc in spec["scenarios"]:
        q = sc["domain"]["baseline_audit_probability"]
        rates = yearly_rates(archive, sc["name"], seeds, mk)
        mean, se = float(rates.mean()), float(rates.std(ddof=1) / math.sqrt(len(rates)))
        ci = [mean - tq * se, mean + tq * se]
        cls = "clearly_below" if ci[1] < lo_k1 else "clearly_above" if ci[0] > hi_k1 else "cannot_exclude"
        stats = point_stats(archive, sc["name"], seeds, 0.0, mk, cuts)
        cs, ys, cl, yl = np.concatenate([s.ratio for s in stats]).sum(0)
        m = mk.pooled_model_moments(stats)
        infl = {g: [0, 0] for g in ("all", *groups)}
        aud = {g: [0, 0] for g in ("all", *groups)}
        for seed in seeds:
            a = load_arrays(archive, sc["name"], seed, ("inflation", "audit_selected", "detections", "market_size_class"))
            days = np.zeros(a["inflation"].shape[0], dtype=bool)
            days[mk.BLOCK_START:mk.BLOCK_START + mk.BLOCKS * mk.BLOCK_DAYS] = True
            pdays = np.arange(a["detections"].shape[0]) * 60
            periods = (pdays >= mk.BLOCK_START) & ((pdays - mk.BLOCK_START) // mk.BLOCK_DAYS < mk.BLOCKS)
            inflating = a["inflation"][days] > 0
            selected = a["audit_selected"][periods] == 1
            found = a["detections"][periods] > 0
            without_audit += int((found & ~selected).sum())
            for g in infl:
                cols = np.ones(a["market_size_class"].size, dtype=bool) if g == "all" else a["market_size_class"] == groups[g]
                infl[g][0] += int(inflating[:, cols].sum())
                infl[g][1] += int(inflating[:, cols].size)
                aud[g][0] += int((found & selected)[:, cols].sum())
                aud[g][1] += int(selected[:, cols].sum())
        points.append({
            "q": q, "yearly_audit_probability": 1 - (1 - q) ** (252 / 60), "rate": mean, "rate_se": se, "ci95": ci,
            "class": cls, "mean_in_k1_ci": bool(lo_k1 <= mean <= hi_k1), "per_seed": rates.tolist(),
            "small_rate": float(cs / ys), "large_rate": float(cl / yl), "m9": mk.log_ratio(np.concatenate([s.ratio for s in stats])),
            "inflating_firm_day_share": {g: v[0] / v[1] for g, v in infl.items()},
            "audits_finding_inflation_share": {g: (v[0] / v[1] if v[1] else None) for g, v in aud.items()},
            "audits": {g: v[1] for g, v in aud.items()},
            "moments_sigma0": m.tolist(), "J": float(criterion(m_data, W, m))})
    steps = [{"from": a["q"], "to": b["q"], "rising": b["rate"] > a["rate"]} for a, b in zip(points, points[1:])]
    crossings = {}
    for level in (lo_k1, hi_k1):
        crossings[f"{level:.6f}"] = [a["q"] + (level - a["rate"]) / (b["rate"] - a["rate"]) * (b["q"] - a["q"])
                                     for a, b in zip(points, points[1:]) if a["rate"] < level <= b["rate"]]
    hat = json.loads((HERE / "qw_k3.json").read_text(encoding="utf-8"))["theta_hat_point"]
    first = points[0]
    return {"contract": "28.34", "theta_hat": theta_from_domain(spec["scenarios"][0]["domain"]), "seeds": seeds,
            "k1_rate": k1["rate"], "k1_ci95": [lo_k1, hi_k1], "t_quantile": tq, "points": points, "steps": steps,
            "all_steps_rising": all(s["rising"] for s in steps), "crossings_on_rising_steps": crossings,
            "cannot_exclude": [p["q"] for p in points if p["class"] == "cannot_exclude"],
            "clearly_below": [p["q"] for p in points if p["class"] == "clearly_below"],
            "clearly_above": [p["q"] for p in points if p["class"] == "clearly_above"],
            "point_rule_set": [p["q"] for p in points if p["mean_in_k1_ci"]],
            "cross_check_first_point": {"q": first["q"], "scan_rate": first["rate"], "scan_se": first["rate_se"],
                                        "final_stage_rate": hat["rate"], "final_stage_se": hat["rate_se"],
                                        "difference": first["rate"] - hat["rate"],
                                        "combined_se": math.hypot(first["rate_se"], hat["rate_se"])},
            "detections_without_audit": without_audit}


# -- contract 28.37: the size-dependent audit scan ------------------------------------------------------------------

SIZE_GROUPS = {"large": 0, "mid": 1, "small": 2}


def size_audit_cells(spec: dict) -> list[dict]:
    """Name, arm and beta of each sizeaudit scenario. The scenario without a policy is random audits (beta = 0), shared
    by both arms; one with the initial_value allocation at beta = 0 (development only) is a check cell."""
    cells = []
    for sc in spec["scenarios"]:
        base = sc.get("policy", {}).get("baseline")
        if base is None:
            cells.append({"name": sc["name"], "arm": "both", "beta": 0.0})
            continue
        beta = float(base["audit_value_exponent"])
        arm = ("public" if base.get("audit_rule_public", True) else "hidden") if beta else "check"
        cells.append({"name": sc["name"], "arm": arm, "beta": beta})
    return cells


def bootstrap_m9(per_seed: dict[str, np.ndarray], mk, draws: int = 1000, seed: int = 28906) -> dict[str, np.ndarray]:
    """Seed-bootstrap draws of each cell's pooled m9; per_seed[cell] is seeds x [cases small, firm-years small, cases
    large, firm-years large]. The resampling is that of moments_k3.seed_bootstrap_cov (the same generator, one
    integers() call per draw), so all cells, which share their seeds in one order, get the same resampled seeds."""
    (count,) = {len(sums) for sums in per_seed.values()}
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, count, count) for _ in range(draws)]
    return {name: np.array([mk.log_ratio(sums[pick]) for pick in picks]) for name, sums in per_seed.items()}


def percentile_interval(draws: np.ndarray) -> dict:
    """2.5% and 97.5% quantiles by the inverse empirical distribution (no interpolation, so infinite draws, where a
    group has no detection, sort to the ends); NaN draws (no detection in either group) are left out and counted. The
    standard error is the standard deviation of the finite draws. With every draw NaN there is no interval (None)."""
    valid = draws[~np.isnan(draws)]
    ci = ([float(x) for x in np.quantile(valid, [0.025, 0.975], method="inverted_cdf")] if valid.size else None)
    finite = valid[np.isfinite(valid)]
    return {"ci95": ci, "se": float(finite.std(ddof=1)) if finite.size > 1 else None,
            "plus_inf_draws": int((valid == np.inf).sum()), "minus_inf_draws": int((valid == -np.inf).sum()),
            "nan_draws": int(draws.size - valid.size)}


def classify_m9(m9: float, ci: list[float], data_ci: list[float]) -> dict:
    """Contract 28.37: the sign of m9 and its interval against the data interval; an undefined m9 is in no class."""
    if not math.isfinite(m9) or ci is None:
        return {"sign": "undefined", "data": "undefined", "mean_in_data_ci": False}
    sign = "clearly_positive" if ci[0] > 0 else "clearly_negative" if ci[1] < 0 else "sign_undetermined"
    data = "clearly_below" if ci[1] < data_ci[0] else "clearly_above" if ci[0] > data_ci[1] else "cannot_exclude"
    return {"sign": sign, "data": data, "mean_in_data_ci": bool(data_ci[0] <= m9 <= data_ci[1])}


def arm_path(betas: list[float], m9s: list[float], levels: dict[str, float]) -> dict:
    """Each adjacent step along beta (from 0 downwards): does m9 rise; and, on rising steps only, the linear
    interpolation of the beta at which m9 crosses each level."""
    pairs = list(zip(betas, m9s))
    steps = [{"from": a, "to": b, "rising": bool(math.isfinite(ma) and math.isfinite(mb) and mb > ma)}
             for (a, ma), (b, mb) in zip(pairs, pairs[1:])]
    crossings = {label: [a + (level - ma) / (mb - ma) * (b - a)
                         for ((a, ma), (b, mb)), step in zip(zip(pairs, pairs[1:]), steps)
                         if step["rising"] and ma < level <= mb]
                 for label, level in levels.items()}
    return {"steps": steps, "all_steps_rising": all(s["rising"] for s in steps),
            "crossings_on_rising_steps": crossings}


def threshold_cells(rows: list[dict]) -> list[str]:
    """Contract 28.37: the cells (beta != 0) whose m9 is clearly positive and cannot be excluded from the data
    interval; an arm reaches the threshold if it has one."""
    return [r["name"] for r in rows if r["beta"] != 0 and r["sign"] == "clearly_positive"
            and r["data"] == "cannot_exclude"]


def size_audit(archive: Path, spec_path: Path) -> dict:
    """Contract 28.37 at theta_hat: m9 at each beta and rule (30 seeds pooled), its seed-bootstrap percentile interval,
    the sign and data-interval classes, the point rule, monotonicity and interpolated crossings per arm, the public
    minus hidden difference and the threshold. Descriptive by-products per cell and size group: the mean audit rate
    per period and the multipliers, audits, the share of audits that found inflation, the share of firm-days with
    inflation, yearly detection rates beside K1's, the audit and hit-share terms of m9, m1-m9 at sigma_m = 0 and J,
    price cap hits, detections without an audit, and the beta = 0 cross-check."""
    from scipy.stats import t as student_t
    mk = _load("moments_k3")
    k1 = json.loads((HERE / "k1_size_correction.json").read_text(encoding="utf-8"))["main"]
    data_ci = [float(x) for x in k1["log_ratio_small_over_large"]["T1_T2"]["ci95_log"]]
    k1_rates = {"all": k1["pooled"]["T1_T2"], **{g: k1["by_size"][g]["T1_T2"] for g in SIZE_GROUPS}}
    m_data, Omega_data, W, cuts = data_target()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    seeds, cells = spec["seeds"], size_audit_cells(spec)
    q_bar = spec["scenarios"][0]["domain"]["baseline_audit_probability"]
    tq = float(student_t.ppf(0.975, len(seeds) - 1))
    stats = {c["name"]: point_stats(archive, c["name"], seeds, 0.0, mk, cuts) for c in cells}
    per_seed = {name: np.array([s.ratio.sum(0) for s in st]) for name, st in stats.items()}
    boot = bootstrap_m9(per_seed, mk)
    rows, without_audit = [], 0
    blocks = slice(mk.BLOCK_START, mk.BLOCK_START + mk.BLOCKS * mk.BLOCK_DAYS)
    for cell in cells:
        name, st = cell["name"], stats[cell["name"]]
        targeted = cell["arm"] != "both"
        m9 = mk.log_ratio(np.concatenate([s.ratio for s in st]))
        interval = percentile_interval(boot[name])
        g_all = ("all", *SIZE_GROUPS)
        tally = {g: dict.fromkeys(("audits", "found", "inflating", "firm_days", "cases", "firm_years"), 0) for g in g_all}
        rate = {g: [] for g in g_all}
        caps, constant, budget_error = 0, True, 0.0
        keys = ("inflation", "audit_selected", "detections", "market_size_class", "market_price_cap_hits")
        for seed in seeds:
            a = load_arrays(archive, name, seed, keys + (("audit_probability",) if targeted else ()))
            caps += int(a["market_price_cap_hits"].sum())
            size_class = a["market_size_class"]
            pdays = np.arange(a["detections"].shape[0]) * 60
            periods = (pdays >= mk.BLOCK_START) & ((pdays - mk.BLOCK_START) // mk.BLOCK_DAYS < mk.BLOCKS)
            block = (pdays[periods] - mk.BLOCK_START) // mk.BLOCK_DAYS
            inflating = a["inflation"][blocks] > 0
            selected = a["audit_selected"][periods] == 1
            found = a["detections"][periods] > 0
            without_audit += int((found & ~selected).sum())
            hit = np.array([found[block == b].any(axis=0) for b in range(mk.BLOCKS)])
            if targeted:
                q_rows = a["audit_probability"][1:]
                constant &= bool((q_rows == q_rows[0]).all())
                budget_error = max(budget_error, float(np.abs(q_rows.sum(axis=1) - q_bar * size_class.size).max()))
                q = q_rows[0]
            else:
                q = np.full(size_class.size, q_bar)
            for g in g_all:
                cols = np.ones(size_class.size, dtype=bool) if g == "all" else size_class == SIZE_GROUPS[g]
                rate[g].append(float(q[cols].mean()))
                t = tally[g]
                t["audits"] += int(selected[:, cols].sum())
                t["found"] += int((found & selected)[:, cols].sum())
                t["inflating"] += int(inflating[:, cols].sum())
                t["firm_days"] += int(inflating[:, cols].size)
                t["cases"] += int(hit[:, cols].sum())
                t["firm_years"] += int(hit[:, cols].size)
        cs, ys, cl, yl = per_seed[name].sum(0)
        # The group tallies recount m9's cases from the same arrays.
        assert (tally["small"]["cases"], tally["small"]["firm_years"], tally["large"]["cases"],
                tally["large"]["firm_years"]) == (cs, ys, cl, yl), name
        per_audit = {g: t["audits"] / t["firm_years"] for g, t in tally.items()}
        hit_share = {g: (t["found"] / t["audits"] if t["audits"] else None) for g, t in tally.items()}
        with np.errstate(divide="ignore"):
            log_audits = float(np.log(per_audit["small"] / per_audit["large"])) if per_audit["large"] else math.inf
            log_hits = (float(np.log(hit_share["small"] / hit_share["large"]))
                        if hit_share["small"] and hit_share["large"] else None)
        ratios = {"small_over_large": np.array(rate["small"]) / np.array(rate["large"]),
                  "mid_over_large": np.array(rate["mid"]) / np.array(rate["large"])}
        moments = mk.pooled_model_moments(st)
        all_rates = yearly_rates(archive, name, seeds, mk)
        se_all = float(all_rates.std(ddof=1) / math.sqrt(len(all_rates)))
        rows.append({
            **cell, "m9": m9, **interval, **classify_m9(m9, interval["ci95"], data_ci),
            "cases": {"small": int(cs), "small_firm_years": int(ys), "large": int(cl), "large_firm_years": int(yl)},
            "rate_per_period": {g: float(np.mean(v)) for g, v in rate.items()},
            "yearly_audit_probability_of_mean_rate": {g: 1 - (1 - float(np.mean(v))) ** (252 / 60)
                                                      for g, v in rate.items()},
            "multipliers": {k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
                            for k, v in ratios.items()},
            "rates_constant_over_periods": constant, "budget_max_abs_error": budget_error,
            "audits": {g: t["audits"] for g, t in tally.items()}, "audits_per_firm_year": per_audit,
            "audits_finding_inflation_share": hit_share,
            "inflating_firm_day_share": {g: t["inflating"] / t["firm_days"] for g, t in tally.items()},
            "yearly_detection_rate": {g: t["cases"] / t["firm_years"] for g, t in tally.items()},
            "all_firm_rate_seed_mean": float(all_rates.mean()), "all_firm_rate_se": se_all,
            "all_firm_rate_ci95": [float(all_rates.mean()) - tq * se_all, float(all_rates.mean()) + tq * se_all],
            "m9_terms": {"log_audits_per_firm_year_ratio": log_audits, "log_hit_share_ratio": log_hits,
                         "m9_minus_terms": (m9 - log_audits - log_hits) if log_hits is not None else None},
            "moments_sigma0": moments.tolist(), "J": float(criterion(m_data, W, moments)), "price_cap_hits": caps})
    shared = next(r for r in rows if r["arm"] == "both")
    levels = {"zero": 0.0, "data_ci_low": data_ci[0], "data_ci_high": data_ci[1]}
    arms = {}
    for arm in ("public", "hidden"):
        path = [shared] + sorted((r for r in rows if r["arm"] == arm), key=lambda r: -r["beta"])
        own = path[1:]
        reached = threshold_cells(own)
        arms[arm] = {"cells": [r["name"] for r in path], "betas": [r["beta"] for r in path],
                     "m9": [r["m9"] for r in path], **arm_path([r["beta"] for r in path], [r["m9"] for r in path], levels),
                     "clearly_positive": [r["name"] for r in own if r["sign"] == "clearly_positive"],
                     "cannot_exclude": [r["name"] for r in own if r["data"] == "cannot_exclude"],
                     "point_rule": [r["name"] for r in own if r["mean_in_data_ci"]],
                     "threshold_cells": reached, "threshold_reached": bool(reached)}
    by_arm = {arm: {r["beta"]: r for r in rows if r["arm"] == arm} for arm in ("public", "hidden")}
    differences = []
    for beta in sorted(set(by_arm["public"]) & set(by_arm["hidden"]), reverse=True):
        pub, hid = by_arm["public"][beta], by_arm["hidden"][beta]
        with np.errstate(invalid="ignore"):
            draws = boot[pub["name"]] - boot[hid["name"]]
        differences.append({"beta": beta, "public_minus_hidden": pub["m9"] - hid["m9"], **percentile_interval(draws)})
    final_m9 = json.loads((HERE / "final_k3.json").read_text(encoding="utf-8"))["moments_final"][8]
    first = json.loads((HERE / "qscan_k3.json").read_text(encoding="utf-8"))["points"][0]
    assert first["q"] == q_bar
    return {"contract": "28.37", "spec": spec_path.name, "theta_hat": theta_from_domain(spec["scenarios"][0]["domain"]),
            "seeds": seeds, "q_bar": q_bar, "bootstrap": {"draws": 1000, "seed": 28906},
            "data": {"m9": float(m_data[8]), "se": float(math.sqrt(Omega_data[8, 8])), "ci95": data_ci},
            "k1_rates_T1_T2": k1_rates, "t_quantile": tq, "cells": rows, "arms": arms,
            "public_minus_hidden": differences,
            "threshold": {arm: v["threshold_reached"] for arm, v in arms.items()},
            "cross_check_beta0": {"m9": shared["m9"], "final_stage_m9": final_m9, "qscan_q0p010_m9": first["m9"],
                                  "minus_final_stage": shared["m9"] - final_m9,
                                  "minus_qscan_q0p010": shared["m9"] - first["m9"]},
            "detections_without_audit": without_audit, "price_cap_hits_total": sum(r["price_cap_hits"] for r in rows)}


def nonfinite_paths(node, path: str = "") -> list[str]:
    """Paths of the non-finite numbers in a nested result (for the development check, which prints no number)."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in nonfinite_paths(v, f"{path}/{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in nonfinite_paths(v, f"{path}[{i}]")]
    if isinstance(node, float) and not math.isfinite(node):
        return [path]
    return []


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
    p = sub.add_parser("qw")
    for name in ("qw_archive", "qw_spec", "final_archive", "final_spec"):
        p.add_argument(name, type=Path)
    p = sub.add_parser("qscan")
    p.add_argument("archive", type=Path)
    p.add_argument("spec", type=Path)
    p = sub.add_parser("sizeaudit")
    p.add_argument("archive", type=Path)
    p.add_argument("spec", type=Path)
    p.add_argument("--check", action="store_true",
                   help="development archive: report field names and non-finite paths only, print no value, write nothing")
    args = ap.parse_args()
    if args.cmd == "sizeaudit":
        out = size_audit(args.archive, args.spec)
        if args.check:
            print(json.dumps({"cells": [c["name"] for c in out["cells"]], "cell_fields": sorted(out["cells"][0]),
                              "top_fields": sorted(out), "nonfinite_paths": nonfinite_paths(out),
                              "detections_without_audit_zero": out["detections_without_audit"] == 0}, indent=1))
            return
        (HERE / "sizeaudit_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        for c in out["cells"]:
            print(f"{c['name']:15s} beta {c['beta']:5.2f}  m9 {c['m9']:7.3f}  95% [{c['ci95'][0]:7.3f}, {c['ci95'][1]:7.3f}]"
                  f"  se {c['se']}  {c['sign']:18s} {c['data']:15s} point {c['mean_in_data_ci']}"
                  f"  large cases {c['cases']['large']}")
        print(json.dumps({k: out[k] for k in ("threshold", "public_minus_hidden", "cross_check_beta0",
                                              "detections_without_audit", "price_cap_hits_total")}, indent=1))
        print(json.dumps({arm: {k: v[k] for k in ("all_steps_rising", "crossings_on_rising_steps", "threshold_cells")}
                          for arm, v in out["arms"].items()}, indent=1))
        return
    if args.cmd == "qscan":
        out = qscan(args.archive, args.spec)
        (HERE / "qscan_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("cannot_exclude", "clearly_below", "clearly_above", "point_rule_set",
                                              "all_steps_rising", "crossings_on_rising_steps",
                                              "cross_check_first_point", "detections_without_audit")}, indent=1))
        return
    if args.cmd == "qw":
        out = qw_set(args.qw_archive, args.qw_spec, args.final_archive, args.final_spec)
        (HERE / "qw_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: out[k] for k in ("k1_ci95", "esg_upper_95", "set_inside_k1_ci", "set_below_esg_upper",
                                              "theta_hat_point")}, indent=1))
        return
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
