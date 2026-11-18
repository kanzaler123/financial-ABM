"""Stage H: simulated minimum distance estimation of the natural dynamics (research contract 25.4-25.7, 25.15).

  python estimate.py --grid <grid archive> [--final <final archive>] [--recovery <recovery archive>]

Reads the sampled public signals of the frozen grid experiment (configs/disclosure_persistence_grid_v1.json), adds
measurement noise sigma_m in post-processing, computes the target moments with the functions of data_moments.py,
interpolates the seed-mean moments over each form's grid, estimates every form against the data moments, applies
the selection rule, and writes estimation_v1.json and estimation_v1.md. With --final (the selected form at its
estimate, seeds 48001-48030) it recomputes fit and standard errors; with --recovery (seeds 52001-52030 at the
estimate) it re-estimates on three pseudo-datasets. Only finished archives are read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_moments = _load("data_moments")
build_spec = _load("build_spec")

LAGS = data_moments.LAGS
SIGMAS = np.round(np.arange(0.0, 0.6001, 0.05), 2)          # contract 25.2, 13 values
NOISE_STREAM = 2504                                         # noise draws: SeedSequence([seed, NOISE_STREAM])
MC_DRAWS = 1_000_000                                        # contract 25.5
MC_SEED = 25005
FORMS = {
    "U": (("delta",), (build_spec.DECAYS,), lambda d: f"u_d{build_spec.stage_f.label(d)}"),
    "R": (("kappa",), (build_spec.KAPPAS,), lambda k: f"r_k{build_spec.stage_f.label(k)}"),
    "Q": (("kappa_T", "s_P"), (build_spec.KAPPAS_T, build_spec.SHARES),
          lambda k, s: f"q_k{build_spec.stage_f.label(k)}_s{build_spec.stage_f.label(s)}"),
}
ORDER = ("U", "R", "Q")                                     # contract 25.6: first form that fits


# -- simulated moments ------------------------------------------------------------------------------------------

def finished_signal(archive: Path, scenario: str, seed: int) -> np.ndarray:
    """Sampled public signal (dates x firms) of the completed attempt of one trial."""
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*"), reverse=True):
        manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] == "complete":
            with np.load(attempt / "result.npz") as saved:
                return saved["sampled_public_signal"].copy()
    raise FileNotFoundError(f"no completed attempt for {scenario} seed {seed}")


def noise(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Standard normal measurement errors of one seed, common to every grid point and sigma_m."""
    return np.random.default_rng(np.random.SeedSequence([seed, NOISE_STREAM])).standard_normal(shape)


def seed_moments(signal: np.ndarray, sigma: float, errors: np.ndarray) -> np.ndarray:
    """The target moments of one simulated market: normal scores per date, pooled lag correlations."""
    Z = data_moments.normal_scores((signal + sigma * errors).T)
    sums = data_moments.pair_sums(Z, LAGS)
    return np.array([data_moments.pooled(sums[k]) for k in LAGS])


def point_moments(signals: dict[int, np.ndarray]) -> np.ndarray:
    """Per sigma_m and seed: moments, shape (sigmas, seeds, lags)."""
    seeds = sorted(signals)
    errors = {seed: noise(seed, signals[seed].shape) for seed in seeds}
    return np.array([[seed_moments(signals[seed], sigma, errors[seed]) for seed in seeds] for sigma in SIGMAS])


def grid_moments(archive: Path, seeds: list[int]) -> dict[str, np.ndarray]:
    """Form -> array (*parameter grid, sigmas, seeds, lags)."""
    out = {}
    for form, (_, axes, name) in FORMS.items():
        shape = tuple(len(axis) for axis in axes)
        values = np.empty(shape + (len(SIGMAS), len(seeds), len(LAGS)))
        for index in np.ndindex(*shape):
            point = name(*(axis[i] for axis, i in zip(axes, index)))
            values[index] = point_moments({seed: finished_signal(archive, point, seed) for seed in seeds})
        out[form] = values
    return out


# -- estimation -------------------------------------------------------------------------------------------------

class Form:
    """Interpolated seed-mean moments of one form over (parameters, sigma_m)."""

    def __init__(self, name: str, moments: np.ndarray):
        self.name = name
        self.labels, axes, _ = FORMS[name]
        self.axes = [np.asarray(axis, dtype=float) for axis in axes] + [SIGMAS]
        self.moments = moments
        self.surface = RegularGridInterpolator(self.axes, moments.mean(axis=-2), method="cubic")
        self.bounds = [(axis[0], axis[-1]) for axis in self.axes]

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.surface(np.clip(theta, *np.array(self.bounds).T)[None, :])[0]

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        columns = []
        for j, (low, high) in enumerate(self.bounds):
            step = 1e-3 * (high - low)
            up, down = theta.copy(), theta.copy()
            up[j], down[j] = min(theta[j] + step, high), max(theta[j] - step, low)
            columns.append((self(up) - self(down)) / (up[j] - down[j]))
        return np.column_stack(columns)

    def seed_covariance(self, theta: np.ndarray) -> np.ndarray:
        """Covariance of per-seed moments at the grid node nearest to theta, divided by the number of seeds."""
        index = tuple(int(np.argmin(np.abs((axis - value) / (axis[-1] - axis[0]))))
                      for axis, value in zip(self.axes, theta))
        draws = self.moments[index]
        return np.cov(draws, rowvar=False) / draws.shape[0]


def criterion(target: np.ndarray, weights: np.ndarray, form: Form):
    return lambda theta: float(((target - form(theta)) ** 2 * weights).sum())


def minimise(objective, form: Form, starts: int = 10) -> np.ndarray:
    nodes = np.array(np.meshgrid(*form.axes, indexing="ij")).reshape(len(form.axes), -1).T
    values = np.array([objective(node) for node in nodes])
    best = None
    for node in nodes[np.argsort(values)[:starts]]:
        fit = minimize(objective, node, method="L-BFGS-B", bounds=form.bounds)
        if best is None or fit.fun < best.fun:
            best = fit
    return best.x


def weighted_chi2_p(statistic: float, eigenvalues: np.ndarray) -> float:
    rng = np.random.default_rng(MC_SEED)
    draws = rng.standard_normal((MC_DRAWS, eigenvalues.size)) ** 2 @ eigenvalues
    return float((draws >= statistic).mean())


def inference(form: Form, theta: np.ndarray, target: np.ndarray, data_cov: np.ndarray, weights: np.ndarray,
              simulated: np.ndarray | None = None, sim_cov: np.ndarray | None = None) -> dict:
    """Fit, sandwich standard errors and the fit test at theta (contract 25.5)."""
    model = form(theta) if simulated is None else simulated
    gap = target - model
    omega = data_cov + (form.seed_covariance(theta) if sim_cov is None else sim_cov)
    W = np.diag(weights)
    G = form.jacobian(theta)
    A = np.linalg.pinv(G.T @ W @ G)
    V = A @ G.T @ W @ omega @ W @ G @ A
    P = np.eye(len(target)) - G @ A @ G.T @ W
    root = np.linalg.cholesky(omega + 1e-15 * np.eye(len(target)))
    eigenvalues = np.clip(np.linalg.eigvalsh(root.T @ P.T @ W @ P @ root), 0.0, None)
    statistic = float(gap @ W @ gap)
    p_value = weighted_chi2_p(statistic, eigenvalues)
    band = 1.96 * np.sqrt(np.diag(omega))
    within = np.abs(gap) <= band
    on_boundary = [bool(np.isclose(value, low) or np.isclose(value, high)) for value, (low, high) in zip(theta, form.bounds)]
    labels = list(form.labels) + ["sigma_m"]
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return {
        "estimate": dict(zip(labels, theta.tolist())), "se": dict(zip(labels, se.tolist())),
        "ci95": {name: [value - 1.96 * s, value + 1.96 * s] for name, value, s in zip(labels, theta, se)},
        "on_grid_boundary": dict(zip(labels, on_boundary)),
        "moments_model": model.tolist(), "moments_data": target.tolist(), "gap": gap.tolist(),
        "band95": band.tolist(), "within_band": within.tolist(),
        "T": statistic, "T_p_value": p_value, "T_weights": eigenvalues.tolist(),
        "fits": bool(p_value >= 0.05 and within.all()),
    }


def profile_intervals(form: Form, theta: np.ndarray, target: np.ndarray, omega: np.ndarray) -> dict:
    """Auxiliary (contract 25.5, 25.15): distance-metric intervals of the persistence parameters."""
    inverse = np.linalg.inv(omega)
    distance = lambda t: float((target - form(t)) @ inverse @ (target - form(t)))
    floor = distance(theta)
    out = {}
    for j, name in enumerate(form.labels):
        low, high = form.bounds[j]
        values, kept = np.linspace(low, high, 61), []
        for value in values:
            others = [k for k in range(len(theta)) if k != j]

            def restricted(free, value=value, others=others):
                full = theta.copy()
                full[j] = value
                full[others] = free
                return distance(full)

            fit = minimize(restricted, theta[others], method="L-BFGS-B", bounds=[form.bounds[k] for k in others])
            if fit.fun - floor <= 3.841458820694124:
                kept.append(float(value))
        out[name] = [min(kept), max(kept)] if kept else None
    return out


def estimate_all(grid: dict[str, np.ndarray], target: np.ndarray, data_cov: np.ndarray) -> dict:
    weights = 1.0 / np.diag(data_cov)
    forms = {}
    for name in ORDER:
        form = Form(name, grid[name])
        theta = minimise(criterion(target, weights, form), form)
        row = inference(form, theta, target, data_cov, weights)
        omega = data_cov + form.seed_covariance(theta)
        row["profile_intervals"] = profile_intervals(form, theta, target, omega)
        forms[name] = row
    fitting = [name for name in ORDER if forms[name]["fits"]]
    selected = fitting[0] if fitting else min(ORDER, key=lambda name: forms[name]["T"])
    return {"forms": forms, "fitting_forms": fitting, "selected": selected,
            "selected_fits": bool(fitting), "rule": "contract 25.6: first of U, R, Q that fits, else smallest T"}


# -- final stage and recovery -----------------------------------------------------------------------------------

def final_stage(form: Form, theta: np.ndarray, archive: Path, scenario: str, seeds: list[int],
                target: np.ndarray, data_cov: np.ndarray) -> dict:
    sigma = theta[-1]
    draws = []
    for seed in seeds:
        signal = finished_signal(archive, scenario, seed)
        draws.append(seed_moments(signal, sigma, noise(seed, signal.shape)))
    draws = np.array(draws)
    row = inference(form, theta, target, data_cov, 1.0 / np.diag(data_cov), simulated=draws.mean(axis=0),
                    sim_cov=np.cov(draws, rowvar=False) / len(seeds))
    row["seeds"] = len(seeds)
    return row


def pseudo_moments(signals: dict[int, np.ndarray], replications: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-data from several markets: per-market normal scores, firm units pooled, firm-cluster bootstrap."""
    Z = np.vstack([data_moments.normal_scores(signals[seed].T) for seed in sorted(signals)])
    sums = data_moments.pair_sums(Z, LAGS)
    value = np.array([data_moments.pooled(sums[k]) for k in LAGS])
    rng = np.random.default_rng(data_moments.BOOTSTRAP_SEED)
    draws = np.empty((replications, len(LAGS)))
    for b in range(replications):
        weights = rng.multinomial(Z.shape[0], np.full(Z.shape[0], 1 / Z.shape[0])).astype(float)
        draws[b] = [data_moments.pooled(sums[k], weights) for k in LAGS]
    return value, np.cov(draws, rowvar=False)


def recovery(grid: dict[str, np.ndarray], archive: Path, scenario: str, groups: list[list[int]],
             main: dict) -> dict:
    rows = []
    for seeds in groups:
        target, cov = pseudo_moments({seed: finished_signal(archive, scenario, seed) for seed in seeds})
        result = estimate_all(grid, target, cov)
        selected = main["selected"]
        estimate = result["forms"][selected]["estimate"]
        ci = main["final"]["ci95"] if "final" in main else main["forms"][selected]["ci95"]
        inside = {name: bool(ci[name][0] <= value <= ci[name][1]) for name, value in estimate.items()}
        rows.append({"seeds": [seeds[0], seeds[-1]], "selected": result["selected"], "estimate": estimate,
                     "inside_main_ci95": inside, "fitting_forms": result["fitting_forms"]})
    recovered = all(row["selected"] == main["selected"] and all(row["inside_main_ci95"].values()) for row in rows)
    return {"groups": rows, "recovered": recovered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--target", default="R1", choices=("R1", "R1_csi300_current"))
    args = parser.parse_args()
    frozen = json.loads((args.grid / "experiment.json").read_text(encoding="utf-8"))
    seeds = frozen["spec"]["seeds"]
    targets = json.loads((HERE / "data_moments.json").read_text(encoding="utf-8"))["targets"]
    data = targets[args.target]
    assert data["lags"] == list(LAGS)
    target, data_cov = np.array(data["moments"]), np.array(data["bootstrap_cov"])
    grid = grid_moments(args.grid, seeds)
    out = {"grid_experiment_id": frozen["experiment_id"], "target": args.target, "grid_seeds": seeds,
           **estimate_all(grid, target, data_cov)}
    if args.final:
        final = json.loads((args.final / "experiment.json").read_text(encoding="utf-8"))
        scenario = final["scenarios"][0]["name"]
        form = Form(out["selected"], grid[out["selected"]])
        theta = np.array(list(out["forms"][out["selected"]]["estimate"].values()))
        out["final"] = {"experiment_id": final["experiment_id"], "scenario": scenario,
                        **final_stage(form, theta, args.final, scenario, final["spec"]["seeds"], target, data_cov)}
    if args.recovery:
        recovery_spec = json.loads((args.recovery / "experiment.json").read_text(encoding="utf-8"))
        seeds_r = recovery_spec["spec"]["seeds"]
        groups = [seeds_r[i:i + 10] for i in range(0, len(seeds_r), 10)]
        out["recovery"] = {"experiment_id": recovery_spec["experiment_id"],
                           **recovery(grid, args.recovery, recovery_spec["scenarios"][0]["name"], groups, out)}
    suffix = "" if args.target == "R1" else f"_{args.target}"
    path = HERE / f"estimation_v1{suffix}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({name: {"estimate": row["estimate"], "T": row["T"], "p": row["T_p_value"], "fits": row["fits"]}
                      for name, row in out["forms"].items()}, indent=1))
    print("selected", out["selected"], "fits" if out["selected_fits"] else "(no form fits)")


if __name__ == "__main__":
    sys.exit(main())
