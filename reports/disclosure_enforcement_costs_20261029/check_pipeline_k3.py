"""Pipeline check of a K3 development archive (contract 28.7): counts only, no comparison with the data moments.

    python -I -X utf8 check_pipeline_k3.py <archive> <spec>

Checks per run: complete attempt, 3,500 days, 36 sampled dates at day 1260 + 63j, sampled signal 36 x 200 and finite,
size classes 20 / 60 / 120, detections finite and non-negative, the scenario's domain fields as in the spec, price cap
hits; and that m1-m9 can be computed for every point and sigma_m (finite, correlations within [-1, 1]). Prints counts
and writes <archive name>_pipeline.json here; it prints no moment value.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    archive, spec_path = Path(sys.argv[1]), Path(sys.argv[2])
    ek, mk = _load("estimate_k3"), _load("moments_k3")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cuts = np.array(json.loads((HERE / "data_moments_k3.json").read_text(encoding="utf-8"))["inputs"]["notch_cuts"])
    out = {"archive": archive.as_posix(), "runs": {}, "problems": []}
    for scenario in spec["scenarios"]:
        name = scenario["name"]
        stats = {s: [] for s in ek.SIGMAS}
        for seed in spec["seeds"]:
            run = ek.finished(archive, name, seed)
            attempt = sorted((archive / name / f"seed_{seed}").glob("attempt_*"))[-1]
            frozen = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
            with np.load(attempt / "result.npz") as z:
                days = z["sampled_days"].tolist()
                n_days = z["inflation"].shape[0]
            ok = {"days": n_days == 3501, "sampled_days": days == [1260 + 63 * j for j in range(36)],
                  "signal_shape": run["signal"].shape == (36, 200), "signal_finite": bool(np.isfinite(run["signal"]).all()),
                  "size_classes": np.bincount(run["size_class"], minlength=3).tolist() == [20, 60, 120],
                  "detections_ok": bool((run["detections"] >= 0).all())}
            text = json.dumps(frozen)
            for key, value in scenario["domain"].items():
                ok[f"domain_{key}"] = json.dumps(value) in text
            for k, v in ok.items():
                if not v:
                    out["problems"].append([name, seed, k])
            out["runs"][f"{name}/{seed}"] = {"price_cap_hits": int(run["price_cap_hits"].sum()),
                                             "detections_total": int(run["detections"].sum())}
            for sigma in ek.SIGMAS:
                sig = run["signal"] + sigma * ek.noise(seed, run["signal"].shape)
                stats[sigma].append(mk.model_stats(sig, run["size_class"], run["detections"], cuts))
        for sigma, st in stats.items():
            m = mk.pooled_model_moments(st)
            if not (np.isfinite(m[:8]).all() and (np.abs(m[:5]) <= 1).all()):
                out["problems"].append([name, float(sigma), "moments m1-m8 not finite or outside [-1, 1]"])
            if not np.isfinite(m[8]):
                out["problems"].append([name, float(sigma), "m9 undefined (no detection in a size group)"])
    (HERE / f"{archive.name}_pipeline.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(len(out["runs"]), "runs;", len(out["problems"]), "problems:", out["problems"][:10])
    print("price cap hits total:", sum(r["price_cap_hits"] for r in out["runs"].values()))


if __name__ == "__main__":
    main()
