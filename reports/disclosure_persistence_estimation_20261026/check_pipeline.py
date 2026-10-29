"""Pipeline check of the development run (contract 25.10): sampled archive, normal scores, moments, variances.

  python check_pipeline.py <development archive>

For every trial of configs/disclosure_persistence_grid_check_v1.json: the sampled days are 1260 + 63 j (j = 0..35);
the sampled arrays have one row per day and one column per firm and are finite; the cross-sectional SD of the true
level does not shrink under reversion; under Q the anchors have the planned variance and do not move; the moments
of estimate.py are finite and lie in [-1, 1] for every sigma_m. Moments are counted, not printed, and are not
compared with the targets. Writes check_v1_pipeline.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import estimate as est

HERE = Path(__file__).resolve().parent
EXPECTED_DAYS = 1260 + 63 * np.arange(36)


def completed_attempt(folder: Path) -> Path:
    for attempt in sorted(folder.glob("attempt_*"), reverse=True):
        if json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete":
            return attempt
    raise FileNotFoundError(folder)


def main(archive: Path) -> None:
    frozen = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
    rows, failures = [], []
    for scenario in frozen["scenarios"]:
        name, domain = scenario["name"], scenario["domain"]
        for seed in frozen["spec"]["seeds"]:
            attempt = completed_attempt(archive / name / f"seed_{seed}")
            manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
            validation = json.loads((attempt / "validation.json").read_text(encoding="utf-8"))
            with np.load(attempt / "result.npz") as saved:
                days, signal, true = saved["sampled_days"], saved["sampled_public_signal"], saved["sampled_true_goodness"]
                anchor = saved["sampled_reversion_anchor"] if "sampled_reversion_anchor" in saved.files else None
                omitted = {"true_goodness", "public_signal", "reversion_anchor"} & set(saved.files)
            row = {"scenario": name, "seed": seed, "attempts": len(list((archive / name / f"seed_{seed}").glob("attempt_*"))),
                   "core_validation": {gate: validation[gate]["status"] for gate in ("mechanical", "stylized")},
                   "sampled_days_ok": bool(np.array_equal(days, EXPECTED_DAYS)),
                   "shapes_ok": bool(signal.shape == true.shape == (36, 200)),
                   "finite": bool(np.isfinite(signal).all() and np.isfinite(true).all()),
                   "full_arrays_omitted": not omitted,
                   "sampled_archive_manifest": manifest.get("sampled_archive"),
                   "true_sd_min_max": [float(true.std(axis=1).min()), float(true.std(axis=1).max())]}
            if anchor is not None:
                share = domain["reversion_permanent_share"]
                row["anchor"] = {"planned_variance": share, "variance": float(anchor[0].var()),
                                 "constant": bool((anchor == anchor[0]).all()),
                                 "transitory_sd_min_max": [float((true - anchor).std(axis=1).min()),
                                                           float((true - anchor).std(axis=1).max())]}
            moments = est.point_moments({seed: signal})
            row["moments_ok"] = bool(np.isfinite(moments).all() and (np.abs(moments) <= 1).all())
            row["moment_count"] = int(moments.size)
            checks = [row["sampled_days_ok"], row["shapes_ok"], row["finite"], row["full_arrays_omitted"], row["moments_ok"]]
            if domain.get("natural_reversion"):
                checks.append(row["true_sd_min_max"][0] > 0.8)
            if anchor is not None:
                checks += [row["anchor"]["constant"], abs(row["anchor"]["variance"] - share) < 0.15]
            if not all(checks):
                failures.append(f"{name} seed {seed}")
            rows.append(row)
    out = {"experiment_id": frozen["experiment_id"], "trials": rows, "failures": failures, "passed": not failures}
    (HERE / "check_v1_pipeline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: out[key] for key in ("experiment_id", "failures", "passed")}, indent=1))
    for row in rows:
        print(row["scenario"], row["seed"], row["core_validation"], "true SD", [round(v, 3) for v in row["true_sd_min_max"]],
              row.get("anchor", ""))
    if failures:
        raise SystemExit("pipeline check failed")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
