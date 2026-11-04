"""Checks of a contract 28.37 archive that the K3 pipeline check (check_pipeline_k3.py) does not make; counts only.

    .venv/Scripts/python.exe -I -X utf8 reports/disclosure_enforcement_costs_20261029/check_sizeaudit_k3.py <archive> <spec>

Per run: a complete attempt whose manifest policy is the frozen scenario's; under the initial_value allocation, the
archived audit rates of every audit day from the first on sum to n * q_bar and do not change from day to day; no
detection without an audit. When the archive has both random audits (sa_b0p00) and the initial_value allocation at
beta = 0 (sa_pub_b0p00, development only), their result arrays must be equal bit for bit, the arrays that only
targeted runs record (audit_probability, audit_flags) aside. Writes <archive name>_sizeaudit_check.json here and
prints no moment or rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETED_ONLY = {"audit_probability", "audit_flags"}


def attempt_of(archive: Path, scenario: str, seed: int) -> Path:
    for attempt in sorted((archive / scenario / f"seed_{seed}").glob("attempt_*"), reverse=True):
        if json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete":
            return attempt
    raise FileNotFoundError(f"no completed attempt for {scenario} seed {seed}")


def main() -> None:
    archive, spec_path = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    frozen = {s["name"]: s for s in json.loads((archive / "experiment.json").read_text(encoding="utf-8"))["scenarios"]}
    out = {"archive": archive.as_posix(), "runs": 0, "problems": [], "detections_without_audit": 0,
           "targeted_runs": 0, "max_budget_error": 0.0, "bitwise_pairs": 0}
    for scenario in spec["scenarios"]:
        name = scenario["name"]
        q_bar = scenario["domain"]["baseline_audit_probability"]
        for seed in spec["seeds"]:
            attempt = attempt_of(archive, name, seed)
            manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
            out["runs"] += 1
            if manifest["policy"] != frozen[name]["policy"]:
                out["problems"].append([name, seed, "manifest policy differs from the frozen scenario"])
            with np.load(attempt / "result.npz") as z:
                found, selected = z["detections"] > 0, z["audit_selected"] == 1
                out["detections_without_audit"] += int((found & ~selected).sum())
                if "policy" in scenario:
                    out["targeted_runs"] += 1
                    if "audit_probability" not in z.files:
                        out["problems"].append([name, seed, "no audit_probability array"])
                        continue
                    rates = z["audit_probability"][1:]
                    error = float(np.abs(rates.sum(axis=1) - q_bar * rates.shape[1]).max())
                    out["max_budget_error"] = max(out["max_budget_error"], error)
                    if error > 1e-12:
                        out["problems"].append([name, seed, "audit rates do not sum to the budget"])
                    if not (rates == rates[0]).all():
                        out["problems"].append([name, seed, "audit rates change between audit days"])
    names = {s["name"] for s in spec["scenarios"]}
    if {"sa_b0p00", "sa_pub_b0p00"} <= names:
        for seed in spec["seeds"]:
            with np.load(attempt_of(archive, "sa_b0p00", seed) / "result.npz") as plain, \
                    np.load(attempt_of(archive, "sa_pub_b0p00", seed) / "result.npz") as zero:
                if set(zero.files) - TARGETED_ONLY != set(plain.files):
                    out["problems"].append(["sa_pub_b0p00", seed, "arrays other than the targeted ones differ in name"])
                    continue
                unequal = [k for k in plain.files if plain[k].dtype != zero[k].dtype
                           or not np.array_equal(plain[k], zero[k], equal_nan=plain[k].dtype.kind == "f")]
                if unequal:
                    out["problems"].append(["sa_pub_b0p00", seed, f"not bit for bit: {unequal}"])
                out["bitwise_pairs"] += 1
    if out["detections_without_audit"]:
        out["problems"].append(["all", None, "detections without an audit"])
    (HERE / f"{archive.name}_sizeaudit_check.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "archive"}, indent=1))
    if out["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
