"""Final stage and recovery specs at the grid-stage estimate of the selected form (contract 25.3, 25.7, 25.15).

  python build_final_spec.py

Reads estimation_v1.json (grid stage) and writes configs/disclosure_persistence_final_v1.json (estimation seeds
48001-48030) and configs/disclosure_persistence_recovery_v1.json (recovery seeds 52001-52030): one scenario each, the
stage F baseline B under the selected form with its model parameters rounded to three decimals. sigma_m is not a
model parameter; it enters only in post-processing.
"""

from __future__ import annotations

import json
from pathlib import Path

import estimate as est

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = {"final": list(range(48001, 48031)), "recovery": list(range(52001, 52031))}
KEYS = {"delta": "natural_decay", "kappa": "natural_reversion", "kappa_T": "natural_reversion",
        "s_P": "reversion_permanent_share"}


def main() -> None:
    estimation = json.loads((HERE / "estimation_v1.json").read_text(encoding="utf-8"))
    selected = estimation["selected"]
    estimate = estimation["forms"][selected]["estimate"]
    domain = {} if selected == "U" else {"natural_decay": 0.0}
    domain.update({KEYS[name]: round(value, 3) for name, value in estimate.items() if name != "sigma_m"})
    name = f"{selected.lower()}_hat"
    grid = est.build_spec.spec("estimation")
    for stage, seeds in SEEDS.items():
        payload = {**grid, "name": f"disclosure_persistence_{stage}_v1", "seeds": seeds,
                   "_note": f"Stage H {stage} run (research contract section 25): the stage F baseline B at the "
                            f"grid-stage estimate of form {selected} ({json.dumps(domain)}), sampled archive.",
                   "scenarios": [est.build_spec.stage_f.scenario(name, domain=domain)]}
        path = ROOT / "configs" / f"disclosure_persistence_{stage}_v1.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path, name, domain, len(seeds), "seeds")


if __name__ == "__main__":
    main()
