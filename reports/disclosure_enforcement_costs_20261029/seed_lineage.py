"""Seed lineage for stage K (contract section 28.8): the new seed blocks against every seed-like field in the repository.

    python seed_lineage.py

Scans every JSON file under configs, reports, runs, docs and data (this directory excluded) and collects the
integers stored under any key containing "seed", at any depth, as in contract 20.7, 25.14 and 26.10. Writes
seed_lineage.json (LF).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED_KEY = re.compile(r"seed", re.IGNORECASE)
# Contract 28.8 (data-side seeds, model seeds) and 28.15 (the seed bootstrap of the model moments).
BLOCKS = {
    "data_bootstrap": [28901],
    "g2_tests": [28902],
    "selftest": [28903],
    "sobol_stage1": [28904],
    "sobol_stage2": [28905],
    "seed_bootstrap_model_moments": [28906],
    "fit_test_monte_carlo": [28907],
    "development": [38201, 38202],
    "k3a": list(range(68001, 68011)),
    "stage1": list(range(63001, 63006)),
    "stage2": list(range(64001, 64021)),
    "final": list(range(65001, 65031)),
    "recovery": list(range(66001, 66031)),
    "q_w_set": list(range(67001, 67006)),
    "q_scan": list(range(69001, 69031)),                       # contract 28.34
    "size_audit_scan": list(range(70001, 70031)),              # contract 28.37
}


def collect(node, found: set[int], under_seed_key: bool = False) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            collect(value, found, under_seed_key or bool(SEED_KEY.search(str(key))))
    elif isinstance(node, list):
        for value in node:
            collect(value, found, under_seed_key)
    elif under_seed_key and isinstance(node, int) and not isinstance(node, bool):
        found.add(node)


def main() -> None:
    files, unreadable, found = 0, 0, set()
    for folder in ("configs", "reports", "runs", "docs", "data"):
        for path in (ROOT / folder).rglob("*.json"):
            if HERE in path.parents or path.name.startswith("disclosure_stage_k3"):   # this stage's own specs
                continue
            files += 1
            try:
                collect(json.loads(path.read_text(encoding="utf-8")), found)
            except (UnicodeDecodeError, json.JSONDecodeError):
                unreadable += 1
    every = [seed for values in BLOCKS.values() for seed in values]
    out = {"json_files": files, "unreadable": unreadable, "distinct_seed_values": len(found),
           "overlap": sorted(found & set(every)), "unique_within_design": len(every) == len(set(every)),
           "blocks": {name: [min(v), max(v), len(v)] for name, v in BLOCKS.items()},
           "used_nearby": {name: sorted(s for s in found if min(v) - 1000 <= s <= max(v) + 1000)
                           for name, v in BLOCKS.items()}}
    (HERE / "seed_lineage.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
