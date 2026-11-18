"""Seed lineage for stage J (contract section 27): the new seed blocks against every seed-like field in the repository.

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
# Stage J runs no new model seeds (the model replay reuses the frozen H3 seeds 53001-53030); these seed the
# within-date permutation of the smoke run, the synthetic panels of the estimator self-test and the
# permutation size check.
BLOCKS = {
    "permutation": [27901],
    "selftest": [27902],
    "null_check": [27903],
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
            if HERE in path.parents:
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
