"""Where inflating decisions occur, for the working paper.

Run from the repository root:

    python paper/composition.py

Inputs tracked in git:
  reports/disclosure_targeted_audits_20261020/audits_v1_analysis.json
  reports/disclosure_assurance_20261021/assurance_v1_analysis.json

For each scenario the script prints the share of decisions that inflate, the
shares of all inflating decisions made by large, mid and small firms, and, where
the analysis records it, the share made by firms that already carry undetected
inflation when they decide. These are descriptive re-expressions computed after
the experiments, not pre-registered quantities.

Every run has the same 7,600 decisions (200 firms on the 38 decision days
1260-3480) and the same tiers of 20, 60 and 120 firms, so the ratio of seed
means in an experiment's `describe` block equals the ratio of counts pooled over
its 30 seeds. The script checks, for every scenario, that the overall share of
inflating decisions equals the tier-weighted sum of the tier shares.

One further number needs the local run archive of the assurance experiment
(reports/disclosure_assurance_20261021/assurance_v1, not in git): the share of
assured firms' inflating decisions made by clients of the year's lowest-quality
provider, computed with the code of prediction T5 in that experiment's
analyze.py, for ALL_F16 (which reproduces T5) and for decay0_ALL_F16. Without
the archive the script skips it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "Targeted audits, seeds 42001-42030": ROOT / "reports/disclosure_targeted_audits_20261020/audits_v1_analysis.json",
    "Assurance providers, seeds 43001-43030": ROOT / "reports/disclosure_assurance_20261021/assurance_v1_analysis.json",
}
ASSURANCE_ARCHIVE = ROOT / "reports/disclosure_assurance_20261021/assurance_v1"
TIERS = {"large": 20 / 200, "mid": 60 / 200, "small": 120 / 200}
DECISIONS = 200 * 38
WINDOW, INFLATE = slice(21, 59), 1   # decision periods on days 1260-3480, as in the analyze.py scripts


def composition(row: dict) -> dict[str, float]:
    parts = {tier: weight * row[f"wash_share_{tier}"] for tier, weight in TIERS.items()}
    total = sum(parts.values())
    assert abs(total - row["wash_share"]) < 1e-9, (total, row["wash_share"])
    return {tier: part / total for tier, part in parts.items()}


def firm_quality(data: dict) -> np.ndarray:
    """Each firm's check probability in each decision period: its provider's quality, or 0 (as in analyze.py)."""
    assignment = data["assurer_assignment"]
    quality = np.take_along_axis(data["assurer_quality"], np.maximum(assignment, 0), axis=1)
    return np.where(assignment >= 0, quality, 0.)


def laxest_provider_share(scenario: str) -> tuple[int, int]:
    summary = json.loads((ASSURANCE_ARCHIVE / "summary.json").read_text(encoding="utf-8"))
    lax = washing = 0
    for record in summary["records"]:
        if record["scenario"] != scenario or record["status"] != "complete":
            continue
        with np.load(ASSURANCE_ARCHIVE / record["path"] / "result.npz") as file:
            data = {name: file[name] for name in ("actions", "assurer_assignment", "assurer_quality")}
        wash = (data["actions"][WINDOW] == INFLATE) & (data["assurer_assignment"][WINDOW] >= 0)
        lowest = firm_quality(data)[WINDOW] == data["assurer_quality"][WINDOW].min(axis=1, keepdims=True)
        washing += int(wash.sum())
        lax += int((wash & lowest).sum())
    return lax, washing


def main() -> None:
    for label, path in INPUTS.items():
        rows = json.loads(path.read_text(encoding="utf-8"))["describe"]
        print(f"\n{label}: shares of all inflating decisions in the policy window")
        print(f"{'scenario':<16}{'inflating':>10}{'large':>8}{'mid':>8}{'small':>8}{'continuing':>12}")
        for name, row in rows.items():
            if not row.get("wash_share"):
                continue
            share = composition(row)
            continuing = row.get("continuing_wash")
            tail = f"{continuing / (row['wash_share'] * DECISIONS):12.1%}" if continuing is not None else f"{'-':>12}"
            print(f"{name:<16}{row['wash_share']:10.1%}{share['large']:8.1%}{share['mid']:8.1%}{share['small']:8.1%}{tail}")
    print("\nAssured firms' inflating decisions made by clients of the year's lowest-quality provider")
    if not (ASSURANCE_ARCHIVE / "summary.json").exists():
        print("  skipped: the local run archive is not present")
        return
    for scenario in ("ALL_F16", "decay0_ALL_F16"):
        lax, washing = laxest_provider_share(scenario)
        print(f"  {scenario:<16}{lax:>7} of {washing:<7}{lax / washing:8.1%}")


if __name__ == "__main__":
    main()
