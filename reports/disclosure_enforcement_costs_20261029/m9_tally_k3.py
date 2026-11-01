"""Contract 28.32: m9 (ln of the small / large yearly detection ratio, seeds pooled) at every parameter point of every
K3 estimation archive, with the pooled case and firm-year counts behind it. Reads the archives; changes no estimate.
m9 does not depend on sigma_m (it uses detections only), so it is computed once per point.

    python -I -X utf8 m9_tally_k3.py

Writes m9_tally_k3.json here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import estimate_k3 as ek  # noqa: E402

ARCHIVES = (("k3a_v1", "disclosure_stage_k3a_v1"), ("stage1_v1", "disclosure_stage_k3_stage1_v1"),
            ("stage2_v1", "disclosure_stage_k3_stage2_v1"), ("stage2r_v1", "disclosure_stage_k3_stage2r_v1"),
            ("final_v1", "disclosure_stage_k3_final_v1"), ("recovery_v1", "disclosure_stage_k3_recovery_v1"),
            ("qw_v1", "disclosure_stage_k3_qw_v1"))


def main() -> None:
    mk = ek._load("moments_k3")
    m_data, _, _, cuts = ek.data_target()
    out: dict = {"contract": "28.32", "m9_data": float(m_data[8]), "archives": {}}
    total = {"points": 0, "negative": 0, "minus_inf": 0, "undefined_no_detection": 0, "nonnegative": 0}
    for archive, spec_name in ARCHIVES:
        spec = json.loads((ROOT / "configs" / f"{spec_name}.json").read_text(encoding="utf-8"))
        points = {}
        for sc in spec["scenarios"]:
            stats = ek.point_stats(HERE / archive, sc["name"], spec["seeds"], 0.0, mk, cuts)
            cs, ys, cl, yl = np.concatenate([s.ratio for s in stats]).sum(0)
            m9 = mk.log_ratio(np.concatenate([s.ratio for s in stats]))
            kind = ("negative" if m9 < 0 and math.isfinite(m9) else "minus_inf" if m9 == -math.inf
                    else "undefined_no_detection" if math.isnan(m9) else "nonnegative")
            points[sc["name"]] = {"m9": m9 if math.isfinite(m9) else str(m9), "kind": kind,
                                  "small_rate": float(cs / ys), "large_rate": float(cl / yl),
                                  "small_cases": int(cs), "large_cases": int(cl)}
        counts = {k: sum(p["kind"] == k for p in points.values()) for k in total if k != "points"}
        finite = [p["m9"] for p in points.values() if p["kind"] in ("negative", "nonnegative")]
        out["archives"][archive] = {"spec": spec_name, "seeds": len(spec["seeds"]), "points": len(points), **counts,
                                    "max_finite_m9": max(finite) if finite else None, "by_point": points}
        total["points"] += len(points)
        for k in counts:
            total[k] += counts[k]
        print(archive, len(points), counts, "max finite m9", max(finite) if finite else None, flush=True)
    out["total"] = total
    (HERE / "m9_tally_k3.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print("total", total)


if __name__ == "__main__":
    main()
