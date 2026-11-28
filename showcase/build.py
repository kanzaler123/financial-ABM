"""Build the static site from checked-in files. No simulation runs here.

  _site/index.html         greenwashing policy sandbox (plan stage G): exported synthetic firms, the
                           browser decision core and stage F results, each traced to a committed file
  _site/stage1/index.html  the archived stage 1 exhibit (historical v13/v14 mechanism reports)
  _site/social-card.png    link preview image (og:image) of both pages, copied from showcase/social/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "formal": "stage1_mechanism_acceptance_v13_20260814",
    "holdout": "stage1_structural_holdout_v13_20260814",
    "development": "stage1_structural_dev_v14_20260814",
}
METRICS = (
    "daily_volatility", "absolute_return_acf_1", "excess_kurtosis",
    "mean_absolute_log_price_gap",
)
CONFIG_FIELDS = (
    "population_size", "trading_days", "burn_in_days", "strategy_shares",
    "learning_enabled",
)
# Declared by og:image:width / og:image:height in both page templates; render with showcase/social/render.py.
SOCIAL_CARD_SIZE = (1200, 630)


def read_json(path: Path) -> dict:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {path}: {value}")
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def export_report(root: Path, report_id: str, directory: str) -> dict:
    path = root / "reports" / directory / "mechanism_report.json"
    report = read_json(path)
    gate = report["gate"]
    checks = gate["checks"]
    if not checks or any(type(value) is not bool for value in checks.values()):
        raise ValueError(f"Invalid gate checks: {path}")
    passed = sum(checks.values())
    if gate["passed_checks"] != passed or gate["total_checks"] != len(checks):
        raise ValueError(f"Inconsistent gate counts: {path}")
    if type(gate["passed"]) is not bool or gate["passed"] != all(checks.values()):
        raise ValueError(f"Inconsistent gate decision: {path}")
    protocol = report["protocol"]
    seeds = protocol["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError(f"Invalid seed inventory: {path}")
    config = {key: report["config"][key] for key in CONFIG_FIELDS}
    if not 0 <= config["burn_in_days"] < config["trading_days"]:
        raise ValueError(f"Invalid evaluation window: {path}")
    summary = {}
    for scenario, source_metrics in report["summary"].items():
        summary[scenario] = {}
        for metric in METRICS:
            values = {key: source_metrics[metric][key] for key in ("median", "q10", "q90")}
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
                raise ValueError(f"Invalid {scenario}/{metric} in {path}")
            if not values["q10"] <= values["median"] <= values["q90"]:
                raise ValueError(f"Unordered quantiles in {path}")
            summary[scenario][metric] = values
    if "full" not in summary:
        raise ValueError(f"Missing baseline: {path}")
    evidence = {
        key: {field: value[field] for field in ("observed", "threshold", "pass_rate", "failed_seeds") if field in value}
        for key, value in gate.get("evidence", {}).items()
    }
    # Validate the historical formal verdict; this is NOT a new model gate run.
    if report_id == "formal":
        decision = read_json(path.with_name("final_decision.json"))
        if decision["passed_checks"] != passed or decision["total_checks"] != len(checks):
            raise ValueError("Formal final_decision disagrees with archived report")
        if decision["stage1_complete"] is not False or decision["may_enter_stage2"] is not False:
            raise ValueError("Historical exhibit expects incomplete Stage 1; revise its narrative before publication")
        if passed != 18 or len(checks) != 22:
            raise ValueError("Historical 18/22 snapshot changed; revise the dated exhibition copy")
    return {
        "id": report_id, "path": f"reports/{directory}",
        "protocol": protocol["version"], "stage": protocol.get("stage", "development"),
        "seeds": len(seeds), "config": config,
        "code_revision": report.get("code_revision", "not recorded"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": checks, "evidence": evidence, "summary": summary,
    }


SANDBOX_FIRM_FIELDS = {"size_class", "market_cap", "cost_scale", "benefit", "initial_goodness", "goodness",
                       "inflation", "pending", "full_model_decision"}
MAS_UPGRADE_PATH = "reports/disclosure_mas_upgrade_20261125/mas_upgrade_analysis.json"
MAS_UPGRADE_STUDIES = (
    "sec29_stackelberg",
    "sec30_empirical_belief",
    "sec31_assurance",
    "sec32A_q_dynamics",
    "sec32B_phase_pareto",
    "sec32C1_granularity",
    "sec32C2_ai_washing",
)
MAS_SCENARIO_FIELDS = (
    "true_change_eq",
    "true_change_cap",
    "wash_share",
    "wash_share_large",
    "wash_share_mid",
    "wash_share_small",
    "wash_count",
    "hit_rate_pooled",
    "q_gap",
    "resource_cost_bp",
    "assurance_effort_cost",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_or_none(value: object, context: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid numeric value for {context}: {value!r}")
    return float(value)


def export_mas_upgrade(root: Path) -> dict:
    """Compact, SHA-256-verified export of the §§29-32 MAS upgrade confirmatory suite."""
    path = root / MAS_UPGRADE_PATH
    analysis = read_json(path)
    if analysis.get("schema_version") != "mas_upgrade_analysis_v1":
        raise ValueError(f"Unexpected MAS upgrade schema_version in {path}")
    raw_studies = analysis.get("studies", {})
    missing = [name for name in MAS_UPGRADE_STUDIES if name not in raw_studies]
    if missing:
        raise ValueError(f"Missing MAS upgrade studies in {path}: {missing}")
    studies: dict[str, dict] = {}
    total_runs = 0
    total_planned = 0
    total_primary = 0
    total_supported = 0
    parity_mismatches = 0
    seed_min = None
    seed_max = None
    for study_key in MAS_UPGRADE_STUDIES:
        raw = raw_studies[study_key]
        complete = int(raw["complete_trials"])
        planned = int(raw["planned_trials"])
        total_runs += complete
        total_planned += planned
        parity_mismatches += int(raw.get("preswitch_parity_mismatches", 0))
        sr = raw["seed_range"]
        seed_min = sr[0] if seed_min is None else min(seed_min, sr[0])
        seed_max = sr[1] if seed_max is None else max(seed_max, sr[1])
        scenarios: dict[str, dict] = {}
        for sc_name in raw["scenario_order"]:
            sc = raw["scenarios"][sc_name]
            compact: dict[str, object] = {}
            for field in MAS_SCENARIO_FIELDS:
                if field in sc:
                    compact[field] = _finite_or_none(sc[field], f"{study_key}/{sc_name}/{field}")
            vs = sc.get("vs_baseline_eq")
            if isinstance(vs, dict):
                compact["vs_baseline_eq"] = {
                    k: _finite_or_none(vs[k], f"{study_key}/{sc_name}/vs_baseline_eq/{k}")
                    for k in ("mean", "ci_low", "ci_high")
                }
            scenarios[sc_name] = compact
        primary: list[dict] = []
        for comp_name, comp in raw["comparisons"].items():
            if not comp.get("primary"):
                continue
            holm_entries = [(mk, mv) for mk, mv in comp["metrics"].items() if "holm_p_value" in mv]
            if len(holm_entries) != 1:
                raise ValueError(f"Expected 1 Holm metric for {study_key}/{comp_name}, got {len(holm_entries)}")
            metric_key, mv = holm_entries[0]
            holm_p = float(mv["holm_p_value"])
            supported = holm_p < 0.05
            total_primary += 1
            if supported:
                total_supported += 1
            primary.append({
                "name": comp_name,
                "metric": metric_key,
                "mean": _finite_or_none(mv["mean"], f"{study_key}/{comp_name}/mean"),
                "ci_low": _finite_or_none(mv["ci_low"], f"{study_key}/{comp_name}/ci_low"),
                "ci_high": _finite_or_none(mv["ci_high"], f"{study_key}/{comp_name}/ci_high"),
                "p_value": _finite_or_none(mv["p_value"], f"{study_key}/{comp_name}/p_value"),
                "holm_p_value": _finite_or_none(holm_p, f"{study_key}/{comp_name}/holm_p_value"),
                "n": int(mv["n"]),
                "pos": int(mv["pos"]),
                "neg": int(mv["neg"]),
                "supported": supported,
            })
        clean_primary = []
        for row in raw.get("clean_primary_sensitivity", []):
            clean_primary.append({
                "name": row["name"],
                "metric": row["metric"],
                "clean_n": int(row["clean_n"]),
                "clean_mean": _finite_or_none(row["clean_mean"], f"{study_key}/{row['name']}/clean_mean"),
                "clean_ci_low": _finite_or_none(row["clean_ci_low"], f"{study_key}/{row['name']}/clean_ci_low"),
                "clean_ci_high": _finite_or_none(row["clean_ci_high"], f"{study_key}/{row['name']}/clean_ci_high"),
                "clean_holm_p": _finite_or_none(row["clean_holm_p"], f"{study_key}/{row['name']}/clean_holm_p"),
            })
        studies[study_key] = {
            "experiment_id": raw["experiment_id"],
            "seed_range": sr,
            "planned_trials": planned,
            "complete_trials": complete,
            "clean_seed_count": int(raw["clean_seed_count"]),
            "baseline": raw["baseline"],
            "scenario_order": raw["scenario_order"],
            "scenarios": scenarios,
            "primary": primary,
            "clean_primary": clean_primary,
        }
    return {
        "source": {
            "path": MAS_UPGRADE_PATH,
            "sha256": sha256_file(path),
            "report_29_31": "reports/disclosure_mas_upgrade_20261125/README.md",
            "report_32": "reports/disclosure_mas_upgrade_20261125/README_sec32.md",
            "propositions": "docs/disclosure_dynamic_propositions.md",
            "contract": "docs/disclosure_research_contract.md",
        },
        "totals": {
            "studies": len(MAS_UPGRADE_STUDIES),
            "runs": total_runs,
            "planned": total_planned,
            "primary_total": total_primary,
            "supported": total_supported,
            "unresolved": total_primary - total_supported,
            "parity_mismatches": parity_mismatches,
            "seed_range": f"{seed_min}-{seed_max}",
        },
        "studies": studies,
    }


def png_size(path: Path) -> tuple[int, int]:
    """Width and height from the PNG header."""
    with path.open("rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", head[16:24])


def sandbox_payload(root: Path, revision: str) -> dict:
    """Sandbox data, checked against the committed files it was exported from."""
    data = root / "showcase" / "data"
    firms = read_json(data / "sandbox_firms.json")
    stage_f = read_json(data / "stage_f.json")
    simplification = read_json(data / "simplification.json")
    source = stage_f["source"]
    analysis_path = root / source["path"]
    if sha256_file(analysis_path) != source["sha256"]:
        raise ValueError(f"stage F numbers no longer match {source['path']}; rerun the export")
    if read_json(analysis_path)["experiment_id"] != source["experiment_id"]:
        raise ValueError("stage F experiment_id disagrees with its analysis file")
    if sha256_file(root / firms["experiment_spec"]) != firms["experiment_spec_sha256"]:
        raise ValueError("sandbox firms were exported from a different stage F specification")
    ai = firms["domains"]["ai_washing"]
    if sha256_file(root / ai["domain_file"]) != ai["domain_file_sha256"]:
        raise ValueError("the AI-washing domain file changed after the export")
    for key, domain in firms["domains"].items():
        columns = domain["firms"]
        # Synthetic firms only: no names or identifiers beyond the model's own firm order.
        if set(columns) != SANDBOX_FIRM_FIELDS:
            raise ValueError(f"unexpected sandbox firm fields in {key}: {sorted(set(columns) ^ SANDBOX_FIRM_FIELDS)}")
        lengths = {len(columns[name]) for name in SANDBOX_FIRM_FIELDS - {"pending", "full_model_decision"}}
        if lengths != {200}:
            raise ValueError(f"{key}: every firm column needs 200 rows")
    mas_upgrade = export_mas_upgrade(root)
    return {
        "schema_version": 2,
        "source_commit": revision,
        "firms": firms,
        "stage_f": stage_f,
        "simplification": simplification,
        "mas_upgrade": mas_upgrade,
    }


def inline(template: str, replacements: dict[str, str]) -> str:
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Template marker missing or duplicated: {marker}")
        if marker.startswith("/* ") and not marker.endswith("_CSS */") and "</script" in value.lower():
            raise ValueError(f"inline script for {marker} contains a closing script tag")
        template = template.replace(marker, value)
    if re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', template):
        raise ValueError("Exhibit must not require third-party scripts/styles/fonts")
    return template


def serialize(payload: dict) -> str:
    # Escape raw-text HTML delimiters even if a report string contains markup.
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def build(output: Path, root: Path = ROOT) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "master"
    snapshot = {
        "schema_version": "1.0.0", "source_commit": revision,
        "report_date": "2026-08-14", "mode": "archived_read_only",
        "notice": "Historical v13 failed 18/22; frozen v15 passed 22/22 on 2026-09-26. Neither is a rerun of merged source.",
        "reports": [export_report(root, key, directory) for key, directory in REPORTS.items()],
    }
    stage1 = root / "showcase" / "stage1"
    stage1_html = inline((stage1 / "index.html").read_text(encoding="utf-8"), {
        "/* SHOWCASE_CSS */": (stage1 / "styles.css").read_text(encoding="utf-8"),
        "/* SHOWCASE_TEXT */": (stage1 / "i18n.js").read_text(encoding="utf-8"),
        "/* SHOWCASE_JS */": (stage1 / "app.js").read_text(encoding="utf-8"),
        '{"reports":[]}': json.dumps(snapshot, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
    })
    sandbox = root / "showcase" / "sandbox"
    payload = sandbox_payload(root, revision)
    sandbox_html = inline((sandbox / "index.html").read_text(encoding="utf-8"), {
        "/* SANDBOX_CSS */": (sandbox / "styles.css").read_text(encoding="utf-8"),
        "/* SANDBOX_CORE */": (sandbox / "core.js").read_text(encoding="utf-8"),
        "/* SANDBOX_DESK */": (sandbox / "desk.js").read_text(encoding="utf-8"),
        "/* SANDBOX_TEXT */": (sandbox / "i18n.js").read_text(encoding="utf-8"),
        "/* SANDBOX_APP */": (sandbox / "app.js").read_text(encoding="utf-8"),
        '{"sandbox":null}': serialize(payload),
    })
    card = root / "showcase" / "social" / "social-card.png"
    if png_size(card) != SOCIAL_CARD_SIZE:
        raise ValueError(f"{card.name} is {png_size(card)}, but the pages declare {SOCIAL_CARD_SIZE}")
    (output / "stage1").mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(sandbox_html, encoding="utf-8")
    (output / "sandbox-data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "stage1" / "index.html").write_text(stage1_html, encoding="utf-8")
    (output / "stage1" / "data.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(card, output / card.name)
    (output / ".nojekyll").touch()
    print(f"Built {output / 'index.html'}: {len(sandbox_html.encode()):,} bytes (sandbox); "
          f"{output / 'stage1' / 'index.html'}: {len(stage1_html.encode()):,} bytes; no simulation executed")
    return {"stage1": snapshot, "sandbox": payload}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.output.resolve())
