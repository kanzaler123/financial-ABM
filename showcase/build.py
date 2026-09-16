"""Build a self-contained, read-only exhibit from checked-in reports. No simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
        "notice": "Pre-audit reports, not reruns of current code. Stage 1 is not complete.",
        "reports": [export_report(root, key, directory) for key, directory in REPORTS.items()],
    }
    source = root / "showcase"
    template = (source / "index.html").read_text(encoding="utf-8")
    for marker in ("/* SHOWCASE_CSS */", "/* SHOWCASE_JS */", '{"reports":[]}'):
        if template.count(marker) != 1:
            raise ValueError(f"Template marker missing or duplicated: {marker}")
    # Escape raw-text HTML delimiters even if a report string contains markup.
    serialized = json.dumps(snapshot, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = template.replace("/* SHOWCASE_CSS */", (source / "styles.css").read_text(encoding="utf-8"))
    html = html.replace("/* SHOWCASE_JS */", (source / "app.js").read_text(encoding="utf-8"))
    html = html.replace('{"reports":[]}', serialized)
    if re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', html):
        raise ValueError("Exhibit must not require third-party scripts/styles/fonts")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "data.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"Built {output / 'index.html'}: {len(html.encode()):,} bytes; 3 archived reports; no simulation executed")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.output.resolve())
