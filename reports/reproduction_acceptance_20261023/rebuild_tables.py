"""Reproduction acceptance: rebuild the committed report tables from committed analysis files only.

Run from the repository root:

    python reports/reproduction_acceptance_20261023/rebuild_tables.py [--out rebuild_tables.json]

For each report directory below, the script loads the committed analysis JSON, passes it to the table function of
that directory's own script (the code that wrote the committed table), and compares the result with the committed
table byte for byte and after converting CRLF to LF. Nothing in the repository is written except the --out file;
scripts that write their table themselves run with their output folder redirected to a temporary directory. No
local archive (summary.json, records, arrays) is read: the archives are not in git.

Each target runs in its own interpreter, because the report folders share module names (analyze, _no_wmi).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# name: (report folder, script, mode, analysis file, committed table)
# modes: lines = tables(result) joined with "\n" plus a final "\n", the bytes main() writes with newline="\n";
#        text = tables_md(result), written the same way; main = the script's main(run) with HERE redirected;
#        write_tables / write_tables_dest = the script's write_tables(analysis[, dest]) with HERE redirected.
TARGETS = {
    "stage_f_confirmatory": ("disclosure_stage_f_20261014", "tables.py", "main", "confirmatory_v1_analysis.json", "confirmatory_v1_tables.md"),
    "stage_f_development": ("disclosure_stage_f_20261014", "tables.py", "main", "development_v1_analysis.json", "development_v1_tables.md"),
    "decay_grid": ("disclosure_decay_grid_20261016", "analyze.py", "lines", "grid_v1_analysis.json", "grid_v1_tables.md"),
    "feedback_counterfactual": ("disclosure_feedback_counterfactual_20261016", "counterfactual.py", "write_tables", "analysis.json", "tables.md"),
    "feedback_strong": ("disclosure_feedback_strong_20261016", "strong.py", "write_tables", "analysis.json", "tables.md"),
    "feedback_channels": ("disclosure_feedback_channels_20261017", "channels.py", "write_tables", "analysis.json", "tables.md"),
    "feedback_channels_main": ("disclosure_feedback_channels_main_20261017", "channels_main.py", "write_tables_dest", "analysis.json", "tables.md"),
    "robustness": ("disclosure_robustness_20261018", "analyze.py", "lines", "robust_v1_analysis.json", "robust_v1_tables.md"),
    "targeted_audits": ("disclosure_targeted_audits_20261020", "analyze.py", "lines", "audits_v1_analysis.json", "audits_v1_tables.md"),
    "assurance": ("disclosure_assurance_20261021", "analyze.py", "lines", "assurance_v1_analysis.json", "assurance_v1_tables.md"),
    "learning_redo": ("disclosure_learning_redo_20261021", "analyze.py", "text", "redo_v1_analysis.json", "redo_v1_tables.md"),
    "joint_robustness": ("disclosure_joint_robustness_20261023", "analyze.py", "lines", "joint_v1_analysis.json", "joint_v1_tables.md"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_script(folder: Path, script: str):
    import importlib.util

    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"report_{folder.name}", folder / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_one(name: str, scratch: Path) -> bytes:
    """Return the table bytes the directory's own code produces from the committed analysis file."""
    folder_name, script, mode, analysis_name, table_name = TARGETS[name]
    folder = ROOT / "reports" / folder_name
    analysis = json.loads((folder / analysis_name).read_text(encoding="utf-8"))
    module = load_script(folder, script)
    if mode == "lines":
        return ("\n".join(module.tables(analysis)) + "\n").encode("utf-8")
    if mode == "text":
        return module.tables_md(analysis).encode("utf-8")
    module.HERE = scratch
    if mode == "main":
        shutil.copyfile(folder / analysis_name, scratch / analysis_name)
        module.main(analysis_name.removesuffix("_analysis.json"))
    elif mode == "write_tables":
        module.write_tables(analysis)
    elif mode == "write_tables_dest":
        module.write_tables(analysis, scratch)
    return (scratch / table_name).read_bytes()


def compare(name: str) -> dict:
    folder_name, _, _, analysis_name, table_name = TARGETS[name]
    committed = (ROOT / "reports" / folder_name / table_name).read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name, "--scratch", tmp],
                                capture_output=True, cwd=tmp)
        if result.returncode != 0:
            return {"target": name, "table": f"reports/{folder_name}/{table_name}", "status": "error",
                    "stderr_tail": result.stderr.decode("utf-8", "replace").splitlines()[-5:]}
        rebuilt = (Path(tmp) / "rebuilt.bin").read_bytes()
    lf = lambda data: data.replace(b"\r\n", b"\n")  # noqa: E731
    entry = {"target": name, "table": f"reports/{folder_name}/{table_name}",
             "source": f"reports/{folder_name}/{analysis_name}",
             "committed_sha256": sha256(committed), "rebuilt_sha256": sha256(rebuilt),
             "committed_crlf": b"\r\n" in committed, "rebuilt_crlf": b"\r\n" in rebuilt,
             "byte_identical": committed == rebuilt, "identical_after_crlf_to_lf": lf(committed) == lf(rebuilt)}
    if not entry["identical_after_crlf_to_lf"]:
        a, b = lf(committed).decode("utf-8").splitlines(), lf(rebuilt).decode("utf-8").splitlines()
        differing = [i for i in range(max(len(a), len(b))) if (a[i:i + 1] or [None]) != (b[i:i + 1] or [None])]
        entry.update(lines_committed=len(a), lines_rebuilt=len(b), differing_lines=len(differing),
                     first_differences=[{"line": i + 1, "committed": (a[i:i + 1] or [None])[0],
                                         "rebuilt": (b[i:i + 1] or [None])[0]} for i in differing[:3]])
    entry["status"] = "identical" if entry["byte_identical"] else (
        "identical_except_line_endings" if entry["identical_after_crlf_to_lf"] else "different")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "rebuild_tables.json")
    parser.add_argument("--one", choices=sorted(TARGETS), help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        (args.scratch / "rebuilt.bin").write_bytes(build_one(args.one, args.scratch))
        return
    entries = [compare(name) for name in TARGETS]
    for entry in entries:
        print(f"{entry['status']:<30} {entry['table']}")
    report = {"python": platform.python_version(), "platform": sys.platform, "targets": entries}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
