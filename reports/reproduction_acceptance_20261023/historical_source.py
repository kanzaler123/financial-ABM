"""Rebuild the exact source of an existing archive in an independent copy, for reproducing or resuming it.

The identities frozen since stage F were computed in a working copy where src/abm/mechanism_validation.py had CRLF
line endings (git stores it as LF; the content is the same). An archive can only be resumed (`--resume`,
finalize.py) or have its identity recomputed with exactly the source bytes it recorded. Instead of switching line
endings back and forth in the everyday working copy, export the archive's commit into a new directory, restore
CRLF there, and check the complete source map against the archive before using that copy:

    python reports/reproduction_acceptance_20261023/historical_source.py \
        --archive reports/disclosure_targeted_audits_20261020/audits_v1 --commit 3fced38 --dest ../src_3fced38 \
        [--domain configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json]

Checks, each of which must pass (exit code 1 otherwise):
1. the copy's src/abm, hashed by the copy's own abm.disclosure.runner.source_hashes(), equals the archive's
   source_hashes file by file (same file set, same bytes);
2. with --domain, for archives frozen by freeze_spec: freeze_spec run in the copy on the committed spec gives an
   identity equal to the archive's experiment.json in every field, experiment_id included (this is the check
   `--resume` makes); for the learning-firm redo: the spec file's SHA-256, the experiment_id (canonical JSON of the
   spec) and the source digest equal the archive's.
The archive is only read. The source check is not relaxed: files are compared byte for byte, and the CRLF
conversion is applied only to the files given with --crlf (default: src/abm/mechanism_validation.py).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CRLF = ["src/abm/mechanism_validation.py"]

# Runs inside the copy, with PYTHONPATH=<copy>/src; refuses to report if abm is imported from anywhere else.
PROBE = """
import hashlib, json, sys
from pathlib import Path
import abm
copy = Path(sys.argv[1]).resolve()
if Path(abm.__file__).resolve().parents[1] != copy / "src":
    raise SystemExit(f"abm imported from {abm.__file__}, not from the copy")
from abm.disclosure.runner import source_hashes
from abm.manifest import canonical_json_bytes
result = {"source_hashes": source_hashes()}
if len(sys.argv) > 2:
    from abm.disclosure.experiments import freeze_spec
    result["frozen"] = freeze_spec(copy / sys.argv[2])
print(json.dumps(result))
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(commit: str, dest: Path, root: Path = ROOT) -> None:
    """Write the committed tree of `commit` to dest without line-ending conversion."""
    tar = subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                          "archive", "--format=tar", commit], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(dest, filter="data")


def probe(dest: Path, spec: str | None = None) -> dict:
    env = {**os.environ, "PYTHONPATH": str(dest / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    args = [sys.executable, "-c", PROBE, str(dest)] + ([spec] if spec else [])
    result = subprocess.run(args, cwd=dest, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"probe failed in the copy:\n{result.stderr}")
    return json.loads(result.stdout)


def compare_maps(recorded: dict, rebuilt: dict) -> list[str]:
    problems = [f"only in the archive: {name}" for name in sorted(set(recorded) - set(rebuilt))]
    problems += [f"only in the copy: {name}" for name in sorted(set(rebuilt) - set(recorded))]
    problems += [f"different bytes: {name}" for name in sorted(set(recorded) & set(rebuilt)) if recorded[name] != rebuilt[name]]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="archive directory (or its experiment.json)")
    parser.add_argument("--commit", required=True, help="commit whose src/abm the archive was frozen with")
    parser.add_argument("--dest", type=Path, required=True, help="new directory outside this working copy")
    parser.add_argument("--crlf", nargs="*", default=DEFAULT_CRLF, help="repository paths to convert to CRLF")
    parser.add_argument("--domain", type=Path, help="vendor-derived domain file, to check the full frozen identity")
    parser.add_argument("--spec", help="spec path in the copy (default: the archive's own spec path or configs/<name>.json)")
    args = parser.parse_args()

    manifest = args.archive / "experiment.json" if args.archive.is_dir() else args.archive
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    dest = args.dest.resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise SystemExit("--dest must be outside this working copy")
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    export(args.commit, dest)
    for relative in args.crlf:
        path = dest / relative
        data = path.read_bytes()
        if b"\r" in data:
            raise SystemExit(f"{relative} already contains CR at {args.commit}")
        path.write_bytes(data.replace(b"\n", b"\r\n"))
    print(f"exported {args.commit} to {dest}; CRLF restored in: {', '.join(args.crlf) or 'none'}")

    redo = "raw_domain" not in frozen  # the learning-firm redo's orchestrator identity
    spec = args.spec or (frozen["spec"] if redo else f"configs/{frozen['spec']['name']}.json")
    if (args.domain or redo) and not (dest / spec).is_file():
        raise SystemExit(f"{spec} is not in {args.commit}: use the commit that froze the run (the spec and its "
                         "source must both be committed there), or pass --spec")
    if args.domain:
        if redo:
            domain_relative = json.loads((dest / spec).read_text(encoding="utf-8"))["domain"]
            target = dest / domain_relative
        else:
            target = (dest / spec).parent / frozen["spec"]["domain"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.domain, target)
    result = probe(dest, spec if args.domain and not redo else None)

    failures = compare_maps(frozen["source_hashes"], result["source_hashes"])
    print(f"source map: {len(result['source_hashes'])} files, "
          + ("identical to the archive" if not failures else f"{len(failures)} differences"))
    for line in failures:
        print("  " + line)
    if redo:
        from_copy = sha256((dest / spec).read_bytes())
        sys.path.insert(0, str(dest / "src"))
        from abm.manifest import canonical_json_bytes  # noqa: PLC0415

        checks = {"spec_file_sha256": from_copy == frozen["spec_file_sha256"],
                  "experiment_id": sha256(canonical_json_bytes(json.loads((dest / spec).read_text(encoding="utf-8"))))
                  == frozen["experiment_id"],
                  "source_digest": sha256(canonical_json_bytes(result["source_hashes"])) == frozen["source_digest"]}
        for name, ok in checks.items():
            print(f"{name}: {'identical' if ok else 'DIFFERENT'}")
            failures += [] if ok else [name]
    elif args.domain:
        rebuilt = result["frozen"]
        differing = sorted(key for key in set(frozen) | set(rebuilt) if frozen.get(key) != rebuilt.get(key))
        print(f"frozen identity from the copy: experiment_id {rebuilt['experiment_id']}; "
              + ("identical to the archive in every field" if not differing else f"differs in {differing}"))
        failures += differing
    else:
        print("frozen identity: not checked (pass --domain to run freeze_spec in the copy)")
    if failures:
        raise SystemExit(1)
    print(f"ready: run the archive's scripts from {dest} with PYTHONPATH={dest / 'src'}, "
          f"pointing --output at the archive")


if __name__ == "__main__":
    main()
