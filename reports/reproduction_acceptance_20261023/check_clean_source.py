"""Before freezing a new experiment: the source and identity must be what a clean clone of HEAD would produce.

Run from the working copy that will freeze and run the experiment, with the spec committed:

    python reports/reproduction_acceptance_20261023/check_clean_source.py --spec configs/<new spec>.json
    python reports/reproduction_acceptance_20261023/check_clean_source.py --source-only   # source check alone

Checks, each of which must pass (exit code 1 otherwise):
1. every src/abm/**/*.py in the working tree (the files source_hashes() hashes) is tracked and byte-identical to
   its HEAD blob, which is LF for *.py (.gitattributes), and no tracked file is missing; a file that differs only
   in line endings is reported as such, but still fails;
2. the source digest of the working tree equals the digest of the HEAD blobs, i.e. of a clean clone;
3. for each --spec: the spec file is tracked and byte-identical to HEAD, and freeze_spec gives the same identity,
   experiment_id included, in the working copy and in a clean export of HEAD (the domain file named by the spec is
   copied into the export, because the vendor-derived one is not in git); without the domain file this check
   cannot run and fails.
The checks compare bytes and the whole frozen identity; nothing is normalised or skipped. Nothing is written in
the working copy (bytecode writing is switched off for the probes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from historical_source import export, probe  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def head_blob(root: Path, relative: str) -> bytes | None:
    result = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{relative}"], capture_output=True)
    return result.stdout if result.returncode == 0 else None


def check_source(root: Path) -> tuple[list[str], dict, dict]:
    tracked = {p for p in git(root, "ls-tree", "-r", "--name-only", "HEAD", "src/abm").decode().split() if p.endswith(".py")}
    present = {p.relative_to(root).as_posix() for p in (root / "src" / "abm").rglob("*.py")}
    problems = [f"tracked but missing: {p}" for p in sorted(tracked - present)]
    problems += [f"not committed: {p}" for p in sorted(present - tracked)]
    working, clean = {}, {}
    for relative in sorted(present | tracked):
        key = relative.removeprefix("src/")
        blob = head_blob(root, relative)
        if blob is not None:
            clean[key] = sha256(blob)
        if relative in present:
            data = (root / relative).read_bytes()
            working[key] = sha256(data)
            if blob is not None and data != blob:
                same_text = data.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
                problems.append(f"{'line endings differ from HEAD (CRLF in the working tree)' if same_text else 'uncommitted change'}: {relative}")
    return problems, working, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE.parents[1], help="working copy to check (default: this one)")
    parser.add_argument("--spec", action="append", default=[], help="committed spec path, relative to the root")
    parser.add_argument("--source-only", action="store_true", help="check the source alone (no spec, no identity)")
    args = parser.parse_args()
    if not args.spec and not args.source_only:
        parser.error("pass --spec (before a freeze) or --source-only")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from abm.manifest import canonical_json_bytes  # noqa: PLC0415

    commit = git(root, "rev-parse", "--short", "HEAD").decode().strip()
    failures, working, clean = check_source(root)
    digest_working = sha256(canonical_json_bytes(working))
    digest_clean = sha256(canonical_json_bytes(clean))
    print(f"HEAD {commit}; src/abm: {len(working)} files in the working tree, {len(clean)} at HEAD")
    print(f"source digest: working tree {digest_working}; clean clone of HEAD {digest_clean}; "
          + ("identical" if digest_working == digest_clean else "DIFFERENT"))
    if digest_working != digest_clean and not failures:
        failures.append("source digests differ")
    for spec in args.spec:
        blob = head_blob(root, spec)
        if blob is None or (root / spec).read_bytes() != blob:
            failures.append(f"spec not committed or changed since HEAD: {spec}")
            continue
        domain = (root / spec).parent / json.loads(blob)["domain"]
        if not domain.is_file():
            failures.append(f"{spec}: domain file {domain} not found, identity not checked")
            continue
        here = probe(root, spec)["frozen"]
        with tempfile.TemporaryDirectory() as tmp:
            export("HEAD", Path(tmp), root)
            target = (Path(tmp) / spec).parent / json.loads(blob)["domain"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(domain, target)
            there = probe(Path(tmp), spec)["frozen"]
        differing = sorted(key for key in set(here) | set(there) if here.get(key) != there.get(key))
        print(f"{spec}: experiment_id here {here['experiment_id']}, clean clone {there['experiment_id']}; "
              + ("identical in every field" if not differing else f"DIFFERENT in {differing}"))
        failures += [f"{spec}: frozen identity differs in {differing}"] if differing else []
        print(f"  environment in the identity: python {here['python_version']}, numpy {here['numpy_version']}, "
              f"scipy {here['scipy_version']}")
    for line in failures:
        print("FAIL " + line)
    if failures:
        raise SystemExit(1)
    print("ok: the source" + (" and identity are" if args.spec else " is") + " what a clean clone of HEAD produces")


if __name__ == "__main__":
    main()
