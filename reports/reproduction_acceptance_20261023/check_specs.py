"""Reproduction acceptance: check the committed specs against the identities recorded in the research contract.

Run from the repository root, after running the spec scripts (they rewrite configs/*.json in place, so a clean
`git status` afterwards already shows the rewrite reproduced the committed bytes):

    python reports/reproduction_acceptance_20261023/check_specs.py [--out check_specs.json]

For every spec with an identity recorded in docs/disclosure_research_contract.md this script reports
- the SHA-256 of the spec file with CRLF converted to LF, against the recorded "配置文件 SHA256（LF）";
- the experiment_id, where it can be computed: for the learning-firm redo it is the SHA-256 of the canonical JSON
  of the spec itself (run_redo.load_spec), so the public file suffices; for every other spec it is
  abm.disclosure.experiments.freeze_spec, which embeds the full content of the vendor-derived domain file, the
  source hashes of src/abm and the Python, numpy and scipy versions, so it is computed only where the domain file
  exists and is reported as blocked otherwise;
- the source digest (SHA-256 of the canonical JSON of abm.disclosure.runner.source_hashes()) of the checked-out
  src/abm, against the digests the contract records for sections 21-23, and the same digest computed from the
  committed blobs at every commit that changed src/abm (about a minute: it reads each blob with git show), once
  as committed and once with src/abm/mechanism_validation.py converted to CRLF. The recorded identities were
  computed in a working copy where that one file had CRLF line endings (README section 5); with the conversion
  the commits frozen before the runs reproduce the recorded digests.
Nothing in the repository is written except the --out file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abm.manifest import canonical_json_bytes  # noqa: E402

DOMAIN = ROOT / "configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json"
DOMAIN_SHA256 = "c3dd482401afcab2633d57fd58c3945e956435d4daf44ad1e2bfcccb441ef907"  # configs/disclosure_learning_redo_*_v1.json

# spec: (contract line, recorded LF SHA-256 or None, recorded experiment_id or None, id kind)
# Ids not in the contract come from the committed analysis file named in the comment.
RECORDED = {
    "configs/disclosure_stage_f_confirmatory_v1.json": (
        376, "86b6bc771603cfa076e61cf27d37e957ded828eb9ac6fe6bf981f5579fe78832",
        "90ee860663d4e081d9de4c73bcec40de9083902166c6e470d8f430e562d85d57", "freeze_spec"),
    "configs/disclosure_stage_f_development_v1.json": (  # id: reports/disclosure_stage_f_20261014/development_v1_analysis.json
        None, None, "analysis:reports/disclosure_stage_f_20261014/development_v1_analysis.json", "freeze_spec"),
    "configs/disclosure_decay_grid_v1.json": (
        396, "abe331e21dacacf3f29ddf705c41e951fc737120b454b4fbb4d7509089ef96be",
        "193858150c88741272a03ba3bc947c461781c12a7c63dad6fcad333b6f5d7740", "freeze_spec"),
    "configs/disclosure_feedback_strong_v1.json": (
        436, "a54d422a52048c1d4ca8259afea1ea0c24e985e4fc46497d2147932780b0e16d",
        "17b7394fec7624476ee622a836c320b5af97aea8db65b9ca411b0b477e2372aa", "freeze_spec"),
    "configs/disclosure_robustness_v1.json": (
        603, "b3faa48c86fd0747517d586bf522fc46f383c8a101057074bbd90e2d540dd536",
        "949c2eb4cdc25ac0032672cbe75e7bd240f49e93f998f1de1bd542671e044022", "freeze_spec"),
    "configs/disclosure_robustness_check_v1.json": (
        603, None, "db909d3c1ca870994a10d3b36ce6a59a19ecd0e1e2b433da9b17bf6bd441988a", "freeze_spec"),
    "configs/disclosure_targeted_audits_confirmatory_v1.json": (
        722, "f5138e917fde4e6cd1feded88e5a1c7adde4109aeb9560ff9bbd40aa15864913",
        "06bec14854a80b5c863b5bbea6d9df2d68daa96d26888f05f64dfb2ba5522a32", "freeze_spec"),
    "configs/disclosure_targeted_audits_development_v1.json": (
        722, None, "ae5386ac9078571fec398a56dcfcb798271e13c19feca4b76e20fdc56ceaac18", "freeze_spec"),
    "configs/disclosure_assurance_confirmatory_v1.json": (
        842, "61a9959a21f4a3ab59d0462bd65c40dcf20de269af9039fa234a623e5b320a35",
        "35c4f76700f76a389caf05e5a2dd1b896ea9b52ddaebc8142763cf35ce5e339a", "freeze_spec"),
    "configs/disclosure_assurance_development_v1.json": (
        842, None, "3a98d1f16abd4b63ef0bd562288428f587fe9fd6035eab9c5576dd9fdc7a4a84", "freeze_spec"),
    "configs/disclosure_learning_redo_confirmatory_v1.json": (
        951, "27782611b4a9b24c319d04f096970f2a45362c8d6caf869ed894daf9df9348b7",
        "b4430136e1fefc0be2e57678edd0de30e7516f4f0eee50c255642898d6a06062", "canonical_spec"),
    "configs/disclosure_learning_redo_development_v1.json": (
        951, None, "74b6a503f618b88a32b50bd47eb2b27320018dadda9e57daa492dddff311728f", "canonical_spec"),
    "configs/disclosure_joint_robustness_v1.json": (
        1028, "52b0906f137a063b046c9368054ef305cbce7f32aeccca00e946d83a6e96d4ad",
        "c856a0cd3732aa8243420f1e8bc50af2b0adf46a6f20cdeb35941956169b6765", "freeze_spec"),
    "configs/disclosure_joint_robustness_check_v1.json": (
        1028, "53464bf4054a5726318ada212d7a8b34acc49ffe360a1b6dba949de0f7623eff",
        "f29cf7364f2a6d0906564773598bf1deb4e5b9fdf703ac7b58b48f1a5049eeb4", "freeze_spec"),
}
CRLF_FILE = "abm/mechanism_validation.py"  # CRLF in the working copy that froze every identity since stage F
SOURCE_DIGESTS = {"21 targeted audits": "a94bed8bc5bc570cc8918c7b030fd9594c323fd02b074036ba4ccc30819c6ef0",
                  "22 assurance": "99966d0cf9ab5942aaf08eefa9b47e9d514f8c2d1a2589409ff484f579c89e81",
                  "23 learning redo": "7a0732915ffb9ff432c54d6dd6eb60aad98c1839c7b2cbc20db2190a5c409de5"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recorded_id(value: str) -> str:
    if value.startswith("analysis:"):
        return json.loads((ROOT / value.removeprefix("analysis:")).read_text(encoding="utf-8"))["experiment_id"]
    return value


def check_spec(path: str) -> dict:
    line, recorded_sha, recorded, kind = RECORDED[path]
    data = (ROOT / path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    entry = {"spec": path, "contract_line": line, "file_sha256_lf": sha256(lf),
             "file_sha256_crlf": sha256(lf.replace(b"\n", b"\r\n")), "recorded_sha256": recorded_sha}
    # "lf" when the record is the hash of the LF file, "crlf" when it is the hash of the same text with CRLF
    entry["sha256_matches"] = None if recorded_sha is None else (
        "lf" if entry["file_sha256_lf"] == recorded_sha else "crlf" if entry["file_sha256_crlf"] == recorded_sha else False)
    entry["recorded_experiment_id"] = recorded_id(recorded)
    if kind == "canonical_spec":
        entry["experiment_id"] = sha256(canonical_json_bytes(json.loads(data.decode("utf-8"))))
        entry["experiment_id_status"] = "computed from the public spec"
    elif DOMAIN.exists():
        from abm.disclosure.experiments import freeze_spec

        entry["experiment_id"] = freeze_spec(ROOT / path)["experiment_id"]
        entry["experiment_id_status"] = "computed with freeze_spec (domain file present)"
    else:
        entry["experiment_id"] = None
        entry["experiment_id_status"] = "blocked: freeze_spec embeds the vendor-derived domain file, which is not in git"
    entry["experiment_id_matches"] = None if entry["experiment_id"] is None else (
        entry["experiment_id"] == entry["recorded_experiment_id"])
    return entry


def digest_history() -> list[dict]:
    """The source digest of src/abm at every commit that changed it, from the committed blobs (needs git)."""
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout  # noqa: E731
    rows = []
    for line in git("log", "--format=%h %ad", "--date=short", "--", "src/abm").decode().splitlines():
        commit, date = line.split()
        files = [f for f in git("ls-tree", "-r", "--name-only", commit, "src/abm").decode().split() if f.endswith(".py")]
        blobs = {f.removeprefix("src/"): git("show", f"{commit}:{f}") for f in sorted(files)}
        hashes = {name: sha256(blob) for name, blob in blobs.items()}
        digest = sha256(canonical_json_bytes(hashes))
        row = {"commit": commit, "date": date, "files": len(hashes), "digest": digest,
               "matches": [name for name, value in SOURCE_DIGESTS.items() if value == digest]}
        if CRLF_FILE in blobs:
            variant = {**hashes, CRLF_FILE: sha256(blobs[CRLF_FILE].replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))}
            row["digest_with_crlf_file"] = sha256(canonical_json_bytes(variant))
            row["matches_with_crlf_file"] = [name for name, value in SOURCE_DIGESTS.items()
                                             if value == row["digest_with_crlf_file"]]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "check_specs.json")
    args = parser.parse_args()
    from abm.disclosure.runner import source_hashes

    sources = source_hashes()
    digest = sha256(canonical_json_bytes(sources))
    import numpy
    import scipy

    report = {
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                        "platform": sys.platform},
        "domain_file": {"path": DOMAIN.relative_to(ROOT).as_posix(), "present": DOMAIN.exists(),
                        "recorded_sha256": DOMAIN_SHA256,
                        "sha256": sha256(DOMAIN.read_bytes()) if DOMAIN.exists() else None},
        "source_digest": {"checked_out": digest, "files": len(sources),
                          "recorded": {name: {"digest": value, "matches_checked_out": value == digest}
                                       for name, value in SOURCE_DIGESTS.items()},
                          "history": digest_history()},
        "specs": [check_spec(path) for path in RECORDED],
    }
    for entry in report["specs"]:
        print(f"{entry['spec']:<60} sha256 {str(entry['sha256_matches']):<5} "
              f"experiment_id {str(entry['experiment_id_matches']):<5} {entry['experiment_id_status']}")
    print(f"source digest of checked-out src/abm {digest[:16]}...; recorded: "
          + ", ".join(f"{k} {'match' if v['matches_checked_out'] else 'no match'}"
                      for k, v in report["source_digest"]["recorded"].items()))
    history = report["source_digest"]["history"]
    print(f"commits that changed src/abm: {len(history)}; reproducing a recorded digest: "
          + (", ".join(f"{row['commit']} ({', '.join(row['matches'])})" for row in history if row["matches"]) or "none")
          + f"; with {CRLF_FILE} as CRLF: "
          + (", ".join(f"{row['commit']} ({', '.join(row['matches_with_crlf_file'])})"
                       for row in history if row.get("matches_with_crlf_file")) or "none"))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
