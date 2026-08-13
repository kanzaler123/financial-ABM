"""Create and verify immutable Stage 1 formal-acceptance manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .manifest import canonical_json_bytes, sha256_file
from .runner import resolve_code_revision

FREEZE_SCHEMA_VERSION = "1.0.0"


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"freeze path escapes project root: {path}") from exc


def collect_freeze_paths(
    project_root: str | Path,
    *,
    formal_config: str | Path,
    formal_protocol: str | Path,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    required = {
        Path(formal_config).resolve(),
        Path(formal_protocol).resolve(),
        root / "pyproject.toml",
        root / "requirements-lock.txt",
        root / "web" / "package.json",
        root / "web" / "package-lock.json",
    }
    required.update((root / "src" / "abm").glob("*.py"))
    for pattern in ("*.ts", "*.tsx", "*.css"):
        required.update((root / "web" / "src").glob(pattern))
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "freeze inputs are missing: " + ", ".join(missing)
        )
    return tuple(
        sorted(required, key=lambda path: _relative_path(path, root))
    )


def build_freeze_manifest(
    project_root: str | Path,
    *,
    formal_config: str | Path,
    formal_protocol: str | Path,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    config_path = Path(formal_config).resolve()
    protocol_path = Path(formal_protocol).resolve()
    paths = collect_freeze_paths(
        root,
        formal_config=config_path,
        formal_protocol=protocol_path,
    )
    hashes = {
        _relative_path(path, root): sha256_file(path)
        for path in paths
    }
    identity = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "formal_config": _relative_path(config_path, root),
        "formal_protocol": _relative_path(protocol_path, root),
        "sha256": hashes,
    }
    freeze_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return {
        **identity,
        "freeze_id": freeze_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": resolve_code_revision(),
        "formal_run_status_at_freeze": {
            "formal_metrics_read": False,
            "formal_simulations_run": False,
        },
    }


def write_freeze_manifest(
    output: str | Path,
    manifest: dict[str, object],
) -> Path:
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def verify_freeze_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    formal_config: str | Path | None = None,
    formal_protocol: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported freeze manifest schema")
    hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("freeze manifest must contain file hashes")
    if formal_config is not None:
        actual_config = _relative_path(Path(formal_config), root)
        if payload.get("formal_config") != actual_config:
            raise ValueError("freeze manifest formal config does not match")
    if formal_protocol is not None:
        actual_protocol = _relative_path(Path(formal_protocol), root)
        if payload.get("formal_protocol") != actual_protocol:
            raise ValueError("freeze manifest formal protocol does not match")
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted(hashes.items()):
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"freeze manifest path escapes project root: {relative}")
        if not candidate.is_file():
            missing.append(str(relative))
        elif sha256_file(candidate) != str(expected):
            mismatched.append(str(relative))
    identity = {
        "schema_version": payload["schema_version"],
        "formal_config": payload.get("formal_config"),
        "formal_protocol": payload.get("formal_protocol"),
        "sha256": hashes,
    }
    expected_freeze_id = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    if payload.get("freeze_id") != expected_freeze_id:
        raise ValueError("freeze manifest identity hash is invalid")
    result = {
        "verified": not missing and not mismatched,
        "freeze_id": expected_freeze_id,
        "manifest": str(path),
        "checked_files": len(hashes),
        "missing_files": missing,
        "mismatched_files": mismatched,
    }
    if not result["verified"]:
        raise ValueError(
            "freeze verification failed; "
            f"missing={missing}, mismatched={mismatched}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--project-root", type=Path, default=Path.cwd())
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--protocol", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=Path.cwd())
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--protocol", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        manifest = build_freeze_manifest(
            args.project_root,
            formal_config=args.config,
            formal_protocol=args.protocol,
        )
        output = write_freeze_manifest(args.output, manifest)
        print(output)
        return 0
    result = verify_freeze_manifest(
        args.manifest,
        project_root=args.project_root,
        formal_config=args.config,
        formal_protocol=args.protocol,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
