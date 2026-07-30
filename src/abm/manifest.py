"""Deterministic run-manifest construction and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .schemas import JsonValue, RunManifest


def canonical_json_bytes(value: Mapping[str, JsonValue]) -> bytes:
    """Encode JSON with a stable byte representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_manifest(
    *,
    config: Mapping[str, JsonValue],
    data_files: Mapping[str, str | Path],
    random_seed: int,
    code_revision: str,
    schema_version: str = "0.1.0",
    agent_models: Mapping[str, str] | None = None,
    prompt_sha256: Mapping[str, str] | None = None,
    graph_version: str | None = None,
    enabled_plugins: tuple[str, ...] = (),
) -> RunManifest:
    """Build the reproducibility identity for a run.

    Runtime timestamps and failures are deliberately left empty. They may be
    attached when execution starts without changing the deterministic run ID.
    """
    config_sha256 = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    data_sha256 = {
        logical_name: sha256_file(path)
        for logical_name, path in sorted(data_files.items())
    }
    identity = {
        "schema_version": schema_version,
        "code_revision": code_revision,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "random_seed": random_seed,
        "agent_models": dict(sorted((agent_models or {}).items())),
        "prompt_sha256": dict(sorted((prompt_sha256 or {}).items())),
        "graph_version": graph_version,
        "enabled_plugins": list(sorted(enabled_plugins)),
    }
    run_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return RunManifest(
        schema_version=schema_version,
        run_id=run_id,
        code_revision=code_revision,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        random_seed=random_seed,
        agent_models=dict(sorted((agent_models or {}).items())),
        prompt_sha256=dict(sorted((prompt_sha256 or {}).items())),
        graph_version=graph_version,
        enabled_plugins=tuple(sorted(enabled_plugins)),
    )

