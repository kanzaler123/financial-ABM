"""Command-line entry point for an auditable Stage 1 acceptance run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .config import load_stage1_config
from .harness import run_stage1


def resolve_code_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return f"{commit}+dirty" if dirty else commit
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1.json"),
        help="Stage 1 JSON configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; existing directories are not overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_stage1_config(args.config)
    result = run_stage1(config)
    result.write(args.output, config, code_revision=resolve_code_revision())
    print(json.dumps(result.summary(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
