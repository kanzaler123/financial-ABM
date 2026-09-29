"""Command-line entry point for an auditable market run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .stage2 import create_market_harness, load_simulation_config


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
        help="Stage 1 or Stage 2 JSON configuration",
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
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_simulation_config(args.config)
    revision = resolve_code_revision()
    journal = args.output.parent / f".{args.output.name}.events.jsonl"
    result = create_market_harness(config, event_log=journal, code_revision=revision).run()
    result.write(args.output, config, code_revision=revision)
    print(json.dumps(result.summary(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
