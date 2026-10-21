"""Shared readers for the learning-firm redo archives (contract section 23): tables, runs and the coverage gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENARIOS = ("B", "I", "F", "IF")
VERSIONS = ("A", "V")


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_spec(path: Path) -> dict:
    return read(path)


def complete(root: Path, name: str) -> Path | None:
    """Last attempt folder under root whose `name` file reports status complete (or exists, for replays)."""
    done = []
    for folder in sorted(root.glob("attempt_*")):
        path = folder / name
        if path.exists() and (name == "replay.json" or read(path).get("status") == "complete"):
            done.append(folder)
    return done[-1] if done else None


def table_folder(output: Path, replicate: int, version: str, scenario: str | None) -> Path | None:
    root = output / f"rep_{replicate:02d}" / version / ("pre" if scenario is None else scenario)
    return complete(root, "summary.json")


def ql_run(table: Path, seed: int) -> Path:
    return table / "validation" / f"seed_{seed}" / "q_learning"


def npv_run(output: Path, scenario: str, seed: int, rule: str) -> Path | None:
    return complete(output / "npv" / scenario / f"seed_{seed}" / rule, "manifest.json")


def replay_record(output: Path, replicate: int, version: str, scenario: str, seed: int) -> dict | None:
    folder = complete(output / "replay" / f"rep_{replicate:02d}" / version / scenario / f"seed_{seed}", "replay.json")
    return None if folder is None else read(folder / "replay.json")


def metrics(run: Path) -> dict:
    return read(run / "metrics.json")


def outcome(run: Path) -> dict:
    m = metrics(run)
    actions = m["actions"]
    return {"T": m["true_change"]["equal_weight_sd"], "W": actions["wash"] / actions["decisions"],
            "improve_share": actions["improve"] / actions["decisions"],
            "maintain_share": actions["maintain"] / actions["decisions"],
            "wash_share_by_size": actions["wash_share_by_size"],
            "T_cap": m["true_change"].get("fixed_start_cap_weight_sd")}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_gate(table: Path, seeds: list[int]) -> dict:
    """Contract 23.5 on the table's evaluation runs: (a) unseen <= 5% per world and pooled; (b) decisions in states
    with a legal option under 20 training updates <= 5% pooled and <= 10% per world; globally and per group."""
    with np.load(table / "frozen_model.npz") as model:
        counts = model["counts"]
    trained = counts.sum(axis=2) > 0
    pooled = {}
    worlds = []
    for seed in seeds:
        with np.load(ql_run(table, seed) / "result.npz") as result:
            groups = result["learning_groups"][1:].ravel()
            states = result["learning_states"][1:].ravel()
            legal = result["valid_actions"][1:].reshape(groups.size, -1)
        unseen = ~trained[groups, states]
        under = ((counts[groups, states] < 20) & legal).any(axis=1)
        world = {"seed": seed, "scopes": {}}
        for scope in ["all"] + [int(g) for g in np.unique(groups)]:
            member = np.ones_like(groups, dtype=bool) if scope == "all" else groups == scope
            n = int(member.sum())
            row = {"decisions": n, "unseen": int(unseen[member].sum()), "undercovered": int(under[member].sum())}
            world["scopes"][str(scope)] = {**row, "unseen_share": row["unseen"] / n, "undercovered_share": row["undercovered"] / n}
            total = pooled.setdefault(str(scope), {"decisions": 0, "unseen": 0, "undercovered": 0})
            for key in total:
                total[key] += row[key]
        world["pass"] = all(s["unseen_share"] <= .05 and s["undercovered_share"] <= .10 for s in world["scopes"].values())
        worlds.append(world)
    for row in pooled.values():
        row["unseen_share"] = row["unseen"] / row["decisions"]
        row["undercovered_share"] = row["undercovered"] / row["decisions"]
    pooled_pass = all(row["unseen_share"] <= .05 and row["undercovered_share"] <= .05 for row in pooled.values())
    return {"status": "PASS" if pooled_pass and all(w["pass"] for w in worlds) else "FAIL",
            "pooled": pooled, "worlds": worlds,
            "max_pooled_undercovered_share": max(row["undercovered_share"] for row in pooled.values()),
            "max_world_undercovered_share": max(s["undercovered_share"] for w in worlds for s in w["scopes"].values()),
            "max_unseen_share": max([row["unseen_share"] for row in pooled.values()]
                                    + [s["unseen_share"] for w in worlds for s in w["scopes"].values()])}
