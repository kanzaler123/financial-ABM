"""Reproducible static visualizations for a completed Stage 1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#2F5D8C"
GOLD = "#C9941A"
ORANGE = "#D97706"
PINK = "#B65378"
OLIVE = "#788542"
STRATEGY_COLORS = {
    "value": BLUE,
    "trend": GOLD,
    "noise": PINK,
}
STRATEGY_HATCHES = {
    "value": "",
    "trend": "//",
    "noise": "..",
}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(figure: plt.Figure, base_path: Path) -> tuple[Path, Path]:
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def plot_stage1_overview(run_dir: str | Path) -> tuple[Path, Path]:
    """Plot market path, activity, strategy composition, and invariants."""
    run_path = Path(run_dir)
    config = _load_json(run_path / "config.json")
    if not isinstance(config, dict):
        raise TypeError("config.json must contain an object")
    with np.load(run_path / "state_arrays.npz") as arrays:
        prices = arrays["prices"]
        fundamentals = arrays["fundamentals"]
        volumes = arrays["volumes"]
        total_cash = arrays["total_cash"]
        total_shares = arrays["total_shares"]
        counts = arrays["strategy_counts"]

    days = np.arange(prices.size)
    trade_days = np.arange(1, prices.size)
    composition = counts / counts.sum(axis=1, keepdims=True)
    cash_relative_error = np.maximum(
        np.abs(total_cash - total_cash[0]) / total_cash[0], 1e-18
    )
    share_relative_error = np.maximum(
        np.abs(total_shares - total_shares[0]) / total_shares[0], 1e-18
    )

    _apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        bottom=0.07,
        top=0.86,
        hspace=0.34,
        wspace=0.16,
    )
    figure.suptitle(
        "Stage 1 Artificial Market — Simulation Overview",
        x=0.025,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.025,
        0.925,
        f"{config['population_size']:,} traders · {config['trading_days']} daily steps · seed {config['seed']}",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=10,
    )

    price_axis = axes[0, 0]
    price_axis.plot(days, prices, color=BLUE, linewidth=2.2, label="Market price")
    price_axis.plot(
        days,
        fundamentals,
        color=GOLD,
        linewidth=1.8,
        linestyle="--",
        label="Fundamental value",
    )
    for announcement in config["announcements"]:
        day = int(announcement["day"])
        price_axis.axvline(day, color=MUTED, linewidth=0.8, linestyle=":")
        price_axis.annotate(
            f"{float(announcement['fundamental_delta']):+g}",
            xy=(day, fundamentals[day]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=MUTED,
            fontsize=8,
        )
    price_axis.set_title("Market price and fundamental value")
    price_axis.set_xlabel("Trading day")
    price_axis.set_ylabel("Price level")
    price_axis.legend(loc="upper left", ncols=2)
    _finish_axis(price_axis)

    volume_axis = axes[0, 1]
    volume_axis.bar(
        trade_days,
        volumes,
        width=1.0,
        color=ORANGE,
        edgecolor="#9A5504",
        linewidth=0.2,
    )
    volume_axis.axhline(
        float(volumes.mean()),
        color=INK,
        linestyle="--",
        linewidth=1.2,
        label=f"Mean {volumes.mean():.1f}",
    )
    volume_axis.set_title("Executed trading volume")
    volume_axis.set_xlabel("Trading day")
    volume_axis.set_ylabel("Shares per day")
    volume_axis.legend(loc="upper right")
    _finish_axis(volume_axis)

    strategy_axis = axes[1, 0]
    strategy_axis.stackplot(
        days,
        composition[:, 0],
        composition[:, 1],
        composition[:, 2],
        labels=("Value", "Trend", "Noise"),
        colors=(
            STRATEGY_COLORS["value"],
            STRATEGY_COLORS["trend"],
            STRATEGY_COLORS["noise"],
        ),
        alpha=0.88,
        linewidth=0.6,
        edgecolor="white",
    )
    for audit in _load_json(run_path / "learning_audit.json"):
        strategy_axis.axvline(
            int(audit["day"]), color=INK, linestyle=":", linewidth=0.8, alpha=0.7
        )
    strategy_axis.set_title("Strategy composition after Logit updates")
    strategy_axis.set_xlabel("Trading day")
    strategy_axis.set_ylabel("Share of traders")
    strategy_axis.set_ylim(0, 1)
    strategy_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    strategy_axis.legend(loc="upper left", ncols=3)
    _finish_axis(strategy_axis)

    invariant_axis = axes[1, 1]
    invariant_axis.plot(
        days,
        cash_relative_error,
        color=OLIVE,
        linewidth=1.7,
        label="Cash relative error",
    )
    invariant_axis.plot(
        days,
        share_relative_error,
        color=BLUE,
        linewidth=1.5,
        linestyle="--",
        label="Share relative error",
    )
    invariant_axis.set_yscale("log")
    invariant_axis.set_title("Accounting conservation error")
    invariant_axis.set_xlabel("Trading day")
    invariant_axis.set_ylabel("Absolute relative error (log scale)")
    invariant_axis.legend(loc="upper left")
    _finish_axis(invariant_axis)

    return _save_figure(figure, run_path / "stage1_overview")


def plot_learning_diagnostics(run_dir: str | Path) -> tuple[Path, Path]:
    """Plot strategy fitness and the resulting Logit probabilities."""
    run_path = Path(run_dir)
    audits = _load_json(run_path / "learning_audit.json")
    if not isinstance(audits, list) or not audits:
        raise ValueError("learning_audit.json contains no learning updates")
    update_days = np.array([int(audit["day"]) for audit in audits])
    strategy_names = ("value", "trend", "noise")
    fitness = {
        name: np.array(
            [float(audit["mean_fitness_by_strategy"][name]) for audit in audits]
        )
        for name in strategy_names
    }
    probabilities = {
        name: np.array(
            [float(audit["choice_probabilities"][name]) for audit in audits]
        )
        for name in strategy_names
    }

    _apply_style()
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.85,
        hspace=0.36,
    )
    figure.suptitle(
        "Stage 1 Logit Imitation — Learning Diagnostics",
        x=0.025,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.025,
        0.925,
        "Mean strategy fitness drives the probability used to redraw trader strategies every 60 days.",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=10,
    )

    positions = np.arange(update_days.size)
    width = 0.24
    fitness_axis = axes[0]
    for index, name in enumerate(strategy_names):
        fitness_axis.bar(
            positions + (index - 1) * width,
            fitness[name],
            width,
            label=name.title(),
            color=STRATEGY_COLORS[name],
            edgecolor=INK,
            linewidth=0.5,
            hatch=STRATEGY_HATCHES[name],
        )
    fitness_axis.axhline(0, color=INK, linewidth=1)
    fitness_axis.set_title("Mean fitness by strategy at each update")
    fitness_axis.set_ylabel("Fitness per trading day")
    fitness_axis.set_xticks(positions, [f"Day {day}" for day in update_days])
    fitness_axis.legend(loc="lower left", ncols=3)
    _finish_axis(fitness_axis)

    probability_axis = axes[1]
    bottom = np.zeros(update_days.size)
    for name in strategy_names:
        probability_axis.bar(
            positions,
            probabilities[name],
            bottom=bottom,
            width=0.62,
            label=name.title(),
            color=STRATEGY_COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            hatch=STRATEGY_HATCHES[name],
        )
        bottom += probabilities[name]
    probability_axis.set_title("Logit strategy-choice probability")
    probability_axis.set_ylabel("Probability")
    probability_axis.set_xticks(positions, [f"Day {day}" for day in update_days])
    probability_axis.set_ylim(0, 1)
    probability_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    probability_axis.legend(loc="upper center", ncols=3)
    _finish_axis(probability_axis)

    return _save_figure(figure, run_path / "stage1_learning")


def create_stage1_visualizations(run_dir: str | Path) -> tuple[Path, ...]:
    overview = plot_stage1_overview(run_dir)
    audits = _load_json(Path(run_dir) / "learning_audit.json")
    if isinstance(audits, list) and audits:
        learning = plot_learning_diagnostics(run_dir)
        return (*overview, *learning)
    return overview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="completed Stage 1 output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in create_stage1_visualizations(args.run_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
