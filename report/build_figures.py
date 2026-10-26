"""Draw the report's vector figures from the recorded experiment results.

Matplotlib is used only for visualization. This script does not fit models,
choose parameters, or recompute evaluation scores.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
SLATE = "#263746"
MUTED = "#62717F"
TEAL = "#147D92"
CORAL = "#D66B4B"
GRID = "#E4E9ED"


def _save(fig, path):
    """Save fixed-size vector PDF and high-resolution PNG companions."""
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def _plot_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=5, colors=MUTED)


def _main_results(result, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.235, top=0.72, wspace=0.43)
    left, right = axes
    for ax in axes:
        _plot_style(ax)
        ax.set_ylim(-0.72, 2.55)

    strengths = (1e-5, 1e-4, 1e-3)
    points = np.array([2, 1, 0])
    rows = result["experiments"]
    selected = result["selected"]["name"]
    for strength, y in zip(strengths, points):
        pair = [
            next(
                row
                for row in rows
                if row["kind"] == "logistic"
                and row["mode"] == mode
                and row["lambda"] == strength
            )
            for mode in ("numeric", "engineered")
        ]
        values = [row["holdout"]["f1"] for row in pair]
        left.plot(values, [y, y], color="#BFCBD2", linewidth=2, zorder=2)
        for row, value, color, offset in zip(pair, values, (CORAL, TEAL), (-9, 7)):
            chosen = row["name"] == selected
            left.scatter(
                value,
                y,
                s=80 if chosen else 34,
                marker="*" if chosen else "o",
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            left.annotate(
                f"{value:.4f}",
                (value, y),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    left.set_yticks(points, [r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])
    left.set_ylabel(r"Regularization $\lambda$", color=SLATE, labelpad=9)
    left.set_xlim(0.4135, 0.4255)
    left.set_xticks([0.414, 0.418, 0.422, 0.426])
    left.set_xlim(0.4135, 0.4265)
    left.set_xlabel("Holdout F1 (detail view)", color=SLATE, labelpad=7)
    left.set_title(
        "(a) Better features, a modest gain",
        loc="left",
        y=1.30,
        fontsize=9.3,
        color=SLATE,
        fontweight="bold",
    )
    left.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=CORAL, label="Numeric"),
            Line2D(
                [], [], marker="o", linestyle="none", color=TEAL, label="Engineered"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.03, 1.07),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.2,
        borderpad=0,
    )
    left.text(
        0.98,
        1.01,
        "★ selected on tuning",
        transform=left.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=6.8,
    )

    default = result["selected_default_threshold_holdout"]
    tuned = result["selected_holdout"]
    for key, y in zip(("f1", "precision", "recall"), points):
        values = [default[key], tuned[key]]
        right.plot(values, [y, y], color="#BFCBD2", linewidth=2, zorder=2)
        for value, color, offset in zip(values, (CORAL, TEAL), (-9, 7)):
            right.scatter(
                value,
                y,
                s=34,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            right.annotate(
                f"{value:.3f}",
                (value, y),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    right.set_yticks(points, ["F1", "Precision", "Recall"])
    right.set_xlim(0, 1)
    right.set_xticks([0, 0.25, 0.5, 0.75, 1])
    right.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    right.set_xlabel("Holdout score", color=SLATE, labelpad=7)
    right.set_title(
        "(b) Threshold choice matters more",
        loc="left",
        y=1.30,
        fontsize=9.3,
        color=SLATE,
        fontweight="bold",
    )
    right.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color=CORAL, label="Default: 0.50"
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=TEAL,
                label=f"Tuned: {result['selected']['threshold']:.2f}",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.03, 1.07),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        borderpad=0,
    )
    _save(fig, output_dir / "main_results")


def _evaluation_protocol(result, output_dir):
    fig, ax = plt.subplots(figsize=(7.0, 0.95))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    stages = [
        ("1  FIT", "train", "60%", "Learn features and model weights", TEAL, "#EFF7F8"),
        ("2  CHOOSE", "tuning", "20%", "Choose model and threshold", CORAL, "#FBF4F0"),
        (
            "3  EVALUATE",
            "holdout",
            "20%",
            "Freeze choices, then measure",
            SLATE,
            "#F1F4F6",
        ),
    ]
    width = 0.302
    for index, (heading, key, fraction, detail, color, fill) in enumerate(stages):
        x = 0.004 + index * 0.346
        box = FancyBboxPatch(
            (x, 0.08),
            width,
            0.84,
            boxstyle="round,pad=0.004,rounding_size=0.025",
            facecolor=fill,
            edgecolor="none",
        )
        ax.add_patch(box)
        ax.text(
            x + 0.014,
            0.76,
            heading,
            fontsize=8.2,
            color=color,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 0.014,
            0.48,
            f"{result['split_sizes'][key]:,} rows  ·  {fraction}",
            fontsize=10,
            color=SLATE,
            va="center",
        )
        ax.text(x + 0.014, 0.22, detail, fontsize=7.6, color=MUTED, va="center")
        if index < 2:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.006, 0.49),
                    (x + 0.346 - 0.008, 0.49),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color="#A5B2BC",
                    linewidth=1,
                )
            )
    _save(fig, output_dir / "evaluation_protocol")


def generate_figures(metrics_path, output_dir):
    """Create both figures from completed recorded results and return paths."""
    result = json.loads(Path(metrics_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": SLATE,
            "savefig.facecolor": "white",
        }
    ):
        _main_results(result, output_dir)
        _evaluation_protocol(result, output_dir)
    return [output_dir / "main_results.pdf", output_dir / "evaluation_protocol.pdf"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=HERE.parent / "results/metrics.json"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    for output in generate_figures(args.metrics, args.output_dir):
        print(output)
