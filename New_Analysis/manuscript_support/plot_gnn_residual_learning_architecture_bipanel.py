#!/usr/bin/env python3
"""Main-text bipanel plot for architecture-dependent residual learning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
OUT_FIG = FIG_DIR / "gnn_residual_learning_architecture_bipanel.png"
OUT_TABLE = TABLE_DIR / "gnn_residual_learning_architecture_bipanel.csv"

DIRECT = {
    "GemNet": 0.0101,
    "SchNet": 0.0153,
}

ROWS = [
    {"target": r"$\Delta^{\mathrm{xTB}}$", "class": "xTB", "gemnet_mae": 0.0064, "schnet_mae": 0.0093},
    {"target": r"$\Delta^{\mathrm{BoB}}_{\mathrm{ridge}}$", "class": "Ridge", "gemnet_mae": 0.0108, "schnet_mae": 0.0142},
    {"target": r"$\Delta^{\mathrm{PRE}}_{\mathrm{ridge}}$", "class": "Ridge", "gemnet_mae": 0.0123, "schnet_mae": 0.0125},
    {"target": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{ridge}}$", "class": "Ridge", "gemnet_mae": 0.0165, "schnet_mae": 0.0175},
    {"target": r"$\Delta^{\mathrm{BoB}}_{\mathrm{MLP}}$", "class": "MLP", "gemnet_mae": 0.0174, "schnet_mae": 0.0179},
    {"target": r"$\Delta^{\mathrm{PRE}}_{\mathrm{MLP}}$", "class": "MLP", "gemnet_mae": 0.0181, "schnet_mae": 0.0182},
    {"target": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{MLP}}$", "class": "MLP", "gemnet_mae": 0.0168, "schnet_mae": 0.0167},
]

COLORS = {
    "xTB": "#3F6FA6",
    "Ridge": "#4C9A45",
    "MLP": "#B46A4C",
}

MARKERS = {
    "xTB": "D",
    "Ridge": "o",
    "MLP": "s",
}


def draw_panel(ax, df: pd.DataFrame, model: str, mae_col: str, show_ylabels: bool) -> None:
    y_positions = list(range(len(df)))[::-1]
    direct = DIRECT[model]

    ax.axvspan(0.0055, direct, color="#DDEFE3", alpha=0.55, zorder=0)
    ax.axvline(direct, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        direct + 0.00016,
        len(df) - 0.58,
        rf"direct {model} = {direct:.4f}",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#1F2933",
    )

    for y, (_, row) in zip(y_positions, df.iterrows(), strict=True):
        value = float(row[mae_col])
        klass = str(row["class"])
        improved = value < direct

        ax.hlines(y, min(value, direct), max(value, direct), color="#C8CED8", linewidth=1.15, zorder=1)
        ax.scatter(
            value,
            y,
            s=56,
            marker=MARKERS[klass],
            color=COLORS[klass],
            edgecolor="#202020",
            linewidth=0.65,
            zorder=3,
        )

        dx = -0.00028 if improved else 0.00028
        ha = "right" if improved else "left"
        ax.text(
            value + dx,
            y + 0.12,
            f"{value:.4f}",
            ha=ha,
            va="bottom",
            fontsize=7.0,
            color=COLORS[klass],
        )

    ax.set_title(model, fontsize=10.0, fontweight="bold", pad=7)
    ax.set_xlim(0.0055, 0.0190)
    ax.set_xticks([0.006, 0.009, 0.012, 0.015, 0.018])
    ax.set_xlabel(r"Reconstructed validation MAE on $E_{\mathrm{ref},\mathrm{BoE}}$ (Ha)", fontsize=8.5)
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.set_yticklabels(df["target"].tolist(), fontsize=8.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(
        0.00575,
        -0.40,
        "better than direct",
        ha="left",
        va="center",
        fontsize=6.9,
        color="#4B6F55",
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(ROWS)
    df.to_csv(OUT_TABLE, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), constrained_layout=True)
    draw_panel(axes[0], df, "GemNet", "gemnet_mae", show_ylabels=True)
    draw_panel(axes[1], df, "SchNet", "schnet_mae", show_ylabels=False)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS["xTB"],
            markerfacecolor=COLORS["xTB"],
            markeredgecolor="#202020",
            markeredgewidth=0.65,
            markersize=5.8,
            label="xTB baseline",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS["Ridge"],
            markerfacecolor=COLORS["Ridge"],
            markeredgecolor="#202020",
            markeredgewidth=0.65,
            markersize=5.8,
            label="linear ridge descriptor",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS["MLP"],
            markerfacecolor=COLORS["MLP"],
            markeredgecolor="#202020",
            markeredgewidth=0.65,
            markersize=5.8,
            label="MLP descriptor",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.0, bbox_to_anchor=(0.57, 1.07))

    fig.savefig(OUT_FIG, dpi=450, bbox_inches="tight")
    print(df.to_string(index=False))
    print(f"WROTE: {OUT_TABLE}")
    print(f"WROTE: {OUT_FIG}")


if __name__ == "__main__":
    main()
