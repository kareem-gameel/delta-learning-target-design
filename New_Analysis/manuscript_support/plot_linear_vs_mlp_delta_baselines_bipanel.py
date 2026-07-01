#!/usr/bin/env python3
"""Two-panel paired plot for ridge-vs-MLP descriptor residual controls."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"

MODEL_SPECS = {
    "gemnet": {
        "label": "GemNet",
        "direct_mae": 0.0101,
        "mae_col": "gemnet_mae",
        "out_fig": FIG_DIR / "gemnet_val_mae_linear_vs_mlp_delta_baselines.png",
        "out_table": TABLE_DIR / "gemnet_val_mae_linear_vs_mlp_delta_baselines.csv",
        "xlim": (0.0095, 0.0190),
        "xticks": [0.010, 0.0125, 0.015, 0.0175],
    },
    "schnet": {
        "label": "SchNet",
        "direct_mae": 0.0153,
        "mae_col": "schnet_mae",
        "out_fig": FIG_DIR / "schnet_val_mae_linear_vs_mlp_delta_baselines.png",
        "out_table": TABLE_DIR / "schnet_val_mae_linear_vs_mlp_delta_baselines.csv",
        "xlim": (0.0118, 0.0190),
        "xticks": [0.0125, 0.015, 0.0175],
    },
}

ROWS = [
    {"family": "BoB", "fit": "Linear ridge", "iqr": 0.0411, "variance": 0.00190, "gemnet_mae": 0.0108, "schnet_mae": 0.0142},
    {"family": "BoB", "fit": "MLP", "iqr": 0.0260, "variance": 0.00144, "gemnet_mae": 0.0174, "schnet_mae": 0.0179},
    {"family": "PRE", "fit": "Linear ridge", "iqr": 0.0232, "variance": 0.00064, "gemnet_mae": 0.0123, "schnet_mae": 0.0125},
    {"family": "PRE", "fit": "MLP", "iqr": 0.0201, "variance": 0.00108, "gemnet_mae": 0.0181, "schnet_mae": 0.0182},
    {"family": "PRE+A", "fit": "Linear ridge", "iqr": 0.0239, "variance": 0.00087, "gemnet_mae": 0.0165, "schnet_mae": 0.0175},
    {"family": "PRE+A", "fit": "MLP", "iqr": 0.0175, "variance": 0.00084, "gemnet_mae": 0.0168, "schnet_mae": 0.0167},
]

FAMILY_ORDER = ["BoB", "PRE", "PRE+A"]
COLORS = {"Linear ridge": "#4C78A8", "MLP": "#B96F4F"}
MARKERS = {"Linear ridge": "o", "MLP": "D"}


def paired_values(df: pd.DataFrame, family: str, metric: str) -> tuple[float, float]:
    sub = df[df["family"].eq(family)].set_index("fit")
    return float(sub.loc["Linear ridge", metric]), float(sub.loc["MLP", metric])


def draw_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    xlabel: str,
    xlim: tuple[float, float],
    xticks: list[float],
    show_ylabels: bool,
    value_dx: float = 0.00025,
) -> None:
    y_positions = list(range(len(FAMILY_ORDER)))[::-1]
    for y, family in zip(y_positions, FAMILY_ORDER):
        ridge, mlp = paired_values(df, family, metric)
        ax.plot([ridge, mlp], [y, y], color="#9CA3AF", linewidth=1.35, zorder=1)
        ax.scatter(
            ridge,
            y,
            s=58,
            color=COLORS["Linear ridge"],
            marker=MARKERS["Linear ridge"],
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            mlp,
            y,
            s=58,
            color=COLORS["MLP"],
            marker=MARKERS["MLP"],
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )

        if metric == "iqr":
            ridge_label = f"{ridge:.4f}"
            mlp_label = f"{mlp:.4f}"
            ax.text(ridge + 0.0010, y + 0.10, ridge_label, ha="left", va="bottom", fontsize=7.3, color=COLORS["Linear ridge"])
            ax.text(mlp - 0.0010, y - 0.12, mlp_label, ha="right", va="top", fontsize=7.3, color=COLORS["MLP"])
        else:
            ridge_label = f"{ridge:.4f}"
            mlp_label = f"{mlp:.4f}"
            ax.text(ridge - value_dx, y - 0.12, ridge_label, ha="right", va="top", fontsize=7.3, color=COLORS["Linear ridge"])
            ax.text(mlp + value_dx, y + 0.10, mlp_label, ha="left", va="bottom", fontsize=7.3, color=COLORS["MLP"])

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_ylim(-0.55, len(FAMILY_ORDER) - 0.45)
    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.set_yticklabels([rf"$\Delta^{{\mathrm{{{family}}}}}$" for family in FAMILY_ORDER], fontsize=8.4)
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_model(df: pd.DataFrame, spec: dict[str, object]) -> None:
    out_table = spec["out_table"]
    out_fig = spec["out_fig"]
    model_label = str(spec["label"])
    mae_col = str(spec["mae_col"])
    direct_mae = float(spec["direct_mae"])

    df.to_csv(out_table, index=False)

    df = pd.DataFrame(ROWS)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.9, 2.95),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )

    draw_panel(
        axes[0],
        df,
        "iqr",
        "Residual target IQR (Ha)",
        (0.015, 0.044),
        [0.02, 0.03, 0.04],
        show_ylabels=True,
    )
    axes[0].set_title("Residual scale", fontsize=9.5, fontweight="bold", pad=6)
    axes[0].text(
        0.016,
        -0.44,
        "smaller residual",
        ha="left",
        va="center",
        fontsize=7.0,
        color="#6B7280",
    )

    draw_panel(
        axes[1],
        df,
        mae_col,
        f"{model_label} validation MAE (Ha)",
        spec["xlim"],
        spec["xticks"],
        show_ylabels=False,
        value_dx=0.00022,
    )
    axes[1].set_title("Residual learning", fontsize=9.5, fontweight="bold", pad=6)
    axes[1].axvline(direct_mae, color="#1F2933", linestyle="--", linewidth=1.0, zorder=0)
    axes[1].text(
        direct_mae + 0.00012,
        len(FAMILY_ORDER) - 0.62,
        rf"direct {model_label} = {direct_mae:.4f}",
        ha="left",
        va="top",
        fontsize=7.1,
        color="#1F2933",
    )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS["Linear ridge"],
            markerfacecolor=COLORS["Linear ridge"],
            markeredgecolor="#202020",
            markeredgewidth=0.7,
            markersize=5.8,
            label="Linear ridge baseline",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS["MLP"],
            markerfacecolor=COLORS["MLP"],
            markeredgecolor="#202020",
            markeredgewidth=0.7,
            markersize=5.8,
            label="MLP baseline",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.1, bbox_to_anchor=(0.55, 1.07))

    fig.savefig(out_fig, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{model_label}")
    print(df[["family", "fit", "iqr", "variance", mae_col]].to_string(index=False))
    print(f"WROTE: {out_table}")
    print(f"WROTE: {out_fig}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS)
    for spec in MODEL_SPECS.values():
        plot_model(df, spec)


if __name__ == "__main__":
    main()
