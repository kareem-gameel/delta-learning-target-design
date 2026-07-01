#!/usr/bin/env python3
"""Main-text bipanel plot for QM9 transfer after mean-shift calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
SOURCE_TABLE = TABLE_DIR / "qm9_gnn_all_available_predictions_mean_shift_summary.csv"
OUT_FIG = FIG_DIR / "qm9_transfer_architecture_bipanel.png"
OUT_TABLE = TABLE_DIR / "qm9_transfer_architecture_bipanel.csv"

DIRECT_IDS = {
    "GemNet": "gemnet_DFT_ref_BoE",
    "SchNet": "schnet_DFT_ref_BoE",
}

ROW_DEFS = [
    {
        "target": r"$\Delta^{\mathrm{xTB}}$",
        "class": "xTB",
        "GemNet": "gemnet_Delta_xTB_ref_BoE",
        "SchNet": "schnet_Delta_xTB_ref_BoE",
    },
    {
        "target": r"$\Delta^{\mathrm{BoB}}_{\mathrm{ridge}}$",
        "class": "Ridge",
        "GemNet": "gemnet_Delta_BoB_plus_BoB",
        "SchNet": "schnet_Delta_BoB_ref_BoE",
    },
    {
        "target": r"$\Delta^{\mathrm{PRE}}_{\mathrm{ridge}}$",
        "class": "Ridge",
        "GemNet": "gemnet_Delta_PRE_plus_PRE",
        "SchNet": "schnet_Delta_PRE_ref_BoE",
    },
    {
        "target": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{ridge}}$",
        "class": "Ridge",
        "GemNet": "gemnet_Delta_PRE_plus_A_ref_BoE",
        "SchNet": "schnet_Delta_PRE_plus_A_ref_BoE",
    },
    {
        "target": r"$\Delta^{\mathrm{BoB}}_{\mathrm{MLP}}$",
        "class": "MLP",
        "GemNet": "gemnet_delta_mlp_bob",
        "SchNet": "schnet_delta_mlp_bob",
    },
    {
        "target": r"$\Delta^{\mathrm{PRE}}_{\mathrm{MLP}}$",
        "class": "MLP",
        "GemNet": "gemnet_delta_mlp_pre",
        "SchNet": "schnet_delta_mlp_pre",
    },
    {
        "target": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{MLP}}$",
        "class": "MLP",
        "GemNet": "gemnet_delta_mlp_pre_plus_a",
        "SchNet": "schnet_delta_mlp_pre_plus_a",
    },
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


def load_plot_data() -> tuple[pd.DataFrame, dict[str, float]]:
    source = pd.read_csv(SOURCE_TABLE)
    source = source[source["evaluation_target"].eq("DFT_ref,BoE")].copy()

    direct = {}
    for model, model_id in DIRECT_IDS.items():
        match = source[(source["gnn_model"].eq(model)) & (source["model_id"].eq(model_id))]
        if match.empty:
            raise ValueError(f"Missing direct row for {model}: {model_id}")
        direct[model] = float(match.iloc[0]["mae_Ha"])

    rows = []
    for row_def in ROW_DEFS:
        row = {"target": row_def["target"], "class": row_def["class"]}
        for model in ("GemNet", "SchNet"):
            model_id = row_def[model]
            match = source[(source["gnn_model"].eq(model)) & (source["model_id"].eq(model_id))]
            if match.empty:
                raise ValueError(f"Missing transfer row for {model}: {model_id}")
            row[f"{model.lower()}_mae"] = float(match.iloc[0]["mae_Ha"])
        rows.append(row)

    return pd.DataFrame(rows), direct


def draw_panel(ax, df: pd.DataFrame, direct: dict[str, float], model: str, mae_col: str, show_ylabels: bool) -> None:
    y_positions = list(range(len(df)))[::-1]
    direct_mae = direct[model]

    ax.axvspan(0.0045, direct_mae, color="#DDEFE3", alpha=0.55, zorder=0)
    ax.axvline(direct_mae, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        direct_mae + 0.0010,
        len(df) - 0.58,
        rf"direct {model} = {direct_mae:.4f}",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#1F2933",
    )

    for y, (_, row) in zip(y_positions, df.iterrows(), strict=True):
        value = float(row[mae_col])
        klass = str(row["class"])
        improved = value < direct_mae

        ax.hlines(y, min(value, direct_mae), max(value, direct_mae), color="#C8CED8", linewidth=1.15, zorder=1)
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

        use_left_label = improved or value > 0.061
        dx = -0.0010 if use_left_label else 0.0010
        ha = "right" if use_left_label else "left"
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
    ax.set_xlim(0.0045, 0.0735)
    ax.set_xticks([0.005, 0.015, 0.030, 0.045, 0.060, 0.070])
    ax.set_xlabel(r"QM9 transfer MAE after mean-shift calibration (Ha)", fontsize=8.5)
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
        0.0050,
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

    df, direct = load_plot_data()
    df.to_csv(OUT_TABLE, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), constrained_layout=True)
    draw_panel(axes[0], df, direct, "GemNet", "gemnet_mae", show_ylabels=True)
    draw_panel(axes[1], df, direct, "SchNet", "schnet_mae", show_ylabels=False)

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
    print(f"Direct MAE: {direct}")
    print(f"WROTE: {OUT_TABLE}")
    print(f"WROTE: {OUT_FIG}")


if __name__ == "__main__":
    main()
