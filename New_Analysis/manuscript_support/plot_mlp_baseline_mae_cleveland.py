#!/usr/bin/env python3
"""Cleveland plot for MLP-baseline validation MAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "New_Analysis/manuscript_support/tables/mlp_baseline_dft_ref_boe_val_metrics.csv"
GNN_METRICS = ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv"
OUT = ROOT / "New_Analysis/manuscript_support/figures/mlp_baseline_val_mae_cleveland.png"

COLORS = {
    "xTB": "#4C78A8",
    "BoB": "#F58518",
    "PRE": "#54A24B",
    "PRE+A": "#B279A2",
}


def load_reference_mae(model: str) -> float:
    df = pd.read_csv(GNN_METRICS)
    row = df.loc[(df["model_id"] == model) & (df["target_id"] == "dft")]
    if row.empty:
        raise ValueError(f"Missing {model} DFT_ref,BoE reference MAE in {GNN_METRICS}")
    return float(row.iloc[0]["mae"])


def main() -> None:
    df = pd.read_csv(METRICS).sort_values("mae", ascending=False).reset_index(drop=True)
    gemnet_mae = load_reference_mae("GemNet")
    schnet_mae = load_reference_mae("SchNet")

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    y = range(len(df))

    ax.hlines(y, xmin=0, xmax=df["mae"], color="0.82", linewidth=1.5, zorder=1)
    ax.scatter(
        df["mae"],
        y,
        s=92,
        c=[COLORS[label] for label in df["baseline_label"]],
        edgecolor="#202020",
        linewidth=0.9,
        zorder=3,
    )

    ax.axvline(gemnet_mae, color="#1F2933", linestyle="--", linewidth=1.4, zorder=2)
    ax.axvline(schnet_mae, color="#6B7280", linestyle=":", linewidth=1.4, zorder=2)
    ax.text(
        gemnet_mae + 0.00035,
        len(df) - 0.35,
        r"GemNet $E_{\mathrm{ref},\mathrm{BoE}}$",
        rotation=90,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#1F2933",
    )
    ax.text(
        schnet_mae + 0.00035,
        len(df) - 0.35,
        r"SchNet $E_{\mathrm{ref},\mathrm{BoE}}$",
        rotation=90,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#6B7280",
    )

    for x, yy in zip(df["mae"], y):
        ax.text(x + 0.00065, yy, f"{x:.4f}", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["baseline_label"], fontsize=10.5)
    ax.set_xlabel("Validation MAE (Ha)", fontsize=10.5)
    ax.set_xlim(0, max(df["mae"].max(), schnet_mae) * 1.17)
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(df[["baseline_label", "mae", "residual_iqr", "rmse"]].to_string(index=False))
    print(f"GemNet reference MAE: {gemnet_mae:.4f}")
    print(f"SchNet reference MAE: {schnet_mae:.4f}")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
