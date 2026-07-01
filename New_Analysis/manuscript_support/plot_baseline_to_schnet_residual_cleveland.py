#!/usr/bin/env python3
"""Paired Cleveland plot: baseline MAE before/after SchNet residual learning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINE_METRICS = ROOT / "New_Analysis/manuscript_support/tables/mlp_baseline_dft_ref_boe_val_metrics.csv"
GNN_STATUS = ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv"
MLP_GNN = ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary_all.csv"
OUT = ROOT / "New_Analysis/manuscript_support/figures/baseline_to_schnet_residual_mae_cleveland.png"
OUT_TABLE = ROOT / "New_Analysis/manuscript_support/tables/baseline_to_schnet_residual_mae.csv"

COLORS = {
    "xTB": "#4C78A8",
    "BoB": "#F58518",
    "PRE": "#54A24B",
    "PRE+A": "#B279A2",
}

SCHNET_TARGET_IDS = {
    "xTB": "deltaxtb",
    "BoB": "delta_mlp_bob",
    "PRE": "delta_mlp_pre",
    "PRE+A": "delta_mlp_pre_plus_a",
}


def load_direct_schnet_mae() -> float:
    df = pd.read_csv(GNN_STATUS)
    row = df.loc[(df["model_id"] == "SchNet") & (df["target_id"] == "dft")]
    if row.empty:
        raise ValueError(f"Missing direct SchNet E_ref,BoE MAE in {GNN_STATUS}")
    return float(row.iloc[0]["mae"])


def load_schnet_residual_mae(label: str) -> float:
    target_id = SCHNET_TARGET_IDS[label]
    if target_id == "deltaxtb":
        df = pd.read_csv(GNN_STATUS)
        row = df.loc[(df["model_id"] == "SchNet") & (df["target_id"] == target_id)]
        if row.empty:
            raise ValueError(f"Missing SchNet {target_id} MAE in {GNN_STATUS}")
        return float(row.iloc[0]["mae"])

    df = pd.read_csv(MLP_GNN)
    row = df.loc[(df["model_id"] == "SchNet") & (df["target_id"] == target_id)]
    if row.empty:
        raise ValueError(f"Missing SchNet {target_id} MAE in {MLP_GNN}")
    return float(row.iloc[0]["best_val_mae"])


def main() -> None:
    baseline = pd.read_csv(BASELINE_METRICS)
    rows = []
    for _, row in baseline.iterrows():
        label = row["baseline_label"]
        residual_mae = load_schnet_residual_mae(label)
        rows.append(
            {
                "baseline": label,
                "baseline_val_mae": float(row["mae"]),
                "schnet_residual_val_mae": residual_mae,
                "delta_improvement": float(row["mae"]) - residual_mae,
            }
        )
    df = pd.DataFrame(rows).sort_values("baseline_val_mae", ascending=False).reset_index(drop=True)
    direct_mae = load_direct_schnet_mae()

    fig, ax = plt.subplots(figsize=(5.65, 2.85), constrained_layout=True)
    y = list(range(len(df)))

    for yy, row in zip(y, df.to_dict(orient="records")):
        color = COLORS[row["baseline"]]
        ax.plot(
            [row["baseline_val_mae"], row["schnet_residual_val_mae"]],
            [yy, yy],
            color=color,
            linewidth=1.45,
            alpha=0.68,
            zorder=2,
        )
        ax.scatter(
            row["baseline_val_mae"],
            yy,
            s=36,
            color=color,
            marker="o",
            edgecolor="#202020",
            linewidth=0.65,
            zorder=3,
            label="Baseline" if yy == 0 else None,
        )
        ax.scatter(
            row["schnet_residual_val_mae"],
            yy,
            s=42,
            color="white",
            marker="D",
            edgecolor=color,
            linewidth=1.15,
            zorder=4,
            label="SchNet on residual" if yy == 0 else None,
        )
        ax.text(
            row["baseline_val_mae"] + 0.00055,
            yy + 0.070,
            f"{row['baseline_val_mae']:.4f}",
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#202020",
        )
        ax.text(
            row["schnet_residual_val_mae"] + 0.00055,
            yy - 0.070,
            f"{row['schnet_residual_val_mae']:.4f}",
            ha="left",
            va="top",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )

    ax.axvline(direct_mae, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        direct_mae + 0.00035,
        len(df) - 0.40,
        r"SchNet $E_{\mathrm{ref},\mathrm{BoE}}$",
        rotation=90,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#1F2933",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["baseline"], fontsize=7.8)
    ax.set_xlabel("Validation MAE (Ha)", fontsize=8.2)
    ax.set_xlim(0, df["baseline_val_mae"].max() * 1.18)
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.grid(axis="x", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handletextpad=0.40)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE, index=False)
    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"Direct SchNet E_ref,BoE MAE: {direct_mae:.4f}")
    print(f"WROTE: {OUT_TABLE}")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
