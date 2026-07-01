#!/usr/bin/env python3
"""GemNet normalized MAE vs Dirichlet roughness normalizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "New_Analysis/manuscript_support/tables/target_variance_dirichlet_gnn_mae_points.csv"
OUT_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE = ROOT / "New_Analysis/manuscript_support/tables/gemnet_nmae_vs_dirichlet_normalizations.csv"
OUT_IQR = OUT_DIR / "gemnet_nmae_vs_iqr_normalized_dirichlet.png"
OUT_VAR = OUT_DIR / "gemnet_nmae_vs_variance_normalized_dirichlet.png"

TARGET_COLORS = {
    r"$E_{\mathrm{ref},\mathrm{BoE}}$": "#2563EB",
    r"$\Delta^{xTB}$": "#4C78A8",
    r"$\Delta^{BoB}$": "#F58518",
    r"$\Delta^{PRE}$": "#54A24B",
    r"$\Delta^{PRE+A}$": "#B279A2",
}

OFFSETS = {
    "dft_ref_boe": (7, -10),
    "deltaxtb": (7, 7),
    "delta_mlp_bob": (7, 7),
    "delta_mlp_pre": (7, 7),
    "delta_mlp_pre_plus_a": (7, -12),
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN)
    df = df[df["model_id"].eq("GemNet")].copy()
    keep = ["dft_ref_boe", "deltaxtb", "delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"]
    df = df[df["target_id"].isin(keep)].copy()
    order = {target_id: idx for idx, target_id in enumerate(keep)}
    df["plot_order"] = df["target_id"].map(order)
    df["gemnet_val_nmae_iqr"] = df["gnn_val_mae"] / df["target_iqr"]
    df = df.sort_values("plot_order").reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, x_col: str, x_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.55), constrained_layout=True)
    r = df[x_col].corr(df["gemnet_val_nmae_iqr"], method="pearson")
    rho = df[x_col].corr(df["gemnet_val_nmae_iqr"], method="spearman")

    for row in df.to_dict(orient="records"):
        color = TARGET_COLORS[row["target_label"]]
        ax.scatter(
            row[x_col],
            row["gemnet_val_nmae_iqr"],
            s=78,
            color=color,
            edgecolor="#202020",
            linewidth=0.85,
            zorder=3,
        )
        dx, dy = OFFSETS.get(row["target_id"], (7, 6))
        ax.annotate(
            row["target_label"],
            (row[x_col], row["gemnet_val_nmae_iqr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            color=color,
            ha="right" if dx < 0 else "left",
            va="center",
        )

    ax.set_xlabel(x_label, fontsize=8.8)
    ax.set_ylabel(r"GemNet validation nMAE, MAE/IQR", fontsize=8.8)
    ax.grid(color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stats_handle = Line2D([], [], linestyle="none", marker=None)
    ax.legend(
        [stats_handle],
        [rf"$r={r:.2f}$, $\rho={rho:.2f}$"],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=8.2,
        handlelength=0,
        handletextpad=0,
        borderpad=0.45,
    )

    x_pad = (df[x_col].max() - df[x_col].min()) * 0.17
    y_pad = (df["gemnet_val_nmae_iqr"].max() - df["gemnet_val_nmae_iqr"].min()) * 0.14
    ax.set_xlim(df[x_col].min() - x_pad, df[x_col].max() + x_pad)
    ax.set_ylim(df["gemnet_val_nmae_iqr"].min() - y_pad, df["gemnet_val_nmae_iqr"].max() + y_pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "target_id",
            "target_label",
            "target_iqr",
            "gnn_val_mae",
            "gemnet_val_nmae_iqr",
            "normalized_dirichlet",
            "iqr_normalized_dirichlet",
        ]
    ].to_csv(TABLE, index=False)

    plot(
        df,
        "iqr_normalized_dirichlet",
        r"IQR-normalized Dirichlet roughness, $D_{\mathrm{IQR}}$",
        OUT_IQR,
    )
    plot(
        df,
        "normalized_dirichlet",
        r"Variance-normalized Dirichlet roughness, $D_{\mathrm{var}}$",
        OUT_VAR,
    )

    print(df[["target_id", "target_iqr", "gnn_val_mae", "gemnet_val_nmae_iqr", "normalized_dirichlet", "iqr_normalized_dirichlet"]].to_string(index=False))
    print(f"WROTE: {TABLE}")
    print(f"WROTE: {OUT_IQR}")
    print(f"WROTE: {OUT_VAR}")


if __name__ == "__main__":
    main()
