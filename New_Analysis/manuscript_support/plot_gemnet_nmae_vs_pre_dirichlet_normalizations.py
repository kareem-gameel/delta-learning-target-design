#!/usr/bin/env python3
"""GemNet normalized MAE vs PRE-space Dirichlet roughness normalizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
BASE_PRE = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv"
EXTRA_PRE = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_extra_dirichlet_val.csv"
OUT_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE = ROOT / "New_Analysis/manuscript_support/tables/gemnet_nmae_vs_pre_dirichlet_normalizations.csv"
OUT_IQR = OUT_DIR / "gemnet_nmae_vs_pre_iqr_normalized_dirichlet.png"
OUT_VAR = OUT_DIR / "gemnet_nmae_vs_pre_variance_normalized_dirichlet.png"

TARGET_INFO = {
    "dft": {
        "target_id": "dft_ref_boe",
        "target_label": r"$E_{\mathrm{ref},\mathrm{BoE}}$",
        "gemnet_val_mae": 0.0101,
    },
    "deltaxtb": {
        "target_id": "deltaxtb",
        "target_label": r"$\Delta^{xTB}$",
        "gemnet_val_mae": 0.0064,
    },
    "delta_mlp_bob": {
        "target_id": "delta_mlp_bob",
        "target_label": r"$\Delta^{BoB}$",
        "gemnet_val_mae": 0.0174,
    },
    "delta_mlp_pre": {
        "target_id": "delta_mlp_pre",
        "target_label": r"$\Delta^{PRE}$",
        "gemnet_val_mae": 0.0181,
    },
    "delta_mlp_pre_plus_a": {
        "target_id": "delta_mlp_pre_plus_a",
        "target_label": r"$\Delta^{PRE+A}$",
        "gemnet_val_mae": 0.0168,
    },
}

PLOT_ORDER = ["dft", "deltaxtb", "delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"]

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
    base = pd.read_csv(BASE_PRE)
    extra = pd.read_csv(EXTRA_PRE)
    raw = pd.concat([base, extra], ignore_index=True)
    raw = raw[raw["feature_scaling"].eq("raw")].copy()
    raw = raw[raw["target_id"].isin(PLOT_ORDER)].copy()

    rows = []
    for raw_target_id in PLOT_ORDER:
        row = raw[raw["target_id"].eq(raw_target_id)].iloc[0].to_dict()
        info = TARGET_INFO[raw_target_id]
        target_iqr = float(row["target_iqr"])
        gemnet_mae = float(info["gemnet_val_mae"])
        rows.append(
            {
                "source_target_id": raw_target_id,
                "target_id": info["target_id"],
                "target_label": info["target_label"],
                "feature_space": "PRE",
                "feature_scaling": "raw",
                "target_iqr": target_iqr,
                "gemnet_val_mae": gemnet_mae,
                "gemnet_val_nmae_iqr": gemnet_mae / target_iqr,
                "normalized_dirichlet": float(row["edge_avg_norm_by_population_var"]),
                "iqr_normalized_dirichlet": float(row["edge_rmse_over_iqr"]) ** 2,
                "edge_rmse": float(row["edge_rmse"]),
                "edge_rmse_over_iqr": float(row["edge_rmse_over_iqr"]),
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, x_col: str, x_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.55), constrained_layout=True)
    r = df[x_col].corr(df["gemnet_val_nmae_iqr"], method="pearson")
    rho = df[x_col].corr(df["gemnet_val_nmae_iqr"], method="spearman")
    offsets = dict(OFFSETS)
    if x_col == "normalized_dirichlet":
        offsets["delta_mlp_pre"] = (10, 16)
        offsets["delta_mlp_pre_plus_a"] = (8, -20)

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
        dx, dy = offsets.get(row["target_id"], (7, 6))
        ax.annotate(
            row["target_label"],
            (row[x_col], row["gemnet_val_nmae_iqr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            color=color,
            ha="left",
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
    df.to_csv(TABLE, index=False)

    plot(
        df,
        "iqr_normalized_dirichlet",
        r"PRE-space IQR-normalized Dirichlet roughness, $D_{\mathrm{IQR}}$",
        OUT_IQR,
    )
    plot(
        df,
        "normalized_dirichlet",
        r"PRE-space variance-normalized Dirichlet roughness, $D_{\mathrm{var}}$",
        OUT_VAR,
    )

    print(
        df[
            [
                "target_id",
                "target_iqr",
                "gemnet_val_mae",
                "gemnet_val_nmae_iqr",
                "normalized_dirichlet",
                "iqr_normalized_dirichlet",
            ]
        ].to_string(index=False)
    )
    print(f"WROTE: {TABLE}")
    print(f"WROTE: {OUT_IQR}")
    print(f"WROTE: {OUT_VAR}")


if __name__ == "__main__":
    main()
