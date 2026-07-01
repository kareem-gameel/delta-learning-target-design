#!/usr/bin/env python3
"""Plot PRE IQR-normalized Dirichlet roughness vs GNN MAE/IQR.

Generates paired GemNet and SchNet figures for the same residual target panel.
The x-axis is IQR-normalized graph Dirichlet roughness,
edge_mean_squared_diff / IQR(y)^2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
FIG_DIR = ROOT / "New_Analysis/manuscript_support/figures"

DFT_REF_ROUGHNESS = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_extra_dirichlet_val.csv"
LINEAR_ROUGHNESS = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv"
RESIDUAL_ROUGHNESS = TABLE_DIR / "gemnet_nmae_all_dirichlet_combinations.csv"

GEMNET_OUT_TABLE = TABLE_DIR / "gemnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_linear_mlp.csv"
GEMNET_OUT_FIG = FIG_DIR / "gemnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_linear_mlp.png"
SCHNET_OUT_TABLE = TABLE_DIR / "schnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_linear_mlp.csv"
SCHNET_OUT_FIG = FIG_DIR / "schnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_linear_mlp.png"

GEMNET_TARGET_ORDER = [
    "dft_ref_boe",
    "deltaxtb",
    "deltabob",
    "deltapre",
    "deltapre_plus_a",
    "delta_mlp_bob",
    "delta_mlp_pre",
    "delta_mlp_pre_plus_a",
]

SCHNET_TARGET_ORDER = [
    "dft_ref_boe",
    "deltaxtb",
    "deltabob",
    "deltapre",
    "deltapre_plus_a",
    "delta_mlp_bob",
    "delta_mlp_pre",
    "delta_mlp_pre_plus_a",
]

TARGET_LABELS = {
    "dft_ref": r"$E_{\mathrm{ref}}$",
    "dft_ref_boe": r"$E_{\mathrm{ref},\mathrm{BoE}}$",
    "xtb_ref_boe": r"$E_{\mathrm{xTB},\mathrm{ref},\mathrm{BoE}}$",
    "deltaxtb": r"$\Delta^{\mathrm{xTB}}$",
    "deltabob": r"$\Delta^{\mathrm{BoB}}_{\mathrm{lin}}$",
    "deltapre": r"$\Delta^{\mathrm{PRE}}_{\mathrm{lin}}$",
    "deltapre_plus_a": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{lin}}$",
    "delta_mlp_bob": r"$\Delta^{\mathrm{BoB}}_{\mathrm{MLP}}$",
    "delta_mlp_pre": r"$\Delta^{\mathrm{PRE}}_{\mathrm{MLP}}$",
    "delta_mlp_pre_plus_a": r"$\Delta^{\mathrm{PRE+A}}_{\mathrm{MLP}}$",
}

TARGET_COLORS = {
    "dft_ref": "#374151",
    "dft_ref_boe": "#2563EB",
    "xtb_ref_boe": "#0891B2",
    "deltaxtb": "#4C78A8",
    "deltabob": "#F58518",
    "deltapre": "#54A24B",
    "deltapre_plus_a": "#B279A2",
    "delta_mlp_bob": "#F58518",
    "delta_mlp_pre": "#54A24B",
    "delta_mlp_pre_plus_a": "#B279A2",
}

MARKERS = {
    "dft_ref": "o",
    "dft_ref_boe": "o",
    "xtb_ref_boe": "o",
    "deltaxtb": "D",
    "deltabob": "s",
    "deltapre": "s",
    "deltapre_plus_a": "s",
    "delta_mlp_bob": "^",
    "delta_mlp_pre": "^",
    "delta_mlp_pre_plus_a": "^",
}

GEMNET_MAE = {
    "dft_ref": 0.0136,
    "dft_ref_boe": 0.0101,
    "xtb_ref_boe": 0.0086,
    "deltaxtb": 0.0064,
    "deltabob": 0.0108,
    "deltapre": 0.0123,
    "deltapre_plus_a": 0.0165,
    "delta_mlp_bob": 0.0174,
    "delta_mlp_pre": 0.0181,
    "delta_mlp_pre_plus_a": 0.0168,
}

SCHNET_MAE = {
    "dft_ref_boe": 0.0153,
    "deltaxtb": 0.0093,
    "deltabob": 0.0142,
    "deltapre": 0.0125,
    "deltapre_plus_a": 0.0175,
    "delta_mlp_bob": 0.0179,
    "delta_mlp_pre": 0.0182,
    "delta_mlp_pre_plus_a": 0.0167,
}

OFFSETS = {
    "dft_ref": (5, 6),
    "dft_ref_boe": (5, -9),
    "xtb_ref_boe": (5, -10),
    "deltaxtb": (5, 8),
    "deltabob": (-8, 8),
    "deltapre": (-8, -8),
    "deltapre_plus_a": (5, -8),
    "delta_mlp_bob": (5, 8),
    "delta_mlp_pre": (5, 7),
    "delta_mlp_pre_plus_a": (5, 5),
}


def load_pre_rows(mae_map: dict[str, float], include_dft_ref: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if include_dft_ref:
        dft = pd.read_csv(DFT_REF_ROUGHNESS)
        dft = dft[
            dft["target_id"].eq("dft_ref")
            & dft["feature_space"].eq("PRE")
            & dft["feature_scaling"].eq("raw")
            & dft["split"].eq("val")
        ].iloc[0]
        rows.append(row_from_smoothness("dft_ref", dft, mae_map["dft_ref"]))

    linear = pd.read_csv(LINEAR_ROUGHNESS)
    linear = linear[
        linear["feature_space"].eq("PRE")
        & linear["feature_scaling"].eq("raw")
        & linear["split"].eq("val")
        & linear["target_id"].isin(["dft", "deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"])
    ].copy()

    linear_id_map = {"dft": "dft_ref_boe"}
    for source_target_id in ["dft", "deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"]:
        target_id = linear_id_map.get(source_target_id, source_target_id)
        row = linear[linear["target_id"].eq(source_target_id)].iloc[0]
        rows.append(row_from_smoothness(target_id, row, mae_map[target_id]))

    residuals = pd.read_csv(RESIDUAL_ROUGHNESS)
    residuals = residuals[
        residuals["feature_space_key"].eq("pre")
        & residuals["feature_scaling"].eq("raw")
        & residuals["target_id"].isin(["delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"])
    ].copy()
    for target_id in ["delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"]:
        row = residuals[residuals["target_id"].eq(target_id)].iloc[0]
        rows.append(
            {
                "target_id": target_id,
                "target_label": TARGET_LABELS[target_id],
                "target_variance": float(row["target_variance"]),
                "target_iqr": float(row["target_iqr"]),
                "gnn_val_mae": mae_map[target_id],
                "iqr_normalized_dirichlet": float(row["iqr_normalized_dirichlet"]),
            }
        )

    out = pd.DataFrame(rows)
    out["gnn_mae_over_iqr"] = out["gnn_val_mae"] / out["target_iqr"]
    return out


def row_from_smoothness(target_id: str, row: pd.Series, gnn_mae: float) -> dict[str, object]:
    edge_rmse_over_iqr = float(row["edge_rmse_over_iqr"])
    return {
        "target_id": target_id,
        "target_label": TARGET_LABELS[target_id],
        "target_variance": float(row["target_std"]) ** 2,
        "target_iqr": float(row["target_iqr"]),
        "gnn_val_mae": float(gnn_mae),
        "iqr_normalized_dirichlet": edge_rmse_over_iqr**2,
    }


def plot_panel(df: pd.DataFrame, order: list[str], model_label: str, out_path: Path) -> pd.DataFrame:
    order_map = {target_id: i for i, target_id in enumerate(order)}
    df = df[df["target_id"].isin(order)].copy()
    df["plot_order"] = df["target_id"].map(order_map)
    df = df.sort_values("plot_order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.35, 3.9), constrained_layout=True)

    log_x = np.log10(df["iqr_normalized_dirichlet"].to_numpy(float))
    y = df["gnn_mae_over_iqr"].to_numpy(float)
    pearson_log = float(pd.Series(log_x).corr(pd.Series(y), method="pearson"))
    spearman = float(df["iqr_normalized_dirichlet"].corr(df["gnn_mae_over_iqr"], method="spearman"))

    for row in df.to_dict(orient="records"):
        target_id = row["target_id"]
        ax.scatter(
            row["iqr_normalized_dirichlet"],
            row["gnn_mae_over_iqr"],
            s=84,
            marker=MARKERS[target_id],
            color=TARGET_COLORS[target_id],
            edgecolor="#202020",
            linewidth=0.85,
            zorder=3,
        )
        dx, dy = OFFSETS[target_id]
        ax.annotate(
            row["target_label"],
            (row["iqr_normalized_dirichlet"], row["gnn_mae_over_iqr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=TARGET_COLORS[target_id],
            ha="right" if dx < 0 else "left",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"PRE IQR-normalized Dirichlet roughness, $D_{\mathrm{IQR}}$", fontsize=9.2)
    ax.set_ylabel(rf"{model_label} validation n-MAE, MAE/IQR$(y)$", fontsize=9.2)
    ax.grid(color="#E5E7EB", linewidth=0.75, which="both")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_min = df["iqr_normalized_dirichlet"].min()
    x_max = df["iqr_normalized_dirichlet"].max()
    y_pad = (df["gnn_mae_over_iqr"].max() - df["gnn_mae_over_iqr"].min()) * 0.13
    ax.set_xlim(x_min / 1.35, x_max * 1.35)
    ax.set_ylim(0.0, df["gnn_mae_over_iqr"].max() + y_pad)

    stats_handle = Line2D([], [], linestyle="none", marker=None)
    stats_legend = ax.legend(
        [stats_handle],
        [rf"$r_{{\log x}}={pearson_log:.2f}$, $\rho={spearman:.2f}$"],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=8.2,
        handlelength=0,
        handletextpad=0,
        borderpad=0.45,
    )
    ax.add_artist(stats_legend)

    target_ids = set(df["target_id"])
    shape_groups = [
        ({"dft_ref", "dft_ref_boe", "xtb_ref_boe"}, "full/baseline targets", "o"),
        ({"deltaxtb"}, r"$\Delta^{\mathrm{xTB}}$ residual", "D"),
        ({"deltabob", "deltapre", "deltapre_plus_a"}, "linear residuals", "s"),
        ({"delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"}, "MLP residuals", "^"),
    ]
    shape_handles = []
    shape_labels = []
    for ids, label, marker in shape_groups:
        if target_ids.intersection(ids):
            shape_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor="#F3F4F6",
                    markeredgecolor="#202020",
                    markeredgewidth=0.85,
                    markersize=6.2,
                )
            )
            shape_labels.append(label)
    ax.legend(
        shape_handles,
        shape_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=7.4,
        handlelength=1.0,
        handletextpad=0.45,
        borderpad=0.45,
        labelspacing=0.35,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    gemnet = load_pre_rows(GEMNET_MAE, include_dft_ref=False)
    gemnet = plot_panel(gemnet, GEMNET_TARGET_ORDER, "GemNet", GEMNET_OUT_FIG)
    gemnet.drop(columns=["plot_order"]).to_csv(GEMNET_OUT_TABLE, index=False)

    schnet = load_pre_rows(SCHNET_MAE, include_dft_ref=False)
    schnet = plot_panel(schnet, SCHNET_TARGET_ORDER, "SchNet", SCHNET_OUT_FIG)
    schnet.drop(columns=["plot_order"]).to_csv(SCHNET_OUT_TABLE, index=False)

    print("GemNet")
    print(gemnet.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {GEMNET_OUT_TABLE}")
    print(f"WROTE: {GEMNET_OUT_FIG}")
    print("\nSchNet")
    print(schnet.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {SCHNET_OUT_TABLE}")
    print(f"WROTE: {SCHNET_OUT_FIG}")


if __name__ == "__main__":
    main()
