#!/usr/bin/env python3
"""Plot variance-normalized Dirichlet roughness vs GemNet MAE/IQR.

This uses PRE raw feature-space Dirichlet roughness for GemNet-aligned
roughness and includes E_ref, E_ref,BoE, Delta^xTB, linear residual targets,
and MLP residual targets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
FIG_DIR = ROOT / "New_Analysis/manuscript_support/figures"

DFT_REF_ROUGHNESS = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_extra_dirichlet_val.csv"
LINEAR_ROUGHNESS = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv"
RESIDUAL_ROUGHNESS = TABLE_DIR / "gemnet_nmae_all_dirichlet_combinations.csv"

OUT_TABLE = TABLE_DIR / "gemnet_iqr_nmae_vs_pre_var_norm_dirichlet_with_eref_linear_mlp.csv"
OUT_FIG = FIG_DIR / "gemnet_iqr_nmae_vs_pre_var_norm_dirichlet_with_eref_linear_mlp.png"

TARGET_ORDER = [
    "dft_ref",
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
    "deltaxtb": "#4C78A8",
    "deltabob": "#F58518",
    "deltapre": "#54A24B",
    "deltapre_plus_a": "#B279A2",
    "delta_mlp_bob": "#F58518",
    "delta_mlp_pre": "#54A24B",
    "delta_mlp_pre_plus_a": "#B279A2",
}

TARGET_GEMNET_MAE = {
    "dft_ref": 0.0136,
    "dft_ref_boe": 0.0101,
    "deltaxtb": 0.0064,
    "deltabob": 0.0108,
    "deltapre": 0.0123,
    "deltapre_plus_a": 0.0165,
    "delta_mlp_bob": 0.0174,
    "delta_mlp_pre": 0.0181,
    "delta_mlp_pre_plus_a": 0.0168,
}

OFFSETS = {
    "dft_ref": (8, 6),
    "dft_ref_boe": (8, -12),
    "deltaxtb": (8, 9),
    "deltabob": (-50, 7),
    "deltapre": (8, -12),
    "deltapre_plus_a": (8, -18),
    "delta_mlp_bob": (8, 18),
    "delta_mlp_pre": (-48, 10),
    "delta_mlp_pre_plus_a": (8, 7),
}

MARKERS = {
    "dft_ref": "o",
    "dft_ref_boe": "o",
    "deltaxtb": "o",
    "deltabob": "s",
    "deltapre": "s",
    "deltapre_plus_a": "s",
    "delta_mlp_bob": "^",
    "delta_mlp_pre": "^",
    "delta_mlp_pre_plus_a": "^",
}


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    dft = pd.read_csv(DFT_REF_ROUGHNESS)
    dft = dft[
        dft["target_id"].eq("dft_ref")
        & dft["feature_space"].eq("PRE")
        & dft["feature_scaling"].eq("raw")
        & dft["split"].eq("val")
    ].iloc[0]
    rows.append(
        {
            "target_id": "dft_ref",
            "target_label": TARGET_LABELS["dft_ref"],
            "target_variance": float(dft["target_std"]) ** 2,
            "target_iqr": float(dft["target_iqr"]),
            "gemnet_val_mae": TARGET_GEMNET_MAE["dft_ref"],
            "variance_normalized_dirichlet": float(dft["edge_avg_norm_by_population_var"]),
        }
    )

    linear = pd.read_csv(LINEAR_ROUGHNESS)
    linear = linear[
        linear["feature_space"].eq("PRE")
        & linear["feature_scaling"].eq("raw")
        & linear["split"].eq("val")
        & linear["target_id"].isin(["dft", "deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"])
    ].copy()
    linear_id_map = {"dft": "dft_ref_boe"}
    for source_target_id in ["dft", "deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"]:
        row = linear[linear["target_id"].eq(source_target_id)].iloc[0]
        target_id = linear_id_map.get(source_target_id, source_target_id)
        rows.append(
            {
                "target_id": target_id,
                "target_label": TARGET_LABELS[target_id],
                "target_variance": float(row["target_std"]) ** 2,
                "target_iqr": float(row["target_iqr"]),
                "gemnet_val_mae": TARGET_GEMNET_MAE[target_id],
                "variance_normalized_dirichlet": float(row["edge_avg_norm_by_population_var"]),
            }
        )

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
                "gemnet_val_mae": TARGET_GEMNET_MAE[target_id],
                "variance_normalized_dirichlet": float(row["variance_normalized_dirichlet"]),
            }
        )

    out = pd.DataFrame(rows)
    out["gemnet_mae_over_iqr"] = out["gemnet_val_mae"] / out["target_iqr"]
    out["plot_order"] = out["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    return out.sort_values("plot_order").reset_index(drop=True)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.25, 3.85), constrained_layout=True)

    pearson = df["variance_normalized_dirichlet"].corr(df["gemnet_mae_over_iqr"], method="pearson")
    spearman = df["variance_normalized_dirichlet"].corr(df["gemnet_mae_over_iqr"], method="spearman")

    for row in df.to_dict(orient="records"):
        target_id = row["target_id"]
        ax.scatter(
            row["variance_normalized_dirichlet"],
            row["gemnet_mae_over_iqr"],
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
            (row["variance_normalized_dirichlet"], row["gemnet_mae_over_iqr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=TARGET_COLORS[target_id],
            ha="left",
            va="center",
        )

    ax.set_xlabel(r"PRE variance-normalized Dirichlet roughness, $D_{\mathrm{var}}$", fontsize=9.2)
    ax.set_ylabel(r"GemNet validation n-MAE, MAE/IQR$(y)$", fontsize=9.2)
    ax.grid(color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_pad = (df["variance_normalized_dirichlet"].max() - df["variance_normalized_dirichlet"].min()) * 0.16
    ax.set_xlim(
        df["variance_normalized_dirichlet"].min() - x_pad,
        df["variance_normalized_dirichlet"].max() + x_pad,
    )

    stats_handle = Line2D([], [], linestyle="none", marker=None)
    ax.legend(
        [stats_handle],
        [rf"$r={pearson:.2f}$, $\rho={spearman:.2f}$"],
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

    y_pad = (df["gemnet_mae_over_iqr"].max() - df["gemnet_mae_over_iqr"].min()) * 0.13
    ax.set_ylim(0.0, df["gemnet_mae_over_iqr"].max() + y_pad)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_rows()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["plot_order"]).to_csv(OUT_TABLE, index=False)
    plot(df, OUT_FIG)
    print(df.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {OUT_TABLE}")
    print(f"WROTE: {OUT_FIG}")


if __name__ == "__main__":
    main()
