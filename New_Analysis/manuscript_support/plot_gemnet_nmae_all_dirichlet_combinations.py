#!/usr/bin/env python3
"""All PRE/PRE+A Dirichlet normalization combinations vs GemNet normalized MAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SMOOTHNESS_DIR = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness"
OUT_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
TABLE = TABLE_DIR / "gemnet_nmae_all_dirichlet_combinations.csv"

FEATURE_SPACES = {
    "pre_plus_a": {
        "label": "PRE+A",
        "base": SMOOTHNESS_DIR / "pre_plus_a_space_dirichlet_val.csv",
        "extra": SMOOTHNESS_DIR / "pre_plus_a_space_extra_dirichlet_val.csv",
        "prefix": "pre_plus_a",
    },
    "pre": {
        "label": "PRE",
        "base": SMOOTHNESS_DIR / "pre_space_dirichlet_val.csv",
        "extra": SMOOTHNESS_DIR / "pre_space_extra_dirichlet_val.csv",
        "prefix": "pre",
    },
}

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

BASE_OFFSETS = {
    "dft_ref_boe": (7, -10),
    "deltaxtb": (7, 7),
    "delta_mlp_bob": (7, 7),
    "delta_mlp_pre": (7, 7),
    "delta_mlp_pre_plus_a": (7, -12),
}

X_METRICS = {
    "iqr_normalized_dirichlet": {
        "label": r"IQR-normalized Dirichlet roughness, $D_{\mathrm{IQR}}$",
        "slug": "iqr_norm_dirichlet",
    },
    "variance_normalized_dirichlet": {
        "label": r"Variance-normalized Dirichlet roughness, $D_{\mathrm{var}}$",
        "slug": "var_norm_dirichlet",
    },
}

Y_METRICS = {
    "gemnet_val_nmae_iqr": {
        "label": r"GemNet validation nMAE, MAE/IQR",
        "slug": "mae_over_iqr",
    },
    "gemnet_val_nmae_var": {
        "label": r"GemNet validation nMAE, MAE/Var",
        "slug": "mae_over_var",
    },
}


def load_feature_space_rows(space_key: str, spec: dict[str, object]) -> pd.DataFrame:
    base = pd.read_csv(spec["base"])
    extra = pd.read_csv(spec["extra"])
    raw = pd.concat([base, extra], ignore_index=True)
    raw = raw[raw["feature_scaling"].eq("raw")].copy()
    raw = raw[raw["target_id"].isin(PLOT_ORDER)].copy()

    rows = []
    for source_target_id in PLOT_ORDER:
        row = raw[raw["target_id"].eq(source_target_id)].iloc[0].to_dict()
        info = TARGET_INFO[source_target_id]
        target_iqr = float(row["target_iqr"])
        target_var = float(row["target_std"]) ** 2
        gemnet_mae = float(info["gemnet_val_mae"])
        rows.append(
            {
                "feature_space_key": space_key,
                "feature_space": spec["label"],
                "feature_scaling": "raw",
                "source_target_id": source_target_id,
                "target_id": info["target_id"],
                "target_label": info["target_label"],
                "target_iqr": target_iqr,
                "target_variance": target_var,
                "gemnet_val_mae": gemnet_mae,
                "gemnet_val_nmae_iqr": gemnet_mae / target_iqr,
                "gemnet_val_nmae_var": gemnet_mae / target_var,
                "variance_normalized_dirichlet": float(row["edge_avg_norm_by_population_var"]),
                "iqr_normalized_dirichlet": float(row["edge_rmse_over_iqr"]) ** 2,
                "edge_rmse": float(row["edge_rmse"]),
                "edge_rmse_over_iqr": float(row["edge_rmse_over_iqr"]),
            }
        )
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    return pd.concat(
        [load_feature_space_rows(space_key, spec) for space_key, spec in FEATURE_SPACES.items()],
        ignore_index=True,
    )


def offsets_for(x_col: str, y_col: str, feature_space_key: str) -> dict[str, tuple[int, int]]:
    offsets = dict(BASE_OFFSETS)
    if x_col == "variance_normalized_dirichlet":
        offsets["delta_mlp_pre"] = (10, 16)
        offsets["delta_mlp_pre_plus_a"] = (8, -20)
    if y_col == "gemnet_val_nmae_var":
        offsets["dft_ref_boe"] = (7, -12)
        offsets["deltaxtb"] = (7, 8)
        offsets["delta_mlp_pre_plus_a"] = (7, -12)
    if feature_space_key == "pre_plus_a" and x_col == "iqr_normalized_dirichlet":
        offsets["delta_mlp_pre_plus_a"] = (7, -12)
    return offsets


def plot_one(df: pd.DataFrame, feature_space_key: str, x_col: str, y_col: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.55), constrained_layout=True)
    sub = df[df["feature_space_key"].eq(feature_space_key)].copy()
    r = sub[x_col].corr(sub[y_col], method="pearson")
    rho = sub[x_col].corr(sub[y_col], method="spearman")
    offsets = offsets_for(x_col, y_col, feature_space_key)
    space_label = FEATURE_SPACES[feature_space_key]["label"]

    for row in sub.to_dict(orient="records"):
        color = TARGET_COLORS[row["target_label"]]
        ax.scatter(
            row[x_col],
            row[y_col],
            s=78,
            color=color,
            edgecolor="#202020",
            linewidth=0.85,
            zorder=3,
        )
        dx, dy = offsets.get(row["target_id"], (7, 6))
        ax.annotate(
            row["target_label"],
            (row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            color=color,
            ha="left",
            va="center",
        )

    ax.set_xlabel(f"{space_label}-space {X_METRICS[x_col]['label']}", fontsize=8.8)
    ax.set_ylabel(Y_METRICS[y_col]["label"], fontsize=8.8)
    ax.grid(color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_handle = Line2D([], [], linestyle="none", marker=None)
    ax.legend(
        [stats_handle],
        [rf"$r={r:.2f}$, $\rho={rho:.2f}$"],
        loc="upper right" if y_col == "gemnet_val_nmae_var" else "upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=8.2,
        handlelength=0,
        handletextpad=0,
        borderpad=0.45,
    )

    x_range = sub[x_col].max() - sub[x_col].min()
    y_range = sub[y_col].max() - sub[y_col].min()
    ax.set_xlim(sub[x_col].min() - x_range * 0.17, sub[x_col].max() + x_range * 0.17)
    ax.set_ylim(sub[y_col].min() - y_range * 0.14, sub[y_col].max() + y_range * 0.14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE, index=False)

    written = []
    for feature_space_key, spec in FEATURE_SPACES.items():
        for y_col, y_spec in Y_METRICS.items():
            for x_col, x_spec in X_METRICS.items():
                out_path = (
                    OUT_DIR
                    / f"gemnet_{y_spec['slug']}_vs_{spec['prefix']}_{x_spec['slug']}.png"
                )
                plot_one(df, feature_space_key, x_col, y_col, out_path)
                written.append(out_path)

    print(
        df[
            [
                "feature_space",
                "target_id",
                "target_iqr",
                "target_variance",
                "gemnet_val_mae",
                "gemnet_val_nmae_iqr",
                "gemnet_val_nmae_var",
                "variance_normalized_dirichlet",
                "iqr_normalized_dirichlet",
            ]
        ].to_string(index=False)
    )
    print(f"WROTE: {TABLE}")
    for path in written:
        print(f"WROTE: {path}")


if __name__ == "__main__":
    main()
