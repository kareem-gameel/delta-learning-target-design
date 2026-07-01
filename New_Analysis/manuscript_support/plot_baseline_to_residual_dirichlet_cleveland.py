#!/usr/bin/env python3
"""Paired Cleveland plot: PRE+A std-cosine D_IQR before/after residualization."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from run_expanded_smoothness import scale_features  # noqa: E402

from plot_gemnet_iqr_nmae_vs_pre_plus_a_std_iqr_dirichlet_cosine import (  # noqa: E402
    EDGE_CACHE,
    align_pre_plus_a_features,
)
from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet import GEMNET_MAE  # noqa: E402
from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet_cosine import (  # noqa: E402
    build_cosine_edges,
    load_target_panel,
    smoothness_rows,
)

OUT = ROOT / "New_Analysis/manuscript_support/figures/baseline_to_residual_normalized_dirichlet_cleveland.png"
OUT_TABLE = ROOT / "New_Analysis/manuscript_support/tables/baseline_to_residual_normalized_dirichlet.csv"

COLORS = {
    "xTB": "#4C78A8",
    "BoB": "#F58518",
    "PRE": "#54A24B",
    "PRE+A": "#B279A2",
}

RESIDUAL_TARGET_IDS = {
    "xTB": "deltaxtb",
    "MLP-BoB": "delta_mlp_bob",
    "MLP-PRE": "delta_mlp_pre",
    "MLP-PRE+A": "delta_mlp_pre_plus_a",
}

COLORS.update(
    {
        "MLP-BoB": COLORS["BoB"],
        "MLP-PRE": COLORS["PRE"],
        "MLP-PRE+A": COLORS["PRE+A"],
    }
)


def compute_roughness_panel() -> pd.DataFrame:
    frame = load_target_panel()
    x_raw, aligned, feature_meta = align_pre_plus_a_features(frame)
    x_std, n_retained, n_dropped = scale_features(x_raw, "std", 1e-12)
    edges = build_cosine_edges(x_std, k=10, edge_cache=EDGE_CACHE)
    rows = smoothness_rows(aligned, edges, GEMNET_MAE)

    keep = ["dft_ref_boe", *RESIDUAL_TARGET_IDS.values()]
    out = rows[rows["target_id"].isin(keep)].copy()
    out["feature_space"] = "PRE+A"
    out["feature_scaling"] = "std"
    out["neighbor_metric"] = "cosine"
    out["k"] = 10
    out["normalization"] = "IQR"
    out["n_features_total"] = int(feature_meta["n_features"])
    out["n_features_retained"] = int(n_retained)
    out["n_features_dropped"] = int(n_dropped)
    return out


def main() -> None:
    roughness = compute_roughness_panel()
    reference = roughness[roughness["target_id"].eq("dft_ref_boe")].iloc[0]
    reference_roughness = float(reference["iqr_normalized_dirichlet"])

    rows = []
    for label, target_id in RESIDUAL_TARGET_IDS.items():
        residual = roughness[roughness["target_id"].eq(target_id)].iloc[0]
        residual_roughness = float(residual["iqr_normalized_dirichlet"])
        rows.append(
            {
                "baseline": label,
                "feature_space": "PRE+A",
                "feature_scaling": "std",
                "neighbor_metric": "cosine",
                "k": 10,
                "normalization": "IQR",
                "reference_target": "DFT_ref,BoE",
                "residual_target_id": target_id,
                "reference_d_iqr": reference_roughness,
                "residual_d_iqr": residual_roughness,
                "delta_d_iqr": residual_roughness - reference_roughness,
                "residual_over_reference": residual_roughness / reference_roughness,
                "reference_iqr": float(reference["target_iqr"]),
                "residual_iqr": float(residual["target_iqr"]),
                "reference_variance": float(reference["target_variance"]),
                "residual_variance": float(residual["target_variance"]),
            }
        )

    df = pd.DataFrame(rows)
    order = ["xTB", "MLP-BoB", "MLP-PRE", "MLP-PRE+A"]
    df["order"] = df["baseline"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order", ascending=True).drop(columns=["order"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.65, 2.95), constrained_layout=True)
    y = list(range(len(df)))

    for yy, row in zip(y, df.to_dict(orient="records")):
        color = COLORS[row["baseline"]]
        ax.plot(
            [row["reference_d_iqr"], row["residual_d_iqr"]],
            [yy, yy],
            color=color,
            linewidth=1.45,
            alpha=0.68,
            zorder=2,
        )
        ax.scatter(
            row["reference_d_iqr"],
            yy,
            s=30,
            color="#374151",
            marker="o",
            edgecolor="#202020",
            linewidth=0.65,
            zorder=3,
            label=rf"$E_{{\mathrm{{ref}},\mathrm{{BoE}}}}$ ({reference_roughness:.3f})" if yy == 0 else None,
        )
        ax.scatter(
            row["residual_d_iqr"],
            yy,
            s=38,
            color="white",
            marker="D",
            edgecolor=color,
            linewidth=1.15,
            zorder=4,
            label="Residual target" if yy == 0 else None,
        )
        text_dx = -0.08 if row["residual_d_iqr"] < reference_roughness else 0.08
        text_ha = "right" if row["residual_d_iqr"] < reference_roughness else "left"
        ax.text(
            row["residual_d_iqr"] + text_dx,
            yy - 0.060,
            f"{row['residual_d_iqr']:.3f}",
            ha=text_ha,
            va="top",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )

    ax.axvline(reference_roughness, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["baseline"], fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"std-scaled PRE+A cosine roughness, $D_{\mathrm{IQR}}$", fontsize=8.2)
    ax.set_xlim(0.9, 30.0)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.grid(axis="x", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8, handletextpad=0.40)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE, index=False)
    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"WROTE: {OUT_TABLE}")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
