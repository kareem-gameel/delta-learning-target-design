#!/usr/bin/env python3
"""Paired Cleveland plot: residual roughness in raw PRE feature space."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from expanded_panel_common import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    TargetSpec,
    load_feature_space,
    load_split_ids,
    load_target_frame,
)
from run_expanded_smoothness import align_features, dirichlet_metrics, unique_undirected_edges  # noqa: E402


OUT = ROOT / "New_Analysis/manuscript_support/figures/baseline_to_residual_normalized_dirichlet_pre_space_cleveland.png"
OUT_TABLE = ROOT / "New_Analysis/manuscript_support/tables/baseline_to_residual_normalized_dirichlet_pre_space.csv"
EDGE_CACHE = ROOT / "New_Analysis/manuscript_support/tables/pre_space_raw_k10_shared_residual_edges.npz"

COLORS = {
    "xTB": "#4C78A8",
    "BoB": "#F58518",
    "PRE": "#54A24B",
    "PRE+A": "#B279A2",
}

TARGETS = {
    "reference": TargetSpec(
        "dft",
        "DFT_ref,BoE",
        "DFT",
        ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv",
        "DFT_ref-BoE",
        "reference",
    ),
    "xTB": TargetSpec(
        "deltaxtb",
        "Delta^xTB_ref,BoE",
        "Delta^xTB",
        ROOT / "New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv",
        "Delta_xTB-ref-BoE",
        "residual_target",
    ),
    "BoB": TargetSpec(
        "delta_mlp_bob",
        "Delta^MLP_BoB_ref,BoE",
        "Delta^MLP_BoB",
        ROOT / "New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
        "Delta_MLP_BoB-ref-BoE",
        "residual_target",
    ),
    "PRE": TargetSpec(
        "delta_mlp_pre",
        "Delta^MLP_PRE_ref,BoE",
        "Delta^MLP_PRE",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
        "Delta_MLP_PRE-ref-BoE",
        "residual_target",
    ),
    "PRE+A": TargetSpec(
        "delta_mlp_pre_plus_a",
        "Delta^MLP_PRE+A_ref,BoE",
        "Delta^MLP_PRE+A",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
        "Delta_MLP_PRE_plus_A-ref-BoE",
        "residual_target",
    ),
}

ROUGHNESS_COL = "edge_avg_norm_by_population_var"


def load_val_panel() -> pd.DataFrame:
    split_df = load_split_ids(DEFAULT_SPLIT_DIR)
    val_split = split_df[split_df["split"].eq("val")].copy()
    panel = val_split[["mol_id", "split"]].copy()
    for label, spec in TARGETS.items():
        frame = load_target_frame(spec, val_split)[["mol_id", "target"]].rename(columns={"target": label})
        panel = panel.merge(frame, on="mol_id", how="inner", validate="one_to_one")
    return panel


def build_or_load_edges(x, k: int = 10) -> np.ndarray:
    EDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if EDGE_CACHE.exists():
        cached = np.load(EDGE_CACHE)
        if int(cached["n_rows"]) == x.shape[0] and int(cached["n_cols"]) == x.shape[1] and int(cached["k"]) == k:
            return cached["edges"]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    edges = unique_undirected_edges(ind[:, 1:])
    np.savez_compressed(EDGE_CACHE, edges=edges, n_rows=x.shape[0], n_cols=x.shape[1], k=k)
    return edges


def main() -> None:
    panel = load_val_panel()
    x_all, feature_ids, feature_meta = load_feature_space("pre")
    x, aligned = align_features(x_all, feature_ids, panel)
    edges = build_or_load_edges(x, k=10)

    roughness = {}
    for label in TARGETS:
        y = aligned[label].to_numpy(float)
        roughness[label] = dirichlet_metrics(y, edges)

    reference_roughness = roughness["reference"][ROUGHNESS_COL]
    rows = []
    for label in ["xTB", "BoB", "PRE", "PRE+A"]:
        residual_roughness = roughness[label][ROUGHNESS_COL]
        rows.append(
            {
                "baseline": label,
                "feature_space": feature_meta["feature_space"],
                "feature_scaling": "raw",
                "reference_target": TARGETS["reference"].target_label,
                "residual_target_id": TARGETS[label].target_id,
                "n_molecules": int(x.shape[0]),
                "n_edges": int(edges.shape[0]),
                "reference_normalized_dirichlet": reference_roughness,
                "residual_normalized_dirichlet": residual_roughness,
                "delta_normalized_dirichlet": residual_roughness - reference_roughness,
                "residual_over_reference": residual_roughness / reference_roughness,
                "reference_edge_rmse": roughness["reference"]["edge_rmse"],
                "residual_edge_rmse": roughness[label]["edge_rmse"],
                "reference_target_iqr": roughness["reference"]["target_iqr"],
                "residual_target_iqr": roughness[label]["target_iqr"],
            }
        )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.65, 2.85), constrained_layout=True)
    y = list(range(len(df)))

    for yy, row in zip(y, df.to_dict(orient="records")):
        color = COLORS[row["baseline"]]
        ax.plot(
            [row["reference_normalized_dirichlet"], row["residual_normalized_dirichlet"]],
            [yy, yy],
            color=color,
            linewidth=1.45,
            alpha=0.68,
            zorder=2,
        )
        ax.scatter(
            row["reference_normalized_dirichlet"],
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
            row["residual_normalized_dirichlet"],
            yy,
            s=38,
            color="white",
            marker="D",
            edgecolor=color,
            linewidth=1.15,
            zorder=4,
            label="Residual target" if yy == 0 else None,
        )
        ax.text(
            row["residual_normalized_dirichlet"] + 0.014,
            yy - 0.060,
            f"{row['residual_normalized_dirichlet']:.3f}",
            ha="left",
            va="top",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )

    ax.axvline(reference_roughness, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["baseline"], fontsize=7.8)
    ax.set_xlabel(r"Normalized Dirichlet roughness in PRE space, $D_{\mathrm{norm}}$", fontsize=8.2)
    xmin = min(df["residual_normalized_dirichlet"].min(), reference_roughness) - 0.045
    xmax = max(df["residual_normalized_dirichlet"].max(), reference_roughness) + 0.075
    ax.set_xlim(xmin, xmax)
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
