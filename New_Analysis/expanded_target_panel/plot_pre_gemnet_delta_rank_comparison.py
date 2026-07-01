#!/usr/bin/env python3
"""Focused PRE-space rank comparison for GemNet on delta targets only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expanded_panel_common import TARGETS


ROOT = Path(__file__).resolve().parents[2]
TARGET_LABELS = {target.target_id: target.short_label for target in TARGETS}
DELTA_TARGETS = ["deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"]


def main() -> None:
    metrics_path = ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv"
    smooth_path = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv"
    out_dir = ROOT / "New_Analysis/expanded_target_panel/outputs/final_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    gem = metrics[
        metrics["model_id"].eq("GemNet")
        & metrics["split"].eq("val")
        & metrics["target_id"].isin(DELTA_TARGETS)
    ][["target_id", "mae_over_iqr"]].copy()

    smooth = pd.read_csv(smooth_path)
    pre = smooth[
        smooth["feature_space"].eq("PRE")
        & smooth["feature_scaling"].eq("raw")
        & smooth["k"].eq(10)
        & smooth["target_id"].isin(DELTA_TARGETS)
    ][["target_id", "edge_avg_norm_by_population_var"]].copy()

    df = pre.merge(gem, on="target_id", how="inner", validate="one_to_one")
    df["smoothness_rank"] = df["edge_avg_norm_by_population_var"].rank(method="average", ascending=True)
    df["gemnet_nmae_rank"] = df["mae_over_iqr"].rank(method="average", ascending=True)
    df["rank_delta"] = df["gemnet_nmae_rank"] - df["smoothness_rank"]
    df["spearman_rho"] = df["smoothness_rank"].corr(df["gemnet_nmae_rank"], method="spearman")
    df["target_label"] = df["target_id"].map(TARGET_LABELS)
    df = df.sort_values(["smoothness_rank", "gemnet_nmae_rank"]).reset_index(drop=True)

    csv_path = out_dir / "pre_space_gemnet_delta_rank_comparison.csv"
    df.to_csv(csv_path, index=False)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=240)
    for yi, rec in zip(y, df.itertuples(index=False)):
        ax.plot(
            [rec.smoothness_rank, rec.gemnet_nmae_rank],
            [yi, yi],
            color="#AAB2BF",
            linewidth=1.6,
            zorder=1,
        )
    ax.scatter(
        df["smoothness_rank"],
        y,
        s=70,
        color="#7A869A",
        alpha=0.65,
        label="normalized Dirichlet rank",
        zorder=2,
    )
    ax.scatter(
        df["gemnet_nmae_rank"],
        y,
        s=70,
        color="#355C9A",
        alpha=0.65,
        label="GemNet MAE/IQR rank",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["target_label"], fontsize=10)
    ax.set_xticks(range(1, len(df) + 1))
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 = smoothest/easiest)", fontsize=11)
    ax.set_title(
        f"PRE feature space: normalized smoothness rank vs GemNet normalized MAE rank\n"
        f"delta targets only, rho={df['spearman_rho'].iloc[0]:.2f}",
        fontsize=12,
    )
    ax.grid(axis="x", color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = out_dir / "pre_space_gemnet_delta_rank_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)

    print(f"WROTE: {csv_path}")
    print(f"WROTE: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
