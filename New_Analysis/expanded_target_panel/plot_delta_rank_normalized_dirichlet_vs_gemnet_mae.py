#!/usr/bin/env python3
"""Rank normalized Dirichlet smoothness against GemNet MAE for delta targets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from expanded_panel_common import TARGETS


ROOT = Path(__file__).resolve().parents[2]
DELTA_TARGETS = ["deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"]
TARGET_LABELS = {target.target_id: target.short_label for target in TARGETS}
FEATURE_SMOOTHNESS = {
    "PRE": ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv",
    "PRE+A": ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_plus_a_space_dirichlet_val.csv",
}
FEATURE_COLORS = {
    "PRE": "#2E8B57",
    "PRE+A": "#C44E52",
}


def build_frame() -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv")
    gem = metrics[
        metrics["model_id"].eq("GemNet")
        & metrics["split"].eq("val")
        & metrics["target_id"].isin(DELTA_TARGETS)
    ][["target_id", "mae", "mae_over_iqr"]].copy()
    gem["gemnet_mae_rank"] = gem["mae"].rank(method="average", ascending=True)

    rows = []
    for feature_space, path in FEATURE_SMOOTHNESS.items():
        smooth = pd.read_csv(path)
        sm = smooth[
            smooth["feature_scaling"].eq("raw")
            & smooth["k"].eq(10)
            & smooth["target_id"].isin(DELTA_TARGETS)
        ][["target_id", "edge_avg_norm_by_population_var"]].copy()
        sm["normalized_dirichlet_rank"] = sm["edge_avg_norm_by_population_var"].rank(method="average", ascending=True)
        merged = sm.merge(gem, on="target_id", how="inner", validate="one_to_one")
        merged["feature_space"] = feature_space
        merged["target_label"] = merged["target_id"].map(TARGET_LABELS)
        merged["rank_delta"] = merged["gemnet_mae_rank"] - merged["normalized_dirichlet_rank"]
        merged["spearman_rho"] = merged["normalized_dirichlet_rank"].corr(merged["gemnet_mae_rank"], method="spearman")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True).sort_values(["feature_space", "normalized_dirichlet_rank"])


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), dpi=240, sharex=True, sharey=True)
    for ax, feature_space in zip(axes, ["PRE", "PRE+A"]):
        sub = df[df["feature_space"].eq(feature_space)].copy()
        n = len(sub)
        ax.plot([1, n], [1, n], color="#C8CED8", linewidth=1.2, zorder=0)
        ax.scatter(
            sub["normalized_dirichlet_rank"],
            sub["gemnet_mae_rank"],
            s=82,
            color=FEATURE_COLORS[feature_space],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, rec in sub.iterrows():
            ax.annotate(
                rec["target_label"],
                (rec["normalized_dirichlet_rank"], rec["gemnet_mae_rank"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
            )
        rho = sub["spearman_rho"].iloc[0]
        ax.set_title(f"{feature_space} normalized Dirichlet\nrho={rho:.2f}", fontsize=11)
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.grid(color="#D8DEE9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Normalized Dirichlet rank\n(1 = smoothest)", fontsize=10)
    axes[0].set_ylabel("GemNet MAE rank\n(1 = lowest MAE)", fontsize=10)
    fig.suptitle("Delta targets: smoothness rank vs GemNet MAE rank", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "New_Analysis/expanded_target_panel/outputs/final_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    csv_path = out_dir / "delta_rank_normalized_dirichlet_vs_gemnet_mae.csv"
    png_path = out_dir / "delta_rank_normalized_dirichlet_vs_gemnet_mae.png"
    df.to_csv(csv_path, index=False)
    plot(df, png_path)
    print(f"WROTE: {csv_path}")
    print(f"WROTE: {png_path}")
    print(
        df[
            [
                "feature_space",
                "target_id",
                "target_label",
                "edge_avg_norm_by_population_var",
                "mae",
                "normalized_dirichlet_rank",
                "gemnet_mae_rank",
                "spearman_rho",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
