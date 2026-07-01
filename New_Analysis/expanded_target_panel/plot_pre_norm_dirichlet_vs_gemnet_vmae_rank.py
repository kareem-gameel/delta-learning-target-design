#!/usr/bin/env python3
"""PRE-space normalized Dirichlet rank vs GemNet validation MAE rank."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expanded_panel_common import TARGETS


ROOT = Path(__file__).resolve().parents[2]
TARGET_ORDER = [target.target_id for target in TARGETS]
TARGET_LABELS = {target.target_id: target.short_label for target in TARGETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv",
    )
    parser.add_argument(
        "--smoothness",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv",
    )
    parser.add_argument("--feature-scaling", default="raw", choices=["raw", "std"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--exclude-targets",
        default="",
        help="Comma-separated target IDs to exclude.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/final_plots",
    )
    return parser.parse_args()


def excluded_targets(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_frame(args: argparse.Namespace) -> pd.DataFrame:
    exclude = excluded_targets(args.exclude_targets)

    smooth = pd.read_csv(args.smoothness)
    smooth = smooth[
        smooth["target_id"].isin(TARGET_ORDER)
        & ~smooth["target_id"].isin(exclude)
        & smooth["feature_space"].eq("PRE")
        & smooth["feature_scaling"].eq(args.feature_scaling)
        & smooth["k"].eq(args.k)
    ].copy()
    smooth["normalized_dirichlet"] = smooth["edge_avg_norm_by_population_var"].astype(float)

    metrics = pd.read_csv(args.metrics)
    metrics = metrics[
        metrics["model_id"].eq("GemNet")
        & metrics["split"].eq("val")
        & metrics["target_id"].isin(TARGET_ORDER)
        & ~metrics["target_id"].isin(exclude)
    ][["target_id", "mae", "mae_over_iqr"]].copy()
    metrics = metrics.rename(columns={"mae": "gemnet_val_mae", "mae_over_iqr": "gemnet_val_mae_over_iqr"})

    df = smooth.merge(metrics, on="target_id", how="inner", validate="one_to_one")
    df["target_label_plot"] = df["target_id"].map(TARGET_LABELS)
    df["target_order"] = df["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    df["norm_dirichlet_rank"] = df["normalized_dirichlet"].rank(method="average", ascending=True)
    df["gemnet_vmae_rank"] = df["gemnet_val_mae"].rank(method="average", ascending=True)
    df["rank_delta"] = df["gemnet_vmae_rank"] - df["norm_dirichlet_rank"]
    df["abs_rank_delta"] = df["rank_delta"].abs()
    rho = df["norm_dirichlet_rank"].corr(df["gemnet_vmae_rank"], method="spearman") if len(df) >= 2 else np.nan
    df["spearman_rho"] = rho
    return df.sort_values(["norm_dirichlet_rank", "gemnet_vmae_rank", "target_order"]).reset_index(drop=True)


def plot_rank_rank(df: pd.DataFrame, out: Path, *, feature_scaling: str) -> None:
    if df.empty:
        return
    n = df["target_id"].nunique()
    rho = df["spearman_rho"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=220)
    ax.plot([1, n], [1, n], color="#C8CED8", linewidth=1.3, zorder=0)
    ax.scatter(
        df["norm_dirichlet_rank"],
        df["gemnet_vmae_rank"],
        s=62,
        color="#355C9A",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.94,
    )
    for _, rec in df.iterrows():
        ax.annotate(
            rec["target_label_plot"],
            (rec["norm_dirichlet_rank"], rec["gemnet_vmae_rank"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.6,
        )
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("PRE-space normalized Dirichlet rank\n(1 = smoothest)", fontsize=11)
    ax.set_ylabel("GemNet val MAE rank\n(1 = lowest MAE)", fontsize=11)
    ax.set_title(f"PRE-space target smoothness rank vs GemNet val MAE rank\nrho={rho:.2f}, {feature_scaling} feature scaling", fontsize=12.5)
    ax.grid(color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_cleveland(df: pd.DataFrame, out: Path, *, feature_scaling: str) -> None:
    if df.empty:
        return
    sub = df.sort_values(["norm_dirichlet_rank", "gemnet_vmae_rank"]).reset_index(drop=True)
    y = np.arange(len(sub))
    rho = sub["spearman_rho"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=220)
    for yi, rec in zip(y, sub.itertuples(index=False)):
        ax.plot([rec.norm_dirichlet_rank, rec.gemnet_vmae_rank], [yi, yi], color="#AAB2BF", linewidth=1.4, zorder=1)
    ax.scatter(sub["norm_dirichlet_rank"], y, s=50, color="#7A869A", alpha=0.65, label="smoothness rank", zorder=2)
    ax.scatter(sub["gemnet_vmae_rank"], y, s=50, color="#355C9A", alpha=0.65, label="GemNet MAE rank", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["target_label_plot"], fontsize=10)
    ax.set_xticks(range(1, len(sub) + 1))
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 = best)", fontsize=11)
    ax.set_title(f"PRE-space normalized Dirichlet rank vs GemNet val MAE rank\nrho={rho:.2f}, {feature_scaling} feature scaling", fontsize=12.5)
    ax.grid(axis="x", color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_frame(args)

    suffix = f"{args.feature_scaling}_k{args.k}"
    csv_path = args.out_dir / f"pre_norm_dirichlet_vs_gemnet_vmae_rank_{suffix}.csv"
    rank_path = args.out_dir / f"pre_norm_dirichlet_vs_gemnet_vmae_rank_rank_{suffix}.png"
    cleveland_path = args.out_dir / f"pre_norm_dirichlet_vs_gemnet_vmae_rank_cleveland_{suffix}.png"

    df.to_csv(csv_path, index=False)
    plot_rank_rank(df, rank_path, feature_scaling=args.feature_scaling)
    plot_cleveland(df, cleveland_path, feature_scaling=args.feature_scaling)

    print(f"WROTE: {csv_path}")
    print(f"WROTE: {rank_path}")
    print(f"WROTE: {cleveland_path}")
    if not df.empty:
        cols = [
            "target_id",
            "target_label_plot",
            "normalized_dirichlet",
            "norm_dirichlet_rank",
            "gemnet_val_mae",
            "gemnet_vmae_rank",
            "rank_delta",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
