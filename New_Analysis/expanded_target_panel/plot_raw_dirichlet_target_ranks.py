#!/usr/bin/env python3
"""Rank targets by raw graph Dirichlet smoothness.

Raw Dirichlet here means the unnormalized edge-averaged graph energy:

    edge_mean_squared_diff = mean_edges((y_i - y_j)^2)

Lower values mean smoother targets in the selected feature space. This script
uses raw feature scaling by default and ranks targets separately in BoB, PRE,
and PRE+A feature spaces.
"""

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
FEATURE_ORDER = ["BoB", "PRE", "PRE+A"]
FEATURE_COLORS = {
    "BoB": "#111111",
    "PRE": "#2E8B57",
    "PRE+A": "#C44E52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoothness-dir",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness",
    )
    parser.add_argument(
        "--feature-scaling",
        default="raw",
        choices=["raw", "std"],
        help="Feature scaling used for the kNN graph.",
    )
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


def load_smoothness(smoothness_dir: Path) -> pd.DataFrame:
    paths = [
        smoothness_dir / "bob_space_dirichlet_val.csv",
        smoothness_dir / "pre_space_dirichlet_val.csv",
        smoothness_dir / "pre_plus_a_space_dirichlet_val.csv",
    ]
    pieces = [pd.read_csv(path) for path in paths if path.exists()]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def rank_frame(args: argparse.Namespace) -> pd.DataFrame:
    exclude = excluded_targets(args.exclude_targets)
    df = load_smoothness(args.smoothness_dir)
    df = df[
        df["target_id"].isin(TARGET_ORDER)
        & ~df["target_id"].isin(exclude)
        & df["feature_scaling"].eq(args.feature_scaling)
        & df["k"].eq(args.k)
    ].copy()
    if df.empty:
        return df

    df["raw_dirichlet"] = df["edge_mean_squared_diff"].astype(float)
    df["raw_dirichlet_rmse"] = df["edge_rmse"].astype(float)
    df["raw_dirichlet_rank"] = df.groupby("feature_space")["raw_dirichlet"].rank(
        method="average",
        ascending=True,
    )
    df["target_label_plot"] = df["target_id"].map(TARGET_LABELS)
    df["target_order"] = df["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    mean_rank = df.groupby("target_id")["raw_dirichlet_rank"].mean().rename("mean_raw_dirichlet_rank")
    df = df.merge(mean_rank, on="target_id", how="left", validate="many_to_one")
    return df.sort_values(["mean_raw_dirichlet_rank", "target_order", "feature_space"])


def plot_rank_dot(df: pd.DataFrame, out: Path, *, feature_scaling: str) -> None:
    if df.empty:
        return
    target_order = (
        df[["target_id", "target_label_plot", "mean_raw_dirichlet_rank"]]
        .drop_duplicates()
        .sort_values(["mean_raw_dirichlet_rank", "target_id"])
    )
    y_lookup = {target_id: i for i, target_id in enumerate(target_order["target_id"])}

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=220)
    for feature_space in FEATURE_ORDER:
        sub = df[df["feature_space"].eq(feature_space)].copy()
        if sub.empty:
            continue
        ax.scatter(
            sub["raw_dirichlet_rank"],
            sub["target_id"].map(y_lookup),
            s=68,
            color=FEATURE_COLORS[feature_space],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            label=feature_space,
        )

    ax.set_yticks(range(len(target_order)))
    ax.set_yticklabels(target_order["target_label_plot"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Raw Dirichlet rank (1 = smoothest)", fontsize=11)
    ax.set_title(f"Target ranks by raw Dirichlet smoothness ({feature_scaling} feature scaling)", fontsize=13)
    ax.grid(axis="x", color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_raw_values(df: pd.DataFrame, out: Path, *, feature_scaling: str) -> None:
    if df.empty:
        return
    target_order = (
        df[["target_id", "target_label_plot", "mean_raw_dirichlet_rank"]]
        .drop_duplicates()
        .sort_values(["mean_raw_dirichlet_rank", "target_id"])
    )
    y_lookup = {target_id: i for i, target_id in enumerate(target_order["target_id"])}

    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=220)
    for feature_space in FEATURE_ORDER:
        sub = df[df["feature_space"].eq(feature_space)].copy()
        if sub.empty:
            continue
        ax.scatter(
            sub["raw_dirichlet"],
            sub["target_id"].map(y_lookup),
            s=62,
            color=FEATURE_COLORS[feature_space],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.8,
            label=feature_space,
        )

    ax.set_xscale("log")
    ax.set_yticks(range(len(target_order)))
    ax.set_yticklabels(target_order["target_label_plot"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Raw edge Dirichlet energy, mean_edges[(yi - yj)^2] (Ha^2)", fontsize=10.5)
    ax.set_title(f"Raw target smoothness values ({feature_scaling} feature scaling)", fontsize=13)
    ax.grid(axis="x", color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_separate_rank_plots(df: pd.DataFrame, out_dir: Path, *, feature_scaling: str, k: int) -> None:
    for feature_space in FEATURE_ORDER:
        sub = df[df["feature_space"].eq(feature_space)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["raw_dirichlet_rank", "raw_dirichlet", "target_order"]).reset_index(drop=True)
        y = np.arange(len(sub))

        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
        ax.scatter(
            sub["raw_dirichlet_rank"],
            y,
            s=76,
            color=FEATURE_COLORS[feature_space],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.9,
        )
        for yi, rec in zip(y, sub.itertuples(index=False)):
            ax.text(
                rec.raw_dirichlet_rank + 0.08,
                yi,
                f"{rec.raw_dirichlet:.4g}",
                va="center",
                ha="left",
                fontsize=8.2,
                color="#3B4252",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["target_label_plot"], fontsize=10)
        ax.set_xticks(range(1, len(sub) + 1))
        ax.invert_yaxis()
        ax.set_xlabel("Raw Dirichlet rank (1 = smoothest)", fontsize=11)
        ax.set_title(f"{feature_space} feature space: raw Dirichlet target ranking", fontsize=13)
        ax.grid(axis="x", color="#D8DEE9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, len(sub) + 1.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        safe_name = feature_space.lower().replace("+", "plus")
        out = out_dir / f"raw_dirichlet_target_rank_{safe_name}_{feature_scaling}_k{k}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"WROTE: {out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = rank_frame(args)
    suffix = f"{args.feature_scaling}_k{args.k}"
    csv_path = args.out_dir / f"raw_dirichlet_target_ranks_{suffix}.csv"
    rank_path = args.out_dir / f"raw_dirichlet_target_rank_plot_{suffix}.png"
    values_path = args.out_dir / f"raw_dirichlet_target_values_{suffix}.png"

    df.to_csv(csv_path, index=False)
    plot_rank_dot(df, rank_path, feature_scaling=args.feature_scaling)
    plot_raw_values(df, values_path, feature_scaling=args.feature_scaling)
    plot_separate_rank_plots(df, args.out_dir, feature_scaling=args.feature_scaling, k=args.k)

    print(f"WROTE: {csv_path}")
    print(f"WROTE: {rank_path}")
    print(f"WROTE: {values_path}")
    if not df.empty:
        cols = ["feature_space", "target_id", "target_label_plot", "raw_dirichlet", "raw_dirichlet_rank"]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
