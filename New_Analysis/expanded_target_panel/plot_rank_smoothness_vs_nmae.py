#!/usr/bin/env python3
"""Rank-based visualizations for smoothness vs model learnability.

Uses normalized Dirichlet smoothness and normalized MAE with raw feature scaling.
These plots are designed to support Spearman/rank-order interpretation rather
than a linear scatter interpretation.
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
FEATURE_SPACES = ["BoB", "PRE", "PRE+A"]
MODEL_ORDER = ["BoB", "PRE", "PRE_plus_A", "GemNet", "SchNet"]
MODEL_LABELS = {
    "BoB": "BoB ridge",
    "PRE": "PRE ridge",
    "PRE_plus_A": "PRE+A ridge",
    "GemNet": "GemNet",
    "SchNet": "SchNet",
}
MODEL_COLORS = {
    "BoB": "#111111",
    "PRE": "#2E8B57",
    "PRE_plus_A": "#C44E52",
    "GemNet": "#355C9A",
    "SchNet": "#6C5AB5",
}
SMOOTHNESS_COLOR = "#7A869A"
ERROR_COLOR = "#D1495B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv",
    )
    parser.add_argument(
        "--smoothness-dir",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/final_plots",
    )
    parser.add_argument(
        "--exclude-targets",
        default="",
        help="Comma-separated target IDs to exclude from rank plots.",
    )
    return parser.parse_args()


def excluded_targets(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def load_metrics(path: Path, exclude: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].eq("val")].copy()
    df = df[df["target_id"].isin(TARGET_ORDER)]
    df = df[~df["target_id"].isin(exclude)]
    df = df[df["model_id"].isin(MODEL_ORDER)]
    return df


def load_smoothness(smoothness_dir: Path, exclude: set[str]) -> pd.DataFrame:
    paths = [
        smoothness_dir / "bob_space_dirichlet_val.csv",
        smoothness_dir / "pre_space_dirichlet_val.csv",
        smoothness_dir / "pre_plus_a_space_dirichlet_val.csv",
    ]
    pieces = [pd.read_csv(path) for path in paths if path.exists()]
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    df = df[df["target_id"].isin(TARGET_ORDER)]
    df = df[~df["target_id"].isin(exclude)]
    return df


def make_rank_frame(metrics: pd.DataFrame, smoothness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    smooth_col = "edge_avg_norm_by_population_var"
    for feature_space in FEATURE_SPACES:
        sm = smoothness[
            smoothness["feature_space"].eq(feature_space)
            & smoothness["feature_scaling"].eq("raw")
            & smoothness["k"].eq(10)
        ][["target_id", "target_label", "short_label", smooth_col]].copy()
        if sm.empty:
            continue
        sm["smoothness_rank"] = sm[smooth_col].rank(method="average", ascending=True)
        for model_id in MODEL_ORDER:
            mm = metrics[metrics["model_id"].eq(model_id)][["target_id", "mae_over_iqr"]].copy()
            merged = sm.merge(mm, on="target_id", how="inner")
            if merged.empty:
                continue
            merged["mae_rank"] = merged["mae_over_iqr"].rank(method="average", ascending=True)
            merged["feature_space"] = feature_space
            merged["feature_scaling"] = "raw"
            merged["model_id"] = model_id
            merged["rank_delta"] = merged["mae_rank"] - merged["smoothness_rank"]
            merged["abs_rank_delta"] = merged["rank_delta"].abs()
            rho = merged["smoothness_rank"].corr(merged["mae_rank"], method="spearman")
            merged["spearman_rho"] = rho
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["target_order"] = out["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    return out.sort_values(["feature_space", "model_id", "smoothness_rank", "mae_rank"])


def plot_rank_rank(ranks: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(
        len(FEATURE_SPACES),
        len(MODEL_ORDER),
        figsize=(17.5, 9.2),
        dpi=220,
        sharex=True,
        sharey=True,
    )
    for r, feature_space in enumerate(FEATURE_SPACES):
        for c, model_id in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            sub = ranks[ranks["feature_space"].eq(feature_space) & ranks["model_id"].eq(model_id)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            n = int(sub["target_id"].nunique())
            ax.plot([1, n], [1, n], color="#C8CED8", linewidth=1.2, zorder=0)
            ax.scatter(
                sub["smoothness_rank"],
                sub["mae_rank"],
                s=48,
                color=MODEL_COLORS[model_id],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.94,
            )
            for _, rec in sub.iterrows():
                ax.annotate(
                    TARGET_LABELS.get(rec["target_id"], rec["target_id"]),
                    (rec["smoothness_rank"], rec["mae_rank"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7.1,
                )
            rho = sub["spearman_rho"].iloc[0]
            ax.set_title(f"{feature_space} space vs {MODEL_LABELS[model_id]}\nrho={rho:.2f}", fontsize=9.3)
            ax.set_xticks(range(1, n + 1))
            ax.set_yticks(range(1, n + 1))
            ax.grid(color="#D8DEE9", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if r == len(FEATURE_SPACES) - 1:
                ax.set_xlabel("Smoothness rank\n(1 = smoothest)", fontsize=8.8)
            if c == 0:
                ax.set_ylabel("MAE/IQR rank\n(1 = easiest)", fontsize=8.8)
    fig.suptitle("Rank-rank relation: normalized Dirichlet smoothness vs normalized validation MAE", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    plt.close(fig)


def plot_cleveland(ranks: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(
        len(FEATURE_SPACES),
        len(MODEL_ORDER),
        figsize=(18, 11),
        dpi=220,
        sharex=True,
        sharey=False,
    )
    for r, feature_space in enumerate(FEATURE_SPACES):
        for c, model_id in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            sub = ranks[ranks["feature_space"].eq(feature_space) & ranks["model_id"].eq(model_id)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            sub = sub.sort_values(["smoothness_rank", "mae_rank"]).reset_index(drop=True)
            y = np.arange(len(sub))
            for yi, rec in zip(y, sub.itertuples(index=False)):
                ax.plot(
                    [rec.smoothness_rank, rec.mae_rank],
                    [yi, yi],
                    color="#AAB2BF",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=1,
                )
            ax.scatter(
                sub["smoothness_rank"],
                y,
                s=44,
                color=SMOOTHNESS_COLOR,
                alpha=0.62,
                label="smoothness rank",
                zorder=2,
            )
            ax.scatter(
                sub["mae_rank"],
                y,
                s=44,
                color=ERROR_COLOR,
                alpha=0.62,
                label="MAE/IQR rank",
                zorder=3,
            )
            ax.set_yticks(y)
            ax.set_yticklabels([TARGET_LABELS.get(tid, tid) for tid in sub["target_id"]], fontsize=7.5)
            rho = sub["spearman_rho"].iloc[0]
            ax.set_title(f"{feature_space} vs {MODEL_LABELS[model_id]}\nrho={rho:.2f}", fontsize=9.1)
            ax.set_xticks(range(1, len(sub) + 1))
            ax.invert_yaxis()
            ax.grid(axis="x", color="#D8DEE9", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if r == len(FEATURE_SPACES) - 1:
                ax.set_xlabel("Rank (1 = best)", fontsize=8.8)
            if r == 0 and c == len(MODEL_ORDER) - 1:
                ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Sorted paired-rank Cleveland plot: smoothness rank vs model-error rank", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exclude = excluded_targets(args.exclude_targets)

    metrics = load_metrics(args.metrics, exclude)
    smoothness = load_smoothness(args.smoothness_dir, exclude)
    ranks = make_rank_frame(metrics, smoothness)

    rank_out = args.out_dir / "rank_smoothness_vs_nmae_points.csv"
    ranks.to_csv(rank_out, index=False)
    plot_rank_rank(ranks, args.out_dir / "rank_rank_smoothness_vs_nmae_raw.png")
    plot_cleveland(ranks, args.out_dir / "cleveland_rank_smoothness_vs_nmae_raw.png")

    print(f"WROTE: {rank_out}")
    print(f"WROTE: {args.out_dir / 'rank_rank_smoothness_vs_nmae_raw.png'}")
    print(f"WROTE: {args.out_dir / 'cleveland_rank_smoothness_vs_nmae_raw.png'}")


if __name__ == "__main__":
    main()
