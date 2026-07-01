#!/usr/bin/env python3
"""Plot the expanded target panel from the generated status/metrics tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expanded_panel_common import TARGETS


ROOT = Path(__file__).resolve().parents[2]
TARGET_ORDER = [target.target_id for target in TARGETS]
TARGET_LABELS = {target.target_id: target.short_label for target in TARGETS}
MODEL_ORDER = ["GemNet", "SchNet", "BoB", "PRE", "PRE_plus_A"]
MODEL_LABELS = {
    "GemNet": "GemNet",
    "SchNet": "SchNet",
    "BoB": "BoB ridge",
    "PRE": "PRE ridge",
    "PRE_plus_A": "PRE+A ridge",
}
MODEL_COLORS = {
    "GemNet": "#355C9A",
    "SchNet": "#6C5AB5",
    "BoB": "#111111",
    "PRE": "#2E8B57",
    "PRE_plus_A": "#C44E52",
}


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
        "--exclude-smoothness-targets",
        default="deltapre_plus_a",
        help="Comma-separated target IDs to exclude from smoothness scatter plots and correlations.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].eq("val")].copy()
    df = df[df["target_id"].isin(TARGET_ORDER)]
    df = df[df["model_id"].isin(MODEL_ORDER)]
    df["target_order"] = df["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    df["model_order"] = df["model_id"].map({model_id: i for i, model_id in enumerate(MODEL_ORDER)})
    return df.sort_values(["target_order", "model_order"]).reset_index(drop=True)


def plot_metric_bars(metrics: pd.DataFrame, value_col: str, ylabel: str, out: Path) -> None:
    pivot = metrics.pivot_table(index="target_id", columns="model_id", values=value_col, aggfunc="first")
    pivot = pivot.reindex(TARGET_ORDER)
    labels = [TARGET_LABELS[target_id] for target_id in TARGET_ORDER]
    x = np.arange(len(TARGET_ORDER))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=220)
    for i, model_id in enumerate(MODEL_ORDER):
        vals = pivot[model_id].to_numpy(float) if model_id in pivot else np.full(len(TARGET_ORDER), np.nan)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            if math.isfinite(val):
                text = f"{val:.3f}" if value_col == "mae_over_iqr" else f"{val:.4f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90 if value_col == "mae_over_iqr" else 0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_smoothness(smoothness_dir: Path) -> pd.DataFrame:
    paths = [
        smoothness_dir / "bob_space_dirichlet_val.csv",
        smoothness_dir / "pre_space_dirichlet_val.csv",
        smoothness_dir / "pre_plus_a_space_dirichlet_val.csv",
    ]
    pieces = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["target_id"].isin(TARGET_ORDER)].copy()
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def excluded_targets(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def correlation_table(metrics: pd.DataFrame, smoothness: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    rows = []
    if smoothness.empty:
        return pd.DataFrame()
    metrics = metrics[~metrics["target_id"].isin(exclude)].copy()
    smoothness = smoothness[~smoothness["target_id"].isin(exclude)].copy()
    metric_cols = ["mae", "mae_over_iqr", "r2", "kendall_tau"]
    smooth_cols = ["edge_avg_norm_by_population_var", "edge_rmse_over_iqr"]
    for feature_space in sorted(smoothness["feature_space"].dropna().unique()):
        for scaling in ["std", "raw"]:
            sm = smoothness[
                smoothness["feature_space"].eq(feature_space)
                & smoothness["feature_scaling"].eq(scaling)
                & smoothness["k"].eq(10)
            ]
            if sm.empty:
                continue
            for model_id in MODEL_ORDER:
                mm = metrics[metrics["model_id"].eq(model_id)]
                merged = sm.merge(mm[["target_id", *metric_cols]], on="target_id", how="inner")
                for smooth_col in smooth_cols:
                    for metric_col in metric_cols:
                        ok = merged[[smooth_col, metric_col]].replace([np.inf, -np.inf], np.nan).dropna()
                        rows.append(
                            {
                                "feature_space": feature_space,
                                "feature_scaling": scaling,
                                "model_id": model_id,
                                "x_metric": smooth_col,
                                "y_metric": metric_col,
                                "n_targets": len(ok),
                                "pearson_r": ok[smooth_col].corr(ok[metric_col], method="pearson")
                                if len(ok) >= 2
                                else np.nan,
                                "spearman_r": ok[smooth_col].corr(ok[metric_col], method="spearman")
                                if len(ok) >= 2
                                else np.nan,
                            }
                        )
    return pd.DataFrame(rows)


def plot_smoothness_vs_nmae(metrics: pd.DataFrame, smoothness: pd.DataFrame, out: Path, exclude: set[str]) -> None:
    metrics = metrics[~metrics["target_id"].isin(exclude)].copy()
    smoothness = smoothness[~smoothness["target_id"].isin(exclude)].copy()
    rows = []
    for feature_space in ["BoB", "PRE", "PRE+A"]:
        sm = smoothness[
            smoothness["feature_space"].eq(feature_space)
            & smoothness["feature_scaling"].eq("std")
            & smoothness["k"].eq(10)
        ]
        for model_id in ["BoB", "PRE", "PRE_plus_A", "GemNet", "SchNet"]:
            mm = metrics[metrics["model_id"].eq(model_id)]
            merged = sm.merge(mm[["target_id", "mae_over_iqr"]], on="target_id", how="inner")
            for _, rec in merged.iterrows():
                rows.append(
                    {
                        "feature_space": feature_space,
                        "model_id": model_id,
                        "target_id": rec["target_id"],
                        "x": rec["edge_avg_norm_by_population_var"],
                        "y": rec["mae_over_iqr"],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(3, 5, figsize=(17, 9), dpi=220, sharex=False, sharey=False)
    for r, feature_space in enumerate(["BoB", "PRE", "PRE+A"]):
        for c, model_id in enumerate(["BoB", "PRE", "PRE_plus_A", "GemNet", "SchNet"]):
            ax = axes[r, c]
            sub = df[df["feature_space"].eq(feature_space) & df["model_id"].eq(model_id)]
            if sub.empty:
                ax.axis("off")
                continue
            pearson = sub["x"].corr(sub["y"], method="pearson") if len(sub) >= 2 else np.nan
            ax.scatter(sub["x"], sub["y"], s=42, color=MODEL_COLORS[model_id], edgecolor="white", linewidth=0.7)
            for _, rec in sub.iterrows():
                ax.annotate(
                    TARGET_LABELS.get(rec["target_id"], rec["target_id"]),
                    (rec["x"], rec["y"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7.5,
                )
            ax.set_title(f"{feature_space} space vs {MODEL_LABELS[model_id]}\nr={pearson:.2f}", fontsize=9.5)
            ax.grid(color="#D8DEE9", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if r == 2:
                ax.set_xlabel("Dirichlet smoothness", fontsize=9)
            if c == 0:
                ax.set_ylabel("Val MAE / IQR", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics)
    smoothness = load_smoothness(args.smoothness_dir)
    smoothness_exclude = excluded_targets(args.exclude_smoothness_targets)

    metrics_out = args.out_dir / "expanded_model_target_val_metrics_available.csv"
    corr_out = args.out_dir / "expanded_smoothness_correlations_available.csv"
    metrics.to_csv(metrics_out, index=False)
    correlation_table(metrics, smoothness, smoothness_exclude).to_csv(corr_out, index=False)

    plot_metric_bars(metrics, "mae", "tmQM val MAE (Ha)", args.out_dir / "expanded_model_target_val_mae.png")
    plot_metric_bars(
        metrics,
        "mae_over_iqr",
        "tmQM val MAE / target IQR",
        args.out_dir / "expanded_model_target_val_nmae_iqr.png",
    )
    plot_smoothness_vs_nmae(
        metrics,
        smoothness,
        args.out_dir / "expanded_smoothness_vs_nmae_available_std.png",
        smoothness_exclude,
    )

    print(f"WROTE: {metrics_out}")
    print(f"WROTE: {corr_out}")
    print(f"WROTE: {args.out_dir / 'expanded_model_target_val_mae.png'}")
    print(f"WROTE: {args.out_dir / 'expanded_model_target_val_nmae_iqr.png'}")
    print(f"WROTE: {args.out_dir / 'expanded_smoothness_vs_nmae_available_std.png'}")


if __name__ == "__main__":
    main()
