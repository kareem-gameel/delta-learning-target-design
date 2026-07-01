#!/usr/bin/env python3
"""Plot absolute graph Dirichlet smoothness against raw validation MAE.

This deliberately avoids normalized Dirichlet scores and normalized MAE. The
main x-axis is the edge-averaged graph Dirichlet energy:

    edge_mean_squared_diff = mean_edges((y_i - y_j)^2)

with units of Ha^2. A companion table also reports correlations for
``edge_rmse`` because it has the same units as MAE.
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
SCALINGS = ["raw", "std"]
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
        default="deltapre_plus_a",
        help="Comma-separated target IDs to exclude from plotted points and correlations.",
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
    pieces = []
    for path in paths:
        if path.exists():
            pieces.append(pd.read_csv(path))
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    df = df[df["target_id"].isin(TARGET_ORDER)].copy()
    return df[~df["target_id"].isin(exclude)].copy()


def merged_rows(metrics: pd.DataFrame, smoothness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["target_id", "model_id", "mae", "mae_over_iqr", "target_iqr"]
    for feature_space in FEATURE_SPACES:
        for scaling in SCALINGS:
            sm = smoothness[
                smoothness["feature_space"].eq(feature_space)
                & smoothness["feature_scaling"].eq(scaling)
                & smoothness["k"].eq(10)
            ].copy()
            if sm.empty:
                continue
            for model_id in MODEL_ORDER:
                mm = metrics[metrics["model_id"].eq(model_id)][metric_cols]
                merged = sm.merge(mm, on="target_id", how="inner")
                if merged.empty:
                    continue
                merged["model_id"] = model_id
                rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["target_order"] = out["target_id"].map({target_id: i for i, target_id in enumerate(TARGET_ORDER)})
    return out.sort_values(["feature_space", "feature_scaling", "model_id", "target_order"])


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_space, scaling, model_id), sub in df.groupby(["feature_space", "feature_scaling", "model_id"]):
        for x_metric in ["edge_mean_squared_diff", "edge_rmse"]:
            ok = sub[[x_metric, "mae"]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "feature_space": feature_space,
                    "feature_scaling": scaling,
                    "model_id": model_id,
                    "x_metric": x_metric,
                    "y_metric": "mae",
                    "n_targets": len(ok),
                    "pearson_r": ok[x_metric].corr(ok["mae"], method="pearson") if len(ok) >= 2 else np.nan,
                    "spearman_r": ok[x_metric].corr(ok["mae"], method="spearman") if len(ok) >= 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_scaling(df: pd.DataFrame, scaling: str, out: Path) -> None:
    sub_all = df[df["feature_scaling"].eq(scaling)].copy()
    if sub_all.empty:
        return

    fig, axes = plt.subplots(
        len(FEATURE_SPACES),
        len(MODEL_ORDER),
        figsize=(18, 9.2),
        dpi=220,
        sharex=False,
        sharey=False,
    )
    for r, feature_space in enumerate(FEATURE_SPACES):
        for c, model_id in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            sub = sub_all[
                sub_all["feature_space"].eq(feature_space)
                & sub_all["model_id"].eq(model_id)
            ].copy()
            if sub.empty:
                ax.axis("off")
                continue
            pearson = sub["edge_mean_squared_diff"].corr(sub["mae"], method="pearson") if len(sub) >= 2 else np.nan
            spearman = sub["edge_mean_squared_diff"].corr(sub["mae"], method="spearman") if len(sub) >= 2 else np.nan

            ax.scatter(
                sub["edge_mean_squared_diff"],
                sub["mae"],
                s=42,
                color=MODEL_COLORS[model_id],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
            )
            for _, rec in sub.iterrows():
                ax.annotate(
                    TARGET_LABELS.get(rec["target_id"], rec["target_id"]),
                    (rec["edge_mean_squared_diff"], rec["mae"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7.2,
                )

            ax.set_title(
                f"{feature_space} space vs {MODEL_LABELS[model_id]}\n"
                f"r={pearson:.2f}, rho={spearman:.2f}",
                fontsize=9.2,
            )
            ax.grid(color="#D8DEE9", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if r == len(FEATURE_SPACES) - 1:
                ax.set_xlabel("Mean edge Dirichlet energy (Ha$^2$)", fontsize=9)
            if c == 0:
                ax.set_ylabel("Val MAE (Ha)", fontsize=9)
    fig.suptitle(f"Absolute Dirichlet smoothness vs raw validation MAE ({scaling} feature scaling)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exclude = excluded_targets(args.exclude_targets)

    metrics = load_metrics(args.metrics, exclude)
    smoothness = load_smoothness(args.smoothness_dir, exclude)
    merged = merged_rows(metrics, smoothness)

    merged_out = args.out_dir / "absolute_dirichlet_vs_mae_points.csv"
    corr_out = args.out_dir / "absolute_dirichlet_vs_mae_correlations.csv"
    merged.to_csv(merged_out, index=False)
    correlation_table(merged).to_csv(corr_out, index=False)

    for scaling in SCALINGS:
        plot_scaling(
            merged,
            scaling,
            args.out_dir / f"absolute_dirichlet_vs_mae_{scaling}.png",
        )

    print(f"WROTE: {merged_out}")
    print(f"WROTE: {corr_out}")
    print(f"WROTE: {args.out_dir / 'absolute_dirichlet_vs_mae_raw.png'}")
    print(f"WROTE: {args.out_dir / 'absolute_dirichlet_vs_mae_std.png'}")


if __name__ == "__main__":
    main()
