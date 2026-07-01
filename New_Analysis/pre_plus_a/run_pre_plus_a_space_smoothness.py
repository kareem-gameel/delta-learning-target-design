#!/usr/bin/env python3
"""Graph Dirichlet smoothness in PRE+A feature space.

This uses the concatenated PRE typed pair-radial features and compact angular
environment histograms, then correlates target smoothness against validation
performance for GemNet, SchNet, BoB, and PRE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import diags, hstack, load_npz
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
BOB_HELPER_DIR = ROOT / "New_Analysis/bob_predict_targets"
sys.path.insert(0, str(BOB_HELPER_DIR))

from run_bob_space_smoothness import (  # noqa: E402
    TARGETS,
    build_edges,
    color_for,
    dirichlet_metrics,
    label_for,
    load_frame,
    sparse_std,
)


DEFAULT_PRE_CACHE = ROOT / "New_Analysis/features/PRE/trillium_cache"
DEFAULT_ANGULAR_CACHE = ROOT / "New_Analysis/features/PRE_A_debug_full"
DEFAULT_SPLIT_DIR = ROOT / "trillium_project/split_ids"
DEFAULT_MODEL_METRICS = ROOT / "New_Analysis/model_target_comparison/outputs/model_target_val_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "New_Analysis/pre_plus_a/outputs_smoothness"

MODEL_ORDER = ["GemNet", "SchNet", "BoB", "PRE"]
MODEL_COLORS = {
    "GemNet": "#355C9A",
    "SchNet": "#6C5AB5",
    "BoB": "#111111",
    "PRE": "#2E8B57",
}


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_pre_features(cache_dir: Path) -> tuple[object, pd.DataFrame]:
    feature_path = cache_dir / "tmqm_pre_features.npz"
    ids_path = cache_dir / "tmqm_pre_targets_aligned.csv"
    require(feature_path, "PRE feature matrix")
    require(ids_path, "PRE mol_id table")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])
    ids["mol_id"] = ids["mol_id"].astype(str)
    if len(ids) != x.shape[0]:
        raise ValueError(f"PRE row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    ids["_pre_row"] = np.arange(len(ids))
    return x, ids


def load_angular_features(cache_dir: Path) -> tuple[object, pd.DataFrame]:
    feature_path = cache_dir / "tmqm_angular_features.npz"
    ids_path = cache_dir / "tmqm_angular_mol_ids.csv"
    require(feature_path, "PRE+A angular feature matrix")
    require(ids_path, "PRE+A angular mol_id table")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])
    ids["mol_id"] = ids["mol_id"].astype(str)
    if len(ids) != x.shape[0]:
        raise ValueError(f"Angular row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    ids["_angular_row"] = np.arange(len(ids))
    return x, ids


def load_aligned_pre_plus_a_features(
    pre_cache: Path,
    angular_cache: Path,
    frame: pd.DataFrame,
):
    x_pre_all, pre_ids = load_pre_features(pre_cache)
    x_ang_all, ang_ids = load_angular_features(angular_cache)

    aligned = (
        frame.merge(pre_ids, on="mol_id", how="inner", validate="one_to_one")
        .merge(ang_ids, on="mol_id", how="inner", validate="one_to_one")
        .reset_index(drop=True)
    )
    x_pre = x_pre_all[aligned["_pre_row"].to_numpy(int)].tocsr()
    x_ang = x_ang_all[aligned["_angular_row"].to_numpy(int)].tocsr()
    x = hstack([x_pre, x_ang], format="csr")
    aligned = aligned.drop(columns=["_pre_row", "_angular_row"])
    return x, aligned, {"pre_dims": int(x_pre.shape[1]), "angular_dims": int(x_ang.shape[1])}


def scale_features(x, scaling: str, eps: float):
    if scaling == "raw":
        return x.tocsr(), x.shape[1], 0
    if scaling != "std":
        raise ValueError(f"Unknown scaling: {scaling}")
    std = sparse_std(x)
    keep = std > eps
    scaled = x[:, keep] @ diags(1.0 / std[keep], offsets=0, format="csr")
    return scaled.tocsr(), int(keep.sum()), int((~keep).sum())


def attach_model_metrics(summary: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    require(metrics_path, "model-target validation metrics")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["target_id"].isin(summary["target_id"].unique())].copy()
    keep_cols = ["model", "target_id", "mae", "mae_over_iqr"]
    missing = sorted(set(keep_cols) - set(metrics.columns))
    if missing:
        raise ValueError(f"{metrics_path} missing columns: {missing}")
    metrics = metrics[keep_cols].rename(columns={"mae": "model_val_mae", "mae_over_iqr": "model_val_mae_over_iqr"})
    wide = metrics.pivot_table(index="target_id", columns="model", values=["model_val_mae", "model_val_mae_over_iqr"], aggfunc="first")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    return summary.merge(wide, on="target_id", how="left", validate="many_to_one")


def correlation_rows(df: pd.DataFrame, x_cols: list[str], model_names: list[str], k: int, scaling: str) -> pd.DataFrame:
    sub = df[(df["k"] == k) & (df["feature_scaling"] == scaling)].copy()
    rows = []
    for x_col in x_cols:
        for model in model_names:
            for y_col in [f"model_val_mae_{model}", f"model_val_mae_over_iqr_{model}"]:
                if y_col not in sub.columns:
                    continue
                keep = sub[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(keep) < 3:
                    pearson = np.nan
                    spearman = np.nan
                else:
                    pearson = float(pearsonr(keep[x_col], keep[y_col]).statistic)
                    spearman = float(spearmanr(keep[x_col], keep[y_col]).statistic)
                rows.append(
                    {
                        "feature_space": "PRE+A",
                        "feature_scaling": scaling,
                        "k": k,
                        "x_metric": x_col,
                        "model": model,
                        "y_metric": y_col.replace(f"_{model}", ""),
                        "n_targets": int(len(keep)),
                        "pearson_r": pearson,
                        "spearman_r": spearman,
                    }
                )
    return pd.DataFrame(rows)


def plot_model_grid(summary: pd.DataFrame, out_dir: Path, k: int, scaling: str, y_suffix: str, y_label: str) -> None:
    sub = summary[(summary["k"] == k) & (summary["feature_scaling"] == scaling)].copy()
    x_col = "edge_avg_norm_by_population_var"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), dpi=220)
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        y_col = f"model_val_{y_suffix}_{model}"
        if y_col not in sub.columns:
            ax.axis("off")
            continue
        keep = sub[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        r = float(pearsonr(keep[x_col], keep[y_col]).statistic) if len(keep) >= 3 else np.nan
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=85,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                label_for(row["target_id"]),
                (row[x_col], row[y_col]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=10.5,
            )
        ax.set_title(f"{model}; r={r:.2f}", fontsize=13)
        ax.set_xlabel("PRE+A Dirichlet smoothness\n(edge mean squared jump / target variance)", fontsize=10.5)
        ax.set_ylabel(y_label, fontsize=10.5)
        ax.grid(True, color="#D8DEE9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = out_dir / f"pre_plus_a_smoothness_vs_models_{y_suffix}_k{k}_{scaling}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"WROTE: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--k-values", default="10", help="Comma-separated k values")
    parser.add_argument("--scalings", default="std,raw", help="Comma-separated: std,raw")
    parser.add_argument("--std-eps", type=float, default=1e-12)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pre-cache", type=Path, default=DEFAULT_PRE_CACHE)
    parser.add_argument("--angular-cache", type=Path, default=DEFAULT_ANGULAR_CACHE)
    parser.add_argument("--model-metrics", type=Path, default=DEFAULT_MODEL_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.split_dir, args.split)
    x_raw, frame, dims = load_aligned_pre_plus_a_features(args.pre_cache, args.angular_cache, frame)
    if args.max_rows and len(frame) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(frame), size=args.max_rows, replace=False))
        frame = frame.iloc[idx].reset_index(drop=True)
        x_raw = x_raw[idx].tocsr()
    print(
        f"Using n_molecules={len(frame)} split={args.split}; "
        f"PRE dims={dims['pre_dims']} angular dims={dims['angular_dims']} PRE+A dims={x_raw.shape[1]}"
    )

    rows = []
    k_values = [int(v) for v in args.k_values.split(",") if v.strip()]
    scalings = [v.strip() for v in args.scalings.split(",") if v.strip()]
    for scaling in scalings:
        x, n_retained, n_dropped = scale_features(x_raw, scaling, args.std_eps)
        print(f"scaling={scaling}: retained={n_retained}, dropped={n_dropped}")
        for k in k_values:
            print(f"Building PRE+A-space Euclidean kNN graph k={k}, scaling={scaling}")
            edges = build_edges(x, k)
            print(f"k={k}, scaling={scaling}: unique undirected edges={len(edges)}")
            for spec in TARGETS:
                y = frame[spec["target_id"]].to_numpy(float)
                rows.append(
                    {
                        "target_id": spec["target_id"],
                        "target_label": spec["label"],
                        "short_label": spec["short_label"],
                        "split": args.split,
                        "feature_space": "PRE+A typed pair-radial plus angular",
                        "feature_scaling": scaling,
                        "k": k,
                        "n_molecules": int(len(frame)),
                        "n_features_total": int(x_raw.shape[1]),
                        "n_features_retained": int(n_retained),
                        "n_features_dropped": int(n_dropped),
                        "pre_dims": int(dims["pre_dims"]),
                        "angular_dims": int(dims["angular_dims"]),
                        **dirichlet_metrics(y, edges),
                    }
                )

    summary = pd.DataFrame(rows)
    summary = attach_model_metrics(summary, args.model_metrics)
    out_csv = args.out_dir / f"pre_plus_a_space_dirichlet_smoothness_{args.split}.csv"
    summary.to_csv(out_csv, index=False)
    print(f"WROTE: {out_csv}")

    corr = pd.concat(
        [
            correlation_rows(
                summary,
                x_cols=["edge_rmse_over_iqr", "edge_avg_norm_by_population_var", "edge_avg_norm_by_random_pair"],
                model_names=MODEL_ORDER,
                k=k,
                scaling=scaling,
            )
            for scaling in scalings
            for k in k_values
        ],
        ignore_index=True,
    )
    corr_csv = args.out_dir / f"pre_plus_a_space_smoothness_correlations_{args.split}.csv"
    corr.to_csv(corr_csv, index=False)
    print(f"WROTE: {corr_csv}")

    for scaling in scalings:
        for k in k_values:
            plot_model_grid(summary, args.out_dir, k=k, scaling=scaling, y_suffix="mae", y_label="Val MAE (Ha)")
            plot_model_grid(
                summary,
                args.out_dir,
                k=k,
                scaling=scaling,
                y_suffix="mae_over_iqr",
                y_label="Val MAE / target IQR",
            )

    view_cols = [
        "target_id",
        "feature_scaling",
        "k",
        "edge_avg_norm_by_population_var",
        "edge_rmse_over_iqr",
        "model_val_mae_GemNet",
        "model_val_mae_over_iqr_GemNet",
        "model_val_mae_SchNet",
        "model_val_mae_over_iqr_SchNet",
        "model_val_mae_BoB",
        "model_val_mae_over_iqr_BoB",
        "model_val_mae_PRE",
        "model_val_mae_over_iqr_PRE",
    ]
    print(summary[[c for c in view_cols if c in summary.columns]].to_string(index=False))
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
