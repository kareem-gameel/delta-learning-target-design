#!/usr/bin/env python3
"""Compute graph Dirichlet smoothness for the expanded target panel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

from expanded_panel_common import (
    DEFAULT_PRE_A_CACHE,
    DEFAULT_PRE_CACHE,
    DEFAULT_SPLIT_DIR,
    apply_max_rows,
    load_feature_space,
    load_split_ids,
    load_target_frame,
    selected_targets,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_target_panel(targets, split_df: pd.DataFrame, split: str, max_rows_per_split: int | None) -> pd.DataFrame:
    if split != "all":
        split_df = split_df[split_df["split"].eq(split)].copy()
    frame = split_df.copy()
    for target in targets:
        part = load_target_frame(target, split_df)[["mol_id", "target"]].rename(columns={"target": target.target_id})
        frame = frame.merge(part, on="mol_id", how="inner", validate="one_to_one")
    return apply_max_rows(frame, max_rows_per_split)


def align_features(x_all, ids: pd.Series, frame: pd.DataFrame):
    rows = pd.DataFrame({"mol_id": ids, "_row": np.arange(len(ids), dtype=int)})
    aligned = frame.merge(rows, on="mol_id", how="inner", validate="one_to_one").reset_index(drop=True)
    return x_all[aligned["_row"].to_numpy(int)].tocsr(), aligned.drop(columns=["_row"])


def sparse_std(x) -> np.ndarray:
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


def scale_features(x, scaling: str, eps: float):
    if scaling == "raw":
        return x.tocsr(), int(x.shape[1]), 0
    if scaling != "std":
        raise ValueError(f"Unknown scaling: {scaling}")
    std = sparse_std(x)
    keep = std > eps
    x_scaled = x[:, keep].tocsr() @ diags(1.0 / std[keep], offsets=0, format="csr")
    return x_scaled.tocsr(), int(keep.sum()), int((~keep).sum())


def unique_undirected_edges(neighbor_idx: np.ndarray) -> np.ndarray:
    src = np.repeat(np.arange(neighbor_idx.shape[0]), neighbor_idx.shape[1])
    dst = neighbor_idx.reshape(-1)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    edges = np.column_stack([lo, hi])
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def build_edges(x, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    return unique_undirected_edges(ind[:, 1:])


def dirichlet_metrics(y: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    diff = y[edges[:, 0]] - y[edges[:, 1]]
    e_num = float(np.sum(diff**2))
    e_den = float(np.sum((y - y.mean()) ** 2))
    n_edges = int(edges.shape[0])
    edge_mean_sq = e_num / n_edges
    var_pop = e_den / len(y)
    iqr = float(np.subtract(*np.percentile(y, [75, 25])))
    return {
        "n_edges": n_edges,
        "E_num": e_num,
        "E_den": e_den,
        "E_norm_sum": e_num / e_den if e_den > 0 else np.nan,
        "edge_mean_squared_diff": edge_mean_sq,
        "edge_avg_norm_by_population_var": edge_mean_sq / var_pop if var_pop > 0 else np.nan,
        "edge_avg_norm_by_random_pair": edge_mean_sq / (2.0 * var_pop) if var_pop > 0 else np.nan,
        "edge_rmse": float(np.sqrt(edge_mean_sq)),
        "edge_rmse_over_iqr": float(np.sqrt(edge_mean_sq) / iqr) if iqr > 0 else np.nan,
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y, ddof=0)),
        "target_iqr": iqr,
    }


def correlation_rows(summary: pd.DataFrame, metrics_path: Path | None) -> pd.DataFrame:
    if metrics_path is None or not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    val = metrics[metrics["split"].eq("val")].copy()
    join_cols = ["target_id", "mae", "mae_over_iqr", "r2", "kendall_tau"]
    missing = sorted(set(join_cols) - set(val.columns))
    if missing:
        raise ValueError(f"{metrics_path} missing columns: {missing}")
    merged = summary.merge(val[join_cols], on="target_id", how="inner", validate="many_to_one")
    rows = []
    for (feature_space, scaling, k), group in merged.groupby(["feature_space", "feature_scaling", "k"]):
        for x_metric in ["edge_avg_norm_by_population_var", "edge_rmse_over_iqr"]:
            for y_metric in ["mae", "mae_over_iqr", "r2", "kendall_tau"]:
                keep = group[[x_metric, y_metric]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(keep) < 3:
                    pearson = np.nan
                    spearman = np.nan
                else:
                    pearson = float(pearsonr(keep[x_metric], keep[y_metric]).statistic)
                    spearman = float(spearmanr(keep[x_metric], keep[y_metric]).statistic)
                rows.append(
                    {
                        "feature_space": feature_space,
                        "feature_scaling": scaling,
                        "k": int(k),
                        "model_metrics_file": str(metrics_path),
                        "x_metric": x_metric,
                        "y_metric": y_metric,
                        "n_targets": int(len(keep)),
                        "pearson_r": pearson,
                        "spearman_r": spearman,
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-space", required=True, choices=["bob", "pre", "pre_plus_a"])
    parser.add_argument("--targets", default="all")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pre-cache", type=Path, default=DEFAULT_PRE_CACHE)
    parser.add_argument("--angular-cache", type=Path, default=DEFAULT_PRE_A_CACHE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--scalings", default="raw,std", help="Comma-separated: raw,std")
    parser.add_argument("--std-eps", type=float, default=1e-12)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--model-metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("New_Analysis/expanded_target_panel/outputs/smoothness"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = selected_targets(args.targets, require_existing=True)
    split_df = load_split_ids(args.split_dir)
    frame = load_target_panel(targets, split_df, args.split, args.max_rows_per_split)
    x_all, feature_ids, feature_meta = load_feature_space(args.feature_space, args.pre_cache, args.angular_cache)
    x, aligned = align_features(x_all, feature_ids, frame)
    scalings = [part.strip().lower() for part in args.scalings.split(",") if part.strip()]
    rows = []

    log(
        f"Loaded {feature_meta['feature_space']} features: rows={x.shape[0]} dims={x.shape[1]} "
        f"split={args.split}"
    )
    for scaling in scalings:
        log(f"Scaling features: scaling={scaling}")
        x_scaled, n_retained, n_dropped = scale_features(x, scaling, args.std_eps)
        log(f"Building kNN graph: k={args.k} rows={x_scaled.shape[0]} dims={x_scaled.shape[1]}")
        edges = build_edges(x_scaled, args.k)
        for target in targets:
            y = aligned[target.target_id].to_numpy(float)
            rows.append(
                {
                    "feature_space": feature_meta["feature_space"],
                    "feature_scaling": scaling,
                    "k": int(args.k),
                    "target_id": target.target_id,
                    "target_label": target.target_label,
                    "short_label": target.short_label,
                    "target_family": target.target_family,
                    "split": args.split,
                    "n_molecules": int(len(y)),
                    "n_features_total": int(x.shape[1]),
                    "n_features_retained": n_retained,
                    "n_features_dropped": n_dropped,
                    **dirichlet_metrics(y, edges),
                }
            )

    summary = pd.DataFrame(rows)
    out_csv = args.out_dir / f"{feature_meta['feature_space'].lower().replace('+', '_plus_')}_space_dirichlet_{args.split}.csv"
    summary.to_csv(out_csv, index=False)
    log(f"WROTE: {out_csv}")

    corr = correlation_rows(summary, args.model_metrics)
    if not corr.empty:
        corr_path = args.out_dir / f"{feature_meta['feature_space'].lower().replace('+', '_plus_')}_space_correlations_{args.split}.csv"
        corr.to_csv(corr_path, index=False)
        log(f"WROTE: {corr_path}")


if __name__ == "__main__":
    main()
