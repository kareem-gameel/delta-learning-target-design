#!/usr/bin/env python3
"""Compute target roughness in a learned GemNet molecular embedding space."""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from embedding_targets import TargetSpec, load_target_values, selected_targets


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, list) or isinstance(value, tuple):
        arr = np.asarray(value)
    elif isinstance(value, str):
        arr = np.asarray(ast.literal_eval(value))
    else:
        arr = np.asarray(value)
    arr = arr.astype(np.float64, copy=False).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Encountered empty or non-finite molecular embedding")
    return arr


def load_embedding_frame(path: Path, embedding_column: str, max_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing molecular embedding table: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = sorted({"mol_id", embedding_column} - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.dropna(subset=["mol_id", embedding_column]).drop_duplicates("mol_id", keep="first").copy()
    df["mol_id"] = df["mol_id"].astype(str)
    if max_rows is not None:
        df = df.sort_values("mol_id").head(max_rows).reset_index(drop=True)
    return df.reset_index(drop=True)


def apply_mol_id_filter(embedding_df: pd.DataFrame, mol_id_filter: Path | None) -> pd.DataFrame:
    if mol_id_filter is None:
        return embedding_df
    if not mol_id_filter.exists():
        raise FileNotFoundError(f"Missing mol_id filter: {mol_id_filter}")
    filter_df = pd.read_csv(mol_id_filter, usecols=["mol_id"]).dropna().drop_duplicates("mol_id", keep="first")
    filter_df["mol_id"] = filter_df["mol_id"].astype(str)
    before = len(embedding_df)
    filtered = filter_df.merge(embedding_df, on="mol_id", how="inner", validate="one_to_one")
    log(f"Applied mol_id filter {mol_id_filter}: retained {len(filtered)}/{before} embedding rows")
    if filtered.empty:
        raise ValueError(f"mol_id filter retained zero rows: {mol_id_filter}")
    return filtered.reset_index(drop=True)


def load_target_panel(embedding_df: pd.DataFrame, targets: list[TargetSpec]) -> pd.DataFrame:
    panel = embedding_df[["mol_id"]].copy()
    for target in targets:
        values = load_target_values(target)
        before = len(panel)
        panel = panel.merge(values, on="mol_id", how="inner", validate="one_to_one")
        log(f"Aligned target {target.target_id}: retained {len(panel)}/{before} embedding rows")
    return panel


def embedding_matrix(embedding_df: pd.DataFrame, panel: pd.DataFrame, embedding_column: str) -> np.ndarray:
    aligned = panel[["mol_id"]].merge(
        embedding_df[["mol_id", embedding_column]], on="mol_id", how="left", validate="one_to_one"
    )
    if aligned[embedding_column].isna().any():
        raise ValueError("Internal alignment error: missing embeddings after target join")
    x = np.vstack(aligned[embedding_column].map(parse_vector).to_list())
    if not np.isfinite(x).all():
        raise ValueError("Molecular embedding matrix contains non-finite values")
    return x


def scale_features(x: np.ndarray, scaling: str, eps: float) -> tuple[np.ndarray, int, int]:
    if scaling == "raw":
        return x, int(x.shape[1]), 0
    if scaling != "std":
        raise ValueError(f"Unknown scaling: {scaling}")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    keep = std > eps
    x_scaled = (x[:, keep] - mean[keep]) / std[keep]
    return x_scaled, int(keep.sum()), int((~keep).sum())


def unique_undirected_edges(neighbor_idx: np.ndarray) -> np.ndarray:
    src = np.repeat(np.arange(neighbor_idx.shape[0]), neighbor_idx.shape[1])
    dst = neighbor_idx.reshape(-1)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    edges = np.column_stack([lo, hi])
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def build_edges(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) <= k:
        raise ValueError(f"Need more rows than k; got n={len(x)} k={k}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    return unique_undirected_edges(ind[:, 1:])


def dirichlet_metrics(y: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    diff = y[edges[:, 0]] - y[edges[:, 1]]
    e_num = float(np.sum(diff**2))
    n_edges = int(edges.shape[0])
    edge_mean_sq = e_num / n_edges
    centered = y - y.mean()
    e_den = float(np.sum(centered**2))
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
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=0)),
        "target_iqr": iqr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--embedding-column", default="embedding")
    parser.add_argument("--embedding-model-id", required=True)
    parser.add_argument("--embedding-model-label", default=None)
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--scalings", default="std,raw")
    parser.add_argument("--std-eps", type=float, default=1e-12)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--mol-id-filter", type=Path, default=None)
    parser.add_argument("--keep-missing-optional", action="store_true")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-panel", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    targets = selected_targets(args.targets, skip_missing_optional=not args.keep_missing_optional)
    embedding_df = load_embedding_frame(args.embedding_path, args.embedding_column, args.max_rows)
    embedding_df = apply_mol_id_filter(embedding_df, args.mol_id_filter)
    panel = load_target_panel(embedding_df, targets)
    x = embedding_matrix(embedding_df, panel, args.embedding_column)
    pooling = args.pooling or args.embedding_path.stem.replace("molecular_embeddings_", "")
    label = args.embedding_model_label or args.embedding_model_id
    scalings = [part.strip().lower() for part in args.scalings.split(",") if part.strip()]

    log(
        f"Loaded molecular embeddings: model={args.embedding_model_id} pooling={pooling} "
        f"rows={x.shape[0]} dims={x.shape[1]} targets={len(targets)}"
    )

    rows = []
    for scaling in scalings:
        x_scaled, n_retained, n_dropped = scale_features(x, scaling, args.std_eps)
        log(f"Building kNN graph: scaling={scaling} rows={x_scaled.shape[0]} dims={x_scaled.shape[1]} k={args.k}")
        edges = build_edges(x_scaled, args.k)
        for target in targets:
            y = panel[target.target_id].to_numpy(float)
            rows.append(
                {
                    "embedding_model_id": args.embedding_model_id,
                    "embedding_model_label": label,
                    "pooling": pooling,
                    "feature_scaling": scaling,
                    "k": int(args.k),
                    "target_id": target.target_id,
                    "target_label": target.target_label,
                    "short_label": target.short_label,
                    "target_family": target.target_family,
                    "n_molecules": int(len(y)),
                    "n_embedding_dims_total": int(x.shape[1]),
                    "n_embedding_dims_retained": n_retained,
                    "n_embedding_dims_dropped": n_dropped,
                    **dirichlet_metrics(y, edges),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    log(f"WROTE: {args.out_csv}")

    meta_path = args.out_csv.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "embedding_path": str(args.embedding_path),
                "embedding_column": args.embedding_column,
                "embedding_model_id": args.embedding_model_id,
                "embedding_model_label": label,
                "pooling": pooling,
                "targets": [target.target_id for target in targets],
                "mol_id_filter": str(args.mol_id_filter) if args.mol_id_filter is not None else None,
                "k": args.k,
                "scalings": scalings,
                "n_molecules_common": int(len(panel)),
            },
            indent=2,
        )
        + "\n"
    )
    log(f"WROTE: {meta_path}")

    if args.out_panel is not None:
        args.out_panel.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.out_panel, index=False)
        log(f"WROTE: {args.out_panel}")


if __name__ == "__main__":
    main()
