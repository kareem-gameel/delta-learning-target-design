#!/usr/bin/env python3
"""Run learned-embedding roughness for several pooled GemNet embedding tables."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from compute_embedding_roughness import (
    apply_mol_id_filter,
    build_edges,
    dirichlet_metrics,
    embedding_matrix,
    load_embedding_frame,
    load_target_panel,
    scale_features,
)
from embedding_targets import selected_targets


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "CSV with columns embedding_model_id, embedding_model_label, embedding_path. "
            "Optional columns: pooling, embedding_column."
        ),
    )
    parser.add_argument("--targets", default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--scalings", default="std,raw")
    parser.add_argument("--std-eps", type=float, default=1e-12)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--mol-id-filter", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required = {"embedding_model_id", "embedding_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{args.manifest} missing required columns: {missing}")

    targets = selected_targets(args.targets, skip_missing_optional=True)
    scalings = [part.strip().lower() for part in args.scalings.split(",") if part.strip()]
    rows = []
    for item in manifest.to_dict(orient="records"):
        embedding_path = Path(str(item["embedding_path"]))
        embedding_model_id = str(item["embedding_model_id"])
        embedding_model_label = str(item.get("embedding_model_label") or embedding_model_id)
        embedding_column = str(item.get("embedding_column") or "embedding")
        pooling = str(item.get("pooling") or embedding_path.stem.replace("molecular_embeddings_", ""))

        log(f"RUN: {embedding_model_id} pooling={pooling} path={embedding_path}")
        embedding_df = load_embedding_frame(embedding_path, embedding_column, args.max_rows)
        embedding_df = apply_mol_id_filter(embedding_df, args.mol_id_filter)
        panel = load_target_panel(embedding_df, targets)
        x = embedding_matrix(embedding_df, panel, embedding_column)
        for scaling in scalings:
            x_scaled, n_retained, n_dropped = scale_features(x, scaling, args.std_eps)
            log(
                f"Building kNN graph: model={embedding_model_id} scaling={scaling} "
                f"rows={x_scaled.shape[0]} dims={x_scaled.shape[1]} k={args.k}"
            )
            edges = build_edges(x_scaled, args.k)
            for target in targets:
                y = panel[target.target_id].to_numpy(float)
                rows.append(
                    {
                        "embedding_model_id": embedding_model_id,
                        "embedding_model_label": embedding_model_label,
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


if __name__ == "__main__":
    main()
