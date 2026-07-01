#!/usr/bin/env python3
"""Pool exported atomic GemNet embeddings into molecular embeddings."""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Encountered empty or non-finite embedding vector")
    return arr


def read_embedding_shards(atom_embedding_dir: Path, file_glob: str, max_files: int | None) -> pd.DataFrame:
    paths = sorted(atom_embedding_dir.glob(file_glob))
    if max_files is not None:
        paths = paths[: max_files]
    if not paths:
        raise FileNotFoundError(f"No parquet files matched {atom_embedding_dir / file_glob}")
    frames = []
    for path in paths:
        log(f"READ: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def pool_matrix(atom_embeddings: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return atom_embeddings.mean(axis=0)
    if pooling == "sum":
        return atom_embeddings.sum(axis=0)
    if pooling == "mean_std":
        return np.concatenate([atom_embeddings.mean(axis=0), atom_embeddings.std(axis=0, ddof=0)])
    if pooling == "mean_std_sum":
        return np.concatenate(
            [
                atom_embeddings.mean(axis=0),
                atom_embeddings.std(axis=0, ddof=0),
                atom_embeddings.sum(axis=0),
            ]
        )
    if pooling == "mean_std_min_max":
        return np.concatenate(
            [
                atom_embeddings.mean(axis=0),
                atom_embeddings.std(axis=0, ddof=0),
                atom_embeddings.min(axis=0),
                atom_embeddings.max(axis=0),
            ]
        )
    raise ValueError(f"Unknown pooling: {pooling}")


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    available = set(columns)
    return [col for col in candidates if col in available]


def pool_dataframe(df: pd.DataFrame, embedding_column: str, pooling: str) -> pd.DataFrame:
    required = {"mol_id", embedding_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Atomic embedding table is missing required columns: {missing}")

    df = df.copy()
    df["mol_id"] = df["mol_id"].astype(str)
    if "sid" in df.columns:
        group_cols = ["mol_id", "sid"]
    else:
        group_cols = ["mol_id"]

    metadata_cols = first_present(
        df.columns,
        ["formula", "target_value", "atom_count", "n_atoms", "split_name", "model_tag"],
    )
    rows = []
    for group_key, group in df.groupby(group_cols, sort=False, dropna=False):
        vectors = np.vstack(group[embedding_column].map(parse_vector).to_list()).astype(np.float32, copy=False)
        pooled = pool_matrix(vectors, pooling).astype(np.float32, copy=False)
        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key))
        else:
            row = {group_cols[0]: group_key}
        row["n_atoms_pooled"] = int(vectors.shape[0])
        row["embedding_dim"] = int(pooled.size)
        row["embedding"] = pooled.tolist()
        for col in metadata_cols:
            row[col] = group[col].iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["mol_id"] = out["mol_id"].astype(str)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-embedding-dir", type=Path, required=True)
    parser.add_argument("--file-glob", default="*.parquet")
    parser.add_argument("--embedding-column", default="atom_embedding")
    parser.add_argument("--poolings", default="mean,mean_std,sum")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_embedding_shards(args.atom_embedding_dir, args.file_glob, args.max_files)
    log(f"Loaded atomic embeddings: rows={len(df)} columns={list(df.columns)}")
    poolings = [part.strip() for part in args.poolings.split(",") if part.strip()]
    summary_rows = []
    for pooling in poolings:
        log(f"Pooling atomic embeddings: pooling={pooling}")
        pooled = pool_dataframe(df, args.embedding_column, pooling)
        out_path = args.out_dir / f"molecular_embeddings_{pooling}.parquet"
        pooled.to_parquet(out_path, index=False)
        log(f"WROTE: {out_path}")
        summary_rows.append(
            {
                "pooling": pooling,
                "n_molecules": int(len(pooled)),
                "embedding_dim": int(pooled["embedding_dim"].iloc[0]) if len(pooled) else 0,
                "out_path": str(out_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "molecular_embedding_pooling_summary.csv"
    summary.to_csv(summary_path, index=False)
    meta = {
        "atom_embedding_dir": str(args.atom_embedding_dir),
        "file_glob": args.file_glob,
        "embedding_column": args.embedding_column,
        "n_atomic_rows": int(len(df)),
        "poolings": poolings,
        "outputs": summary_rows,
    }
    meta_path = args.out_dir / "molecular_embedding_pooling_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log(f"WROTE: {summary_path}")
    log(f"WROTE: {meta_path}")


if __name__ == "__main__":
    main()

