#!/usr/bin/env python3
"""Shared utilities for the expanded ref,BoE target panel."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "New_Analysis/datasets"
DEFAULT_SPLIT_DIR = ROOT / "trillium_project/split_ids"
DEFAULT_TARGET_REGISTRY = ROOT / "New_Analysis/expanded_target_panel/target_registry.csv"

DEFAULT_BOB_MATRIX = ROOT / "trillium_project/baseline_validation/bob/tmqm_bob_feature_matrix.npz"
DEFAULT_BOB_IDS = ROOT / "trillium_project/baseline_validation/bob/tmqm_bob_feature_mol_ids.csv"
DEFAULT_PRE_CACHE = ROOT / "New_Analysis/features/PRE/trillium_cache"
DEFAULT_PRE_A_CACHE = ROOT / "New_Analysis/features/PRE_A_debug_full"
DEFAULT_PRE_A_OUTPUT = ROOT / "New_Analysis/pre_plus_a/outputs_high_alpha_cv"


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    target_label: str
    short_label: str
    csv_path: Path
    column: str
    target_family: str
    train_gnn: bool = True


TARGETS: list[TargetSpec] = [
    TargetSpec(
        "dft",
        "DFT_ref,BoE",
        "DFT",
        DATASET_DIR / "tmQM__DFT_ref-BoE.csv",
        "DFT_ref-BoE",
        "reference",
    ),
    TargetSpec(
        "xtb",
        "xTB_ref,BoE",
        "xTB",
        DATASET_DIR / "tmQM__xTB_ref-BoE.csv",
        "xTB_ref-BoE",
        "baseline_target",
    ),
    TargetSpec(
        "deltaxtb",
        "Delta^xTB_ref,BoE",
        "Delta^xTB",
        DATASET_DIR / "tmQM__Delta_xTB-ref-BoE.csv",
        "Delta_xTB-ref-BoE",
        "residual_target",
    ),
    TargetSpec(
        "deltabob",
        "Delta^BoB_ref,BoE",
        "Delta^BoB",
        DATASET_DIR / "tmQM__Delta_BoB-ref-BoE.csv",
        "E_resid_after_BoE_BoB_tmQM",
        "residual_target",
    ),
    TargetSpec(
        "deltapre",
        "Delta^PRE_ref,BoE",
        "Delta^PRE",
        DATASET_DIR / "tmQM__Delta_PRE-ref-BoE.csv",
        "E_resid_after_BoE_PRE_tmQM",
        "residual_target",
    ),
    TargetSpec(
        "bob",
        "BoB_ref,BoE",
        "BoB",
        DATASET_DIR / "tmQM__BoB-ref-BoE.csv",
        "BoB_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "pre",
        "PRE_ref,BoE",
        "PRE",
        DATASET_DIR / "tmQM__PRE-ref-BoE.csv",
        "PRE_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "pre_plus_a",
        "PRE+A_ref,BoE",
        "PRE+A",
        DATASET_DIR / "tmQM__PRE_plus_A-ref-BoE.csv",
        "PRE_plus_A_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "deltapre_plus_a",
        "Delta^PRE+A_ref,BoE",
        "Delta^PRE+A",
        DATASET_DIR / "tmQM__Delta_PRE_plus_A-ref-BoE.csv",
        "Delta_PRE_plus_A-ref-BoE",
        "residual_target",
    ),
]

TARGET_BY_ID = {target.target_id: target for target in TARGETS}


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_target_ids(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return [target.target_id for target in TARGETS]
    ids = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(ids) - set(TARGET_BY_ID))
    if unknown:
        raise ValueError(f"Unknown target IDs: {unknown}. Choices: {sorted(TARGET_BY_ID)}")
    return ids


def selected_targets(value: str, *, require_existing: bool = True) -> list[TargetSpec]:
    out = [TARGET_BY_ID[target_id] for target_id in parse_target_ids(value)]
    if require_existing:
        missing = [str(target.csv_path) for target in out if not target.csv_path.exists()]
        if missing:
            raise FileNotFoundError("Missing target CSV(s):\n" + "\n".join(missing))
    return out


def write_registry(path: Path = DEFAULT_TARGET_REGISTRY) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        row = asdict(target)
        row["csv_path"] = str(target.csv_path)
        row["exists"] = target.csv_path.exists()
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    return df


def load_split_ids(split_dir: Path = DEFAULT_SPLIT_DIR) -> pd.DataFrame:
    parts = []
    for split in ["train", "val", "test"]:
        path = split_dir / f"tmqm_{split}_ids.csv"
        require_file(path, f"{split} split IDs")
        df = pd.read_csv(path, usecols=["mol_id"]).copy()
        df["mol_id"] = df["mol_id"].astype(str)
        df["split"] = split
        parts.append(df)
    split_df = pd.concat(parts, ignore_index=True)
    dup = split_df.loc[split_df["mol_id"].duplicated(keep=False), "mol_id"].unique()
    if len(dup):
        raise ValueError(f"Split IDs are not disjoint; duplicate examples: {dup[:10].tolist()}")
    return split_df


def load_target_frame(target: TargetSpec, split_df: pd.DataFrame) -> pd.DataFrame:
    require_file(target.csv_path, target.target_label)
    df = pd.read_csv(target.csv_path)
    missing = sorted({"mol_id", target.column} - set(df.columns))
    if missing:
        raise ValueError(f"{target.csv_path} missing required columns for {target.target_id}: {missing}")
    df = df[["mol_id", target.column]].dropna().drop_duplicates("mol_id", keep="first").copy()
    df["mol_id"] = df["mol_id"].astype(str)
    df = df.rename(columns={target.column: "target"})
    return df.merge(split_df, on="mol_id", how="inner", validate="one_to_one").reset_index(drop=True)


def apply_max_rows(df: pd.DataFrame, max_rows_per_split: int | None) -> pd.DataFrame:
    if max_rows_per_split is None:
        return df
    return (
        df.sort_values(["split", "mol_id"])
        .groupby("split", group_keys=False)
        .head(int(max_rows_per_split))
        .reset_index(drop=True)
    )


def load_bob_features(matrix_path: Path = DEFAULT_BOB_MATRIX, ids_path: Path = DEFAULT_BOB_IDS):
    require_file(matrix_path, "BoB feature matrix")
    require_file(ids_path, "BoB mol_id table")
    x = load_npz(matrix_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"BoB row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    return x, ids


def load_pre_features(cache_dir: Path = DEFAULT_PRE_CACHE):
    feature_path = cache_dir / "tmqm_pre_features.npz"
    ids_path = cache_dir / "tmqm_pre_targets_aligned.csv"
    require_file(feature_path, "PRE feature matrix")
    require_file(ids_path, "PRE mol_id table")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"PRE row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    return x, ids


def load_angular_features(cache_dir: Path = DEFAULT_PRE_A_CACHE):
    feature_path = cache_dir / "tmqm_angular_features.npz"
    ids_path = cache_dir / "tmqm_angular_mol_ids.csv"
    require_file(feature_path, "PRE+A angular feature matrix")
    require_file(ids_path, "PRE+A angular mol_id table")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"Angular row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    return x, ids


def align_feature_block(x: csr_matrix, ids: pd.Series, frame: pd.DataFrame, row_name: str):
    row_df = pd.DataFrame({"mol_id": ids, row_name: np.arange(len(ids), dtype=int)})
    aligned = frame.merge(row_df, on="mol_id", how="inner", validate="one_to_one").reset_index(drop=True)
    rows = aligned[row_name].to_numpy(int)
    return x[rows].tocsr(), aligned.drop(columns=[row_name])


def load_feature_space(feature_space: str, pre_cache: Path = DEFAULT_PRE_CACHE, angular_cache: Path = DEFAULT_PRE_A_CACHE):
    feature_space = feature_space.lower()
    if feature_space == "bob":
        x, ids = load_bob_features()
        return x, ids, {"feature_space": "BoB", "n_features": int(x.shape[1])}
    if feature_space == "pre":
        x, ids = load_pre_features(pre_cache)
        return x, ids, {"feature_space": "PRE", "n_features": int(x.shape[1])}
    if feature_space in {"pre_plus_a", "pre+a"}:
        x_pre, pre_ids = load_pre_features(pre_cache)
        x_ang, ang_ids = load_angular_features(angular_cache)
        base = pd.DataFrame({"mol_id": pre_ids, "_pre_row": np.arange(len(pre_ids), dtype=int)})
        ang = pd.DataFrame({"mol_id": ang_ids, "_ang_row": np.arange(len(ang_ids), dtype=int)})
        rows = base.merge(ang, on="mol_id", how="inner", validate="one_to_one")
        x = hstack(
            [x_pre[rows["_pre_row"].to_numpy(int)], x_ang[rows["_ang_row"].to_numpy(int)]],
            format="csr",
            dtype=np.float64,
        )
        ids = rows["mol_id"].astype(str).reset_index(drop=True)
        return x, ids, {
            "feature_space": "PRE+A",
            "pre_dims": int(x_pre.shape[1]),
            "angular_dims": int(x_ang.shape[1]),
            "n_features": int(x.shape[1]),
        }
    raise ValueError("feature_space must be one of: bob, pre, pre_plus_a")
