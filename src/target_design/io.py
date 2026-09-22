"""Input helpers with explicit molecule-ID alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz


def _validated_ids(values, label: str) -> pd.Series:
    raw = pd.Series(values, dtype="string").reset_index(drop=True)
    if raw.isna().any() or (raw.str.len() == 0).any():
        raise ValueError(f"{label} contains missing or empty molecule IDs.")
    ids = raw.astype(str)
    duplicate = ids[ids.duplicated(keep=False)].unique()
    if len(duplicate):
        raise ValueError(f"{label} contains duplicate molecule IDs: {duplicate[:5].tolist()}")
    return ids


def load_features(path: Path, id_column: str = "mol_id", ids_path: Path | None = None):
    """Load dense CSV or SciPy sparse NPZ features and their row IDs."""

    path = Path(path)
    if path.suffix.lower() == ".npz":
        if ids_path is None:
            raise ValueError("--feature-ids is required for sparse .npz features.")
        matrix = load_npz(path).tocsr()
        ids_frame = pd.read_csv(ids_path)
        if id_column not in ids_frame:
            raise ValueError(f"{ids_path} does not contain {id_column!r}.")
        ids = _validated_ids(ids_frame[id_column], "Feature IDs")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if id_column not in frame:
            raise ValueError(f"{path} does not contain {id_column!r}.")
        ids = _validated_ids(frame.pop(id_column), "Feature IDs")
        try:
            matrix = csr_matrix(frame.to_numpy(dtype=float))
        except ValueError as exc:
            raise ValueError("Every feature column must be numeric.") from exc
    else:
        raise ValueError("Features must be a .csv table or SciPy sparse .npz matrix.")
    if matrix.shape[0] != len(ids):
        raise ValueError(f"Feature row mismatch: matrix={matrix.shape[0]}, IDs={len(ids)}.")
    return matrix, ids


def align_features_and_targets(
    features,
    feature_ids: pd.Series,
    target_path: Path,
    target_columns: list[str],
    id_column: str = "mol_id",
):
    """Align features and complete target rows by molecule ID."""

    targets = pd.read_csv(target_path)
    missing = sorted({id_column, *target_columns} - set(targets.columns))
    if missing:
        raise ValueError(f"Target table is missing columns: {missing}")
    targets = targets[[id_column, *target_columns]].dropna().copy()
    targets[id_column] = targets[id_column].astype(str)
    if targets[id_column].duplicated().any():
        raise ValueError("Target molecule IDs must be unique.")
    feature_rows = pd.DataFrame({id_column: feature_ids, "_feature_row": np.arange(len(feature_ids))})
    aligned = targets.merge(feature_rows, on=id_column, how="inner", validate="one_to_one")
    if aligned.empty:
        raise ValueError("No molecule IDs are shared by features and targets.")
    rows = aligned.pop("_feature_row").to_numpy(dtype=int)
    return features[rows].tocsr(), aligned.reset_index(drop=True)
