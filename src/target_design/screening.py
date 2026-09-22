"""Interpret target scale and roughness jointly without collapsing them."""

from __future__ import annotations

import math

import pandas as pd


def compare_to_parent(
    table: pd.DataFrame,
    parent: str,
    *,
    target_column: str = "target",
    scale_column: str = "target_iqr",
    roughness_column: str = "d_iqr",
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Compare candidate targets with a parent in the scale-roughness plane.

    A candidate is ``favoured`` only when it reduces scale while maintaining or
    lowering roughness. A smaller but rougher residual is reported as a
    ``tradeoff`` because D_IQR alone does not determine absolute error.
    """

    required = {target_column, scale_column, roughness_column}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if table[target_column].duplicated().any():
        raise ValueError(f"{target_column!r} values must be unique.")
    parent_rows = table[table[target_column].eq(parent)]
    if len(parent_rows) != 1:
        raise ValueError(f"Expected exactly one parent row named {parent!r}.")
    parent_row = parent_rows.iloc[0]
    parent_scale = float(parent_row[scale_column])
    parent_roughness = float(parent_row[roughness_column])
    if not (math.isfinite(parent_scale) and parent_scale > 0):
        raise ValueError("Parent scale must be positive and finite.")
    if not (math.isfinite(parent_roughness) and parent_roughness > 0):
        raise ValueError("Parent roughness must be positive and finite.")

    result = table.copy()
    result["scale_ratio_to_parent"] = result[scale_column].astype(float) / parent_scale
    result["roughness_ratio_to_parent"] = result[roughness_column].astype(float) / parent_roughness

    def classify(row: pd.Series) -> str:
        if row[target_column] == parent:
            return "parent"
        scale_ratio = float(row["scale_ratio_to_parent"])
        roughness_ratio = float(row["roughness_ratio_to_parent"])
        if scale_ratio < 1.0 - tolerance and roughness_ratio <= 1.0 + tolerance:
            return "favoured"
        if scale_ratio < 1.0 - tolerance and roughness_ratio > 1.0 + tolerance:
            return "tradeoff"
        return "not_scale_reducing"

    result["screening_class"] = result.apply(classify, axis=1)
    return result
