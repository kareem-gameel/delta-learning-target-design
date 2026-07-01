#!/usr/bin/env python3
"""Compare split-aware PRE and PRE+A baselines on common validation rows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_PRED = ROOT / "New_Analysis/pre_predict_targets/outputs/pre_predictions_dft.csv"
DEFAULT_PRE_PLUS_A_PRED = ROOT / "New_Analysis/pre_plus_a/outputs/pre_plus_a_predictions_dft_ref_boe.csv"
DEFAULT_OUT_DIR = ROOT / "New_Analysis/pre_plus_a/outputs"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def modified_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad <= 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - med) / mad


def metric_row(model_id: str, metric_set: str, df: pd.DataFrame, pred_col: str) -> dict[str, object]:
    y_true = df["target"].to_numpy(float)
    y_pred = df[pred_col].to_numpy(float)
    resid = y_true - y_pred
    target_iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    mae = float(mean_absolute_error(y_true, y_pred))
    tau = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    return {
        "model_id": model_id,
        "metric_set": metric_set,
        "split": "val",
        "n": int(len(df)),
        "mae": mae,
        "mae_over_iqr": float(mae / target_iqr) if target_iqr > 0 else np.nan,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "kendall_tau": float(tau),
        "bias": float(np.mean(y_pred - y_true)),
        "residual_std": float(np.std(resid)),
        "target_std": float(np.std(y_true)),
        "target_iqr": target_iqr,
    }


def residual_stats(label: str, values: np.ndarray) -> dict[str, object]:
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    return {
        "residual_target": label,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "iqr": iqr,
        "median_abs": float(np.median(np.abs(values))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-pred", type=Path, default=DEFAULT_PRE_PRED)
    parser.add_argument("--pre-plus-a-pred", type=Path, default=DEFAULT_PRE_PLUS_A_PRED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--z-threshold", type=float, default=8.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_file(args.pre_pred, "PRE predictions for DFT_ref,BoE")
    require_file(args.pre_plus_a_pred, "PRE+A predictions for DFT_ref,BoE")

    pre = pd.read_csv(args.pre_pred)
    prea = pd.read_csv(args.pre_plus_a_pred)
    pre = pre[pre["split"].eq("val")].copy()
    prea = prea[prea["split"].eq("val")].copy()
    pre = pre[["mol_id", "target", "pre_pred", "abs_error"]].rename(
        columns={"pre_pred": "PRE_pred", "abs_error": "PRE_abs_error"}
    )
    prea = prea[["mol_id", "target", "pre_plus_a_pred", "abs_error"]].rename(
        columns={"pre_plus_a_pred": "PRE_plus_A_pred", "abs_error": "PRE_plus_A_abs_error"}
    )
    merged = pre.merge(prea, on=["mol_id", "target"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No common validation rows between PRE and PRE+A predictions.")

    pre_log = np.log10(merged["PRE_abs_error"].to_numpy(float) + args.eps)
    prea_log = np.log10(merged["PRE_plus_A_abs_error"].to_numpy(float) + args.eps)
    merged["PRE_outlier"] = np.abs(modified_z(pre_log)) > args.z_threshold
    merged["PRE_plus_A_outlier"] = np.abs(modified_z(prea_log)) > args.z_threshold
    merged["remove_shared_outlier"] = merged["PRE_outlier"] | merged["PRE_plus_A_outlier"]

    rows = [
        metric_row("PRE", "raw_common_rows", merged.rename(columns={"PRE_pred": "pred"}), "pred"),
        metric_row("PRE_plus_A", "raw_common_rows", merged.rename(columns={"PRE_plus_A_pred": "pred"}), "pred"),
    ]
    filtered = merged[~merged["remove_shared_outlier"]].copy()
    rows.extend(
        [
            metric_row("PRE", "shared_outlier_mask", filtered.rename(columns={"PRE_pred": "pred"}), "pred"),
            metric_row(
                "PRE_plus_A",
                "shared_outlier_mask",
                filtered.rename(columns={"PRE_plus_A_pred": "pred"}),
                "pred",
            ),
        ]
    )
    metrics = pd.DataFrame(rows)
    metrics_path = args.out_dir / "pre_vs_pre_plus_a_common_mask_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    removed_path = args.out_dir / "pre_vs_pre_plus_a_removed_outliers.csv"
    merged[merged["remove_shared_outlier"]].to_csv(removed_path, index=False)

    residual_rows = [
        residual_stats("Delta^PRE_ref,BoE", merged["target"].to_numpy(float) - merged["PRE_pred"].to_numpy(float)),
        residual_stats(
            "Delta^PRE+A_ref,BoE",
            merged["target"].to_numpy(float) - merged["PRE_plus_A_pred"].to_numpy(float),
        ),
        residual_stats(
            "Delta^PRE_ref,BoE_shared_mask",
            filtered["target"].to_numpy(float) - filtered["PRE_pred"].to_numpy(float),
        ),
        residual_stats(
            "Delta^PRE+A_ref,BoE_shared_mask",
            filtered["target"].to_numpy(float) - filtered["PRE_plus_A_pred"].to_numpy(float),
        ),
    ]
    residual_path = args.out_dir / "pre_vs_pre_plus_a_residual_target_stats.csv"
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)

    log(f"WROTE: {metrics_path}")
    log(f"WROTE: {removed_path}")
    log(f"WROTE: {residual_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
