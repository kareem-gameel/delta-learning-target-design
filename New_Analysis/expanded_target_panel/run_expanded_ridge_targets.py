#!/usr/bin/env python3
"""Fit BoB, PRE, or PRE+A ridge predictors to the expanded target panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from expanded_panel_common import (
    DEFAULT_PRE_A_CACHE,
    DEFAULT_PRE_CACHE,
    DEFAULT_SPLIT_DIR,
    TargetSpec,
    align_feature_block,
    apply_max_rows,
    load_feature_space,
    load_split_ids,
    load_target_frame,
    parse_target_ids,
    selected_targets,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_alphas(value: str) -> np.ndarray:
    if "," in value:
        return np.asarray([float(v.strip()) for v in value.split(",") if v.strip()], dtype=float)
    if ":" in value:
        start, stop, count = value.split(":")
        return np.logspace(float(start), float(stop), int(count))
    return np.asarray([float(value)], dtype=float)


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {split: df["split"].eq(split).to_numpy() for split in ["train", "val", "test"]}


def sparse_train_scale(x_train: csr_matrix, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x_train.mean(axis=0)).ravel()
    mean_sq = np.asarray(x_train.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0)
    keep = var > eps
    scale = np.sqrt(var[keep])
    return keep, scale


def scale_with_train_stats(x: csr_matrix, keep: np.ndarray, scale: np.ndarray) -> csr_matrix:
    return (x[:, keep].tocsr() @ diags(1.0 / scale, offsets=0, format="csr")).tocsr()


def cv_alpha(alpha: float, x_train_scaled: csr_matrix, y_train: np.ndarray, splits) -> tuple[float, float]:
    maes = []
    for tr, va in splits:
        model = Ridge(alpha=float(alpha), solver="auto")
        model.fit(x_train_scaled[tr], y_train[tr])
        pred = model.predict(x_train_scaled[va])
        maes.append(mean_absolute_error(y_train[va], pred))
    return float(alpha), float(np.mean(maes))


def fit_train_only(
    x: csr_matrix,
    y: np.ndarray,
    train_mask: np.ndarray,
    alphas: np.ndarray,
    n_splits: int,
    n_jobs: int,
    parallel_prefer: str,
    zero_var_eps: float,
) -> dict[str, object]:
    x_train = x[train_mask].tocsr()
    y_train = y[train_mask]
    keep, scale = sparse_train_scale(x_train, zero_var_eps)
    if not keep.any():
        raise ValueError("No retained features after train-only variance filtering.")
    x_train_scaled = scale_with_train_stats(x_train, keep, scale)

    if len(alphas) == 1:
        alpha_grid = [(float(alphas[0]), float("nan"))]
        best_alpha = float(alphas[0])
        cv_mae = float("nan")
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(np.arange(x_train_scaled.shape[0])))
        if joblib is None or n_jobs == 1:
            alpha_grid = [cv_alpha(alpha, x_train_scaled, y_train, splits) for alpha in alphas]
        else:
            alpha_grid = joblib.Parallel(n_jobs=n_jobs, prefer=parallel_prefer, verbose=10)(
                joblib.delayed(cv_alpha)(alpha, x_train_scaled, y_train, splits) for alpha in alphas
            )
        best_alpha, cv_mae = min(alpha_grid, key=lambda row: row[1])

    model = Ridge(alpha=float(best_alpha), solver="auto")
    model.fit(x_train_scaled, y_train)
    return {
        "model": model,
        "keep": keep,
        "scale": scale,
        "best_alpha": float(best_alpha),
        "cv_mae": float(cv_mae),
        "alpha_grid": [(float(a), float(m)) for a, m in alpha_grid],
        "n_features_total": int(x.shape[1]),
        "n_features_retained": int(keep.sum()),
        "n_features_dropped": int((~keep).sum()),
    }


def predict(x: csr_matrix, fit: dict[str, object]) -> np.ndarray:
    x_scaled = scale_with_train_stats(x, fit["keep"], fit["scale"])
    return np.asarray(fit["model"].predict(x_scaled), dtype=float).reshape(-1)


def metric_row(
    model_id: str,
    feature_meta: dict[str, object],
    target: TargetSpec,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit: dict[str, object],
    n_train: int,
    fit_minutes: float,
) -> dict[str, object]:
    resid = y_true - y_pred
    target_iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    mae = float(mean_absolute_error(y_true, y_pred))
    tau = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    return {
        "model_id": model_id,
        "target_id": target.target_id,
        "target_label": target.target_label,
        "target_family": target.target_family,
        "target_csv": str(target.csv_path),
        "target_column": target.column,
        "split": split,
        "n": int(len(y_true)),
        "n_train": int(n_train),
        "mae": mae,
        "mae_over_iqr": float(mae / target_iqr) if target_iqr > 0 else np.nan,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "kendall_tau": float(tau),
        "bias": float(np.mean(y_pred - y_true)),
        "residual_std": float(np.std(resid)),
        "target_std": float(np.std(y_true)),
        "target_iqr": target_iqr,
        "best_alpha": float(fit["best_alpha"]),
        "cv_mae": float(fit["cv_mae"]),
        "fit_minutes": float(fit_minutes),
        "n_features_total": int(fit["n_features_total"]),
        "n_features_retained": int(fit["n_features_retained"]),
        "n_features_dropped": int(fit["n_features_dropped"]),
        "feature_space": feature_meta["feature_space"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-space", required=True, choices=["bob", "pre", "pre_plus_a"])
    parser.add_argument("--targets", default="all")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pre-cache", type=Path, default=DEFAULT_PRE_CACHE)
    parser.add_argument("--angular-cache", type=Path, default=DEFAULT_PRE_A_CACHE)
    parser.add_argument("--out-dir", type=Path, default=Path("New_Analysis/expanded_target_panel/outputs/ridge"))
    parser.add_argument("--alphas", default="-6:3:20")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--parallel-prefer", default="threads", choices=["threads", "processes"])
    parser.add_argument("--zero-var-eps", type=float, default=1e-12)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_id = {"bob": "BoB", "pre": "PRE", "pre_plus_a": "PRE_plus_A"}[args.feature_space]

    # Parse early so unknown IDs fail even when some target files are not present yet.
    _ = parse_target_ids(args.targets)
    targets = selected_targets(args.targets, require_existing=True)
    split_df = load_split_ids(args.split_dir)
    x_all, feature_ids, feature_meta = load_feature_space(args.feature_space, args.pre_cache, args.angular_cache)

    validation = []
    rows = []
    alpha_records = {}
    alphas = parse_alphas(args.alphas)

    for target in targets:
        target_df = apply_max_rows(load_target_frame(target, split_df), args.max_rows_per_split)
        x, aligned = align_feature_block(x_all, feature_ids, target_df, "_feature_row")
        validation.append(
            {
                "model_id": model_id,
                "target_id": target.target_id,
                "target_label": target.target_label,
                "target_csv": str(target.csv_path),
                "target_column": target.column,
                "rows_by_split": aligned["split"].value_counts().to_dict(),
                "feature_rows": int(x.shape[0]),
                "feature_dims": int(x.shape[1]),
                "feature_space": feature_meta["feature_space"],
            }
        )
        if args.validate_only:
            continue

        log(f"RUN: {model_id} -> {target.target_label}")
        masks = split_masks(aligned)
        y = aligned["target"].to_numpy(float)
        t0 = time.time()
        fit = fit_train_only(
            x=x,
            y=y,
            train_mask=masks["train"],
            alphas=alphas,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
            parallel_prefer=args.parallel_prefer,
            zero_var_eps=args.zero_var_eps,
        )
        fit_minutes = round((time.time() - t0) / 60.0, 3)
        y_pred = predict(x, fit)
        if not np.isfinite(y_pred).all():
            raise ValueError(f"Non-finite predictions for {model_id} -> {target.target_id}")

        pred = aligned[["mol_id", "split", "target"]].copy()
        pred["target_id"] = target.target_id
        pred["target_label"] = target.target_label
        pred["model_id"] = model_id
        pred["y_pred"] = y_pred
        pred["error"] = pred["target"] - pred["y_pred"]
        pred["abs_error"] = pred["error"].abs()
        pred_path = args.out_dir / f"{model_id.lower()}_predictions_{target.target_id}.csv"
        pred.to_csv(pred_path, index=False)
        log(f"WROTE: {pred_path}")

        alpha_records[target.target_id] = fit["alpha_grid"]
        for split in ["train", "val", "test"]:
            if masks[split].any():
                rows.append(
                    metric_row(
                        model_id,
                        feature_meta,
                        target,
                        split,
                        y[masks[split]],
                        y_pred[masks[split]],
                        fit,
                        int(masks["train"].sum()),
                        fit_minutes,
                    )
                )

    validation_path = args.out_dir / f"{model_id.lower()}_input_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    log(f"WROTE: {validation_path}")
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return

    metrics = pd.DataFrame(rows)
    metrics_path = args.out_dir / f"{model_id.lower()}_target_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    alpha_path = args.out_dir / f"{model_id.lower()}_alpha_grid_results.json"
    alpha_path.write_text(json.dumps(alpha_records, indent=2) + "\n")
    log(f"WROTE: {metrics_path}")
    log(f"WROTE: {alpha_path}")


if __name__ == "__main__":
    main()
