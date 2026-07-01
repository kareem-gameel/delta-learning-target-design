#!/usr/bin/env python3
"""Fit PRE+A ridge baseline to DFT_ref,BoE with GemNet split protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = ROOT / "New_Analysis/bob_predict_targets"
sys.path.insert(0, str(HELPER_DIR))

from run_bob_predict_targets import load_split_ids, parse_alphas, require_file  # noqa: E402


DEFAULT_TARGET_CSV = ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv"
DEFAULT_SPLIT_DIR = ROOT / "trillium_project/split_ids"
DEFAULT_PRE_CACHE = ROOT / "New_Analysis/features/PRE/trillium_cache"
DEFAULT_ANGULAR_CACHE = ROOT / "New_Analysis/features/PRE_A"
DEFAULT_OUT_DIR = ROOT / "New_Analysis/pre_plus_a/outputs"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_pre_features(cache_dir: Path) -> tuple[csr_matrix, pd.Series]:
    feature_path = cache_dir / "tmqm_pre_features.npz"
    ids_path = cache_dir / "tmqm_pre_targets_aligned.csv"
    require_file(feature_path, "PRE feature matrix")
    require_file(ids_path, "PRE mol_id table")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"PRE row mismatch: x rows={x.shape[0]} mol_ids={len(ids)}")
    return x, ids


def load_angular_features(cache_dir: Path) -> tuple[csr_matrix, pd.Series, dict[str, object]]:
    feature_path = cache_dir / "tmqm_angular_features.npz"
    ids_path = cache_dir / "tmqm_angular_mol_ids.csv"
    meta_path = cache_dir / "tmqm_angular_meta.json"
    require_file(feature_path, "angular feature matrix")
    require_file(ids_path, "angular mol_id table")
    require_file(meta_path, "angular metadata")
    x = load_npz(feature_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"Angular row mismatch: x rows={x.shape[0]} mol_ids={len(ids)}")
    meta = json.loads(meta_path.read_text())
    return x, ids, meta


def load_target(target_csv: Path, target_column: str, split_dir: Path) -> pd.DataFrame:
    require_file(target_csv, "DFT_ref,BoE target CSV")
    split_df = load_split_ids(split_dir)
    df = pd.read_csv(target_csv)
    required = {"mol_id", target_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{target_csv} missing required columns: {missing}")
    df = df[["mol_id", target_column]].dropna().drop_duplicates("mol_id", keep="first")
    df["mol_id"] = df["mol_id"].astype(str)
    df = df.rename(columns={target_column: "target"})
    merged = df.merge(split_df, on="mol_id", how="inner", validate="one_to_one")
    missing_split = len(df) - len(merged)
    if missing_split:
        log(f"Excluded {missing_split} target rows without GemNet split assignment")
    return merged.reset_index(drop=True)


def apply_max_rows(df: pd.DataFrame, max_rows_per_split: int | None) -> pd.DataFrame:
    if max_rows_per_split is None:
        return df
    return (
        df.sort_values(["split", "mol_id"])
        .groupby("split", group_keys=False)
        .head(int(max_rows_per_split))
        .reset_index(drop=True)
    )


def align_feature_block(
    x: csr_matrix,
    ids: pd.Series,
    frame: pd.DataFrame,
    row_name: str,
) -> tuple[csr_matrix, pd.DataFrame]:
    row_df = pd.DataFrame({"mol_id": ids, row_name: np.arange(len(ids), dtype=int)})
    aligned = frame.merge(row_df, on="mol_id", how="inner", validate="one_to_one")
    missing = len(frame) - len(aligned)
    if missing:
        log(f"Excluded {missing} rows without {row_name} features")
    rows = aligned[row_name].to_numpy(int)
    return x[rows].tocsr(), aligned.drop(columns=[row_name]).reset_index(drop=True)


def load_aligned_design(
    target: pd.DataFrame,
    pre_cache: Path,
    angular_cache: Path,
) -> tuple[csr_matrix, pd.DataFrame, dict[str, object]]:
    x_pre_all, pre_ids = load_pre_features(pre_cache)
    x_ang_all, ang_ids, angular_meta = load_angular_features(angular_cache)

    x_pre, aligned = align_feature_block(x_pre_all, pre_ids, target, "_pre_row")
    x_ang, aligned = align_feature_block(x_ang_all, ang_ids, aligned, "_angular_row")

    # Re-align PRE after angular inner join so both blocks have identical rows.
    x_pre, aligned = align_feature_block(x_pre_all, pre_ids, aligned, "_pre_row")
    x = hstack([x_pre, x_ang], format="csr", dtype=np.float64)
    meta = {
        "pre_dims": int(x_pre.shape[1]),
        "angular_dims": int(x_ang.shape[1]),
        "pre_plus_a_dims": int(x.shape[1]),
        "angular_meta": angular_meta,
    }
    return x, aligned, meta


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {split: (df["split"] == split).to_numpy() for split in ["train", "val", "test"]}


def sparse_train_variance_mask(x_train: csr_matrix, eps: float) -> np.ndarray:
    mean = np.asarray(x_train.mean(axis=0)).reshape(-1)
    mean_sq = np.asarray(x_train.power(2).mean(axis=0)).reshape(-1)
    var = mean_sq - mean**2
    var[var < 0] = 0
    return var > eps


def cv_mae_for_alpha(
    alpha: float,
    x_scaled: csr_matrix,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    maes = []
    for tr, va in splits:
        model = Ridge(alpha=float(alpha), solver="auto")
        model.fit(x_scaled[tr], y[tr])
        pred = model.predict(x_scaled[va])
        maes.append(mean_absolute_error(y[va], pred))
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
    keep = sparse_train_variance_mask(x_train, zero_var_eps)
    if not keep.any():
        raise ValueError("No retained features after train-only zero-variance filtering.")

    x_train_keep = x_train[:, keep].tocsr()
    scaler = StandardScaler(with_mean=False, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train_keep)

    if len(alphas) == 1:
        alpha_grid_results = [(float(alphas[0]), float("nan"))]
        best_alpha = float(alphas[0])
        cv_mae = float("nan")
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(np.arange(x_train_scaled.shape[0])))
        if joblib is None or n_jobs == 1:
            alpha_grid_results = [cv_mae_for_alpha(a, x_train_scaled, y_train, splits) for a in alphas]
        else:
            alpha_grid_results = joblib.Parallel(n_jobs=n_jobs, prefer=parallel_prefer, verbose=10)(
                joblib.delayed(cv_mae_for_alpha)(a, x_train_scaled, y_train, splits) for a in alphas
            )
        best_alpha, cv_mae = min(alpha_grid_results, key=lambda item: item[1])

    model = Ridge(alpha=float(best_alpha), solver="auto")
    model.fit(x_train_scaled, y_train)
    return {
        "model": model,
        "scaler": scaler,
        "feature_keep_mask": keep,
        "best_alpha": float(best_alpha),
        "cv_mae": float(cv_mae),
        "alpha_grid_results": [(float(a), float(m)) for a, m in alpha_grid_results],
        "n_features_total": int(x.shape[1]),
        "n_features_retained": int(keep.sum()),
        "n_features_dropped": int((~keep).sum()),
        "zero_var_eps": float(zero_var_eps),
    }


def predict(x: csr_matrix, fit: dict[str, object]) -> np.ndarray:
    keep = fit["feature_keep_mask"]
    scaler = fit["scaler"]
    model = fit["model"]
    x_scaled = scaler.transform(x[:, keep].tocsr())
    return np.asarray(model.predict(x_scaled), dtype=float).reshape(-1)


def metric_row(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit: dict[str, object],
    n_train: int,
) -> dict[str, object]:
    resid = y_true - y_pred
    target_iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    mae = float(mean_absolute_error(y_true, y_pred))
    tau = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    return {
        "model_id": "PRE_plus_A",
        "target": "DFT_ref,BoE",
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
        "n_features_total": int(fit["n_features_total"]),
        "n_features_retained": int(fit["n_features_retained"]),
        "n_features_dropped": int(fit["n_features_dropped"]),
    }


def write_outputs(
    out_dir: Path,
    aligned: pd.DataFrame,
    y_pred: np.ndarray,
    fit: dict[str, object],
    design_meta: dict[str, object],
) -> None:
    pred = aligned[["mol_id", "split", "target"]].copy()
    pred["target_label"] = "DFT_ref,BoE"
    pred["pre_plus_a_pred"] = y_pred
    pred["error"] = pred["target"] - pred["pre_plus_a_pred"]
    pred["abs_error"] = pred["error"].abs()

    pred_path = out_dir / "pre_plus_a_predictions_dft_ref_boe.csv"
    pred.to_csv(pred_path, index=False)

    resid_col = "Delta_PRE_plus_A-ref-BoE"
    full = pred[["mol_id", "split", "target", "pre_plus_a_pred", "error", "abs_error"]].rename(
        columns={
            "target": "DFT_ref-BoE",
            "pre_plus_a_pred": "E_PRE_plus_A_ref_BoE",
            "error": resid_col,
        }
    )
    full_path = out_dir / "tmQM__PRE_plus_A_ref_BoE__pred_and_residual.csv"
    full.to_csv(full_path, index=False)

    delta_path = out_dir / "tmQM__Delta_PRE_plus_A-ref-BoE.csv"
    full[["mol_id", resid_col]].to_csv(delta_path, index=False)

    alpha_path = out_dir / "pre_plus_a_alpha_grid_results.json"
    alpha_path.write_text(json.dumps(fit["alpha_grid_results"], indent=2) + "\n")

    model_path = out_dir / "pre_plus_a_ridge_model.joblib"
    if joblib is not None:
        joblib.dump(
            {
                "fit": fit,
                "design_meta": design_meta,
                "target": "DFT_ref,BoE",
                "prediction_column": "E_PRE_plus_A_ref_BoE",
                "residual_column": resid_col,
            },
            model_path,
        )
    else:
        model_path.write_text("joblib is unavailable; model object was not serialized.\n")

    log(f"WROTE: {pred_path}")
    log(f"WROTE: {full_path}")
    log(f"WROTE: {delta_path}")
    log(f"WROTE: {alpha_path}")
    log(f"WROTE: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--target-column", default="DFT_ref-BoE")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pre-cache", type=Path, default=DEFAULT_PRE_CACHE)
    parser.add_argument("--angular-cache", type=Path, default=DEFAULT_ANGULAR_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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

    target = load_target(args.target_csv, args.target_column, args.split_dir)
    x, aligned, design_meta = load_aligned_design(target, args.pre_cache, args.angular_cache)
    if args.max_rows_per_split is not None:
        aligned = aligned.copy()
        aligned["_design_row"] = np.arange(len(aligned), dtype=int)
        aligned = apply_max_rows(aligned, args.max_rows_per_split)
        rows = aligned["_design_row"].to_numpy(int)
        x = x[rows].tocsr()
        aligned = aligned.drop(columns=["_design_row"]).reset_index(drop=True)
    masks = split_masks(aligned)
    validation = {
        "rows_by_split": aligned["split"].value_counts().to_dict(),
        "design": design_meta,
        "n_features_total": int(x.shape[1]),
        "n_rows": int(x.shape[0]),
        "target_csv": str(args.target_csv),
        "target_column": args.target_column,
    }
    validation_path = args.out_dir / "input_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    log(f"WROTE: {validation_path}")
    print(json.dumps(validation, indent=2))
    if args.validate_only:
        return

    if not masks["train"].any():
        raise ValueError("No train rows after alignment.")
    y = aligned["target"].to_numpy(float)
    alphas = parse_alphas(args.alphas)
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
    fit["fit_minutes"] = round((time.time() - t0) / 60.0, 3)
    y_pred = predict(x, fit)
    if not np.isfinite(y_pred).all():
        raise ValueError("Non-finite PRE+A predictions.")

    rows = []
    for split in ["train", "val", "test"]:
        if masks[split].any():
            rows.append(metric_row(split, y[masks[split]], y_pred[masks[split]], fit, int(masks["train"].sum())))
    metrics = pd.DataFrame(rows)
    metrics["fit_minutes"] = float(fit["fit_minutes"])
    metrics_path = args.out_dir / "pre_plus_a_metrics_raw.csv"
    metrics.to_csv(metrics_path, index=False)
    log(f"WROTE: {metrics_path}")

    write_outputs(args.out_dir, aligned, y_pred, fit, design_meta)


if __name__ == "__main__":
    main()
