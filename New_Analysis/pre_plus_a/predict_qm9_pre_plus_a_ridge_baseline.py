#!/usr/bin/env python3
"""Apply the fitted linear PRE+A ridge baseline to QM9.

This is the baseline term needed to score QM9 transfer for
``Delta^PRE+A_ref,BoE`` GNN predictions:

    DFT_ref,BoE_pred = PRE+A_ref,BoE_QM9 + GNN(Delta^PRE+A_ref,BoE)_QM9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz


ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_sparse_with_ids(cache_dir: Path, matrix_name: str, ids_name: str, label: str):
    matrix_path = cache_dir / matrix_name
    ids_path = cache_dir / ids_name
    require(matrix_path, f"{label} matrix")
    require(ids_path, f"{label} mol_id table")
    x = load_npz(matrix_path).tocsr()
    ids = pd.read_csv(ids_path, usecols=["mol_id"])["mol_id"].astype(str).reset_index(drop=True)
    if x.shape[0] != len(ids):
        raise ValueError(f"{label} row mismatch: matrix rows={x.shape[0]} ids={len(ids)}")
    return x, ids


def load_qm9_pre_plus_a(pre_cache: Path, angular_cache: Path):
    x_pre, pre_ids = load_sparse_with_ids(pre_cache, "tmqm_pre_features.npz", "tmqm_pre_targets_aligned.csv", "QM9 PRE")
    x_ang, ang_ids = load_sparse_with_ids(
        angular_cache, "tmqm_angular_features.npz", "tmqm_angular_mol_ids.csv", "QM9 angular"
    )
    left = pd.DataFrame({"mol_id": pre_ids, "_pre_row": np.arange(len(pre_ids), dtype=int)})
    right = pd.DataFrame({"mol_id": ang_ids, "_ang_row": np.arange(len(ang_ids), dtype=int)})
    aligned = left.merge(right, on="mol_id", how="inner", validate="one_to_one")
    if len(aligned) == 0:
        raise ValueError("QM9 PRE and angular caches have no overlapping mol_id values")
    x = hstack(
        [
            x_pre[aligned["_pre_row"].to_numpy(int)],
            x_ang[aligned["_ang_row"].to_numpy(int)],
        ],
        format="csr",
    )
    return x, aligned["mol_id"].reset_index(drop=True), {"pre_dims": x_pre.shape[1], "angular_dims": x_ang.shape[1]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-cache", type=Path, default=ROOT / "New_Analysis/features/QM9_PRE")
    parser.add_argument("--angular-cache", type=Path, default=ROOT / "New_Analysis/features/QM9_PRE_A")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "New_Analysis/pre_plus_a/outputs_high_alpha_cv/pre_plus_a_ridge_model.joblib",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=ROOT / "New_Analysis/source_artifacts/boe/QM9__BoE_tmQM_ridge__residuals.csv",
    )
    parser.add_argument("--truth-column", default="E_resid_BoE_tmQM")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "New_Analysis/pre_plus_a/outputs_qm9/qm9_pre_plus_a_ridge_predictions_dft_ref_boe.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.model, "PRE+A ridge model")
    require(args.truth_csv, "QM9 DFT_ref,BoE truth CSV")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    log("Loading QM9 PRE+A features")
    x, ids, design = load_qm9_pre_plus_a(args.pre_cache, args.angular_cache)
    log(f"QM9 PRE+A rows={x.shape[0]} dims={x.shape[1]} pre_dims={design['pre_dims']} angular_dims={design['angular_dims']}")

    bundle = joblib.load(args.model)
    fit = bundle["fit"]
    keep = np.asarray(fit["feature_keep_mask"], dtype=bool)
    if keep.shape[0] != x.shape[1]:
        raise ValueError(f"Feature dim mismatch: QM9 dims={x.shape[1]} model mask={keep.shape[0]}")
    log(f"Predicting with retained={int(keep.sum())} dropped={int((~keep).sum())}")
    x_scaled = fit["scaler"].transform(x[:, keep].tocsr())
    pred = np.asarray(fit["model"].predict(x_scaled), dtype=float).reshape(-1)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite PRE+A predictions")

    out = pd.DataFrame({"mol_id": ids.astype(str), "PRE_plus_A_ref-BoE": pred})
    truth = pd.read_csv(args.truth_csv, usecols=["mol_id", args.truth_column]).rename(
        columns={args.truth_column: "DFT_ref-BoE"}
    )
    truth["mol_id"] = truth["mol_id"].astype(str)
    out = out.merge(truth, on="mol_id", how="left", validate="one_to_one")
    out["Delta_PRE_plus_A-ref-BoE"] = out["DFT_ref-BoE"] - out["PRE_plus_A_ref-BoE"]
    out.to_csv(args.out_csv, index=False)

    scored = out.dropna(subset=["DFT_ref-BoE"]).copy()
    err = scored["PRE_plus_A_ref-BoE"].to_numpy(float) - scored["DFT_ref-BoE"].to_numpy(float)
    log(f"WROTE: {args.out_csv}")
    print(
        pd.DataFrame(
            [
                {
                    "n": int(len(out)),
                    "n_with_truth": int(len(scored)),
                    "prediction_mean": float(np.mean(pred)),
                    "prediction_std": float(np.std(pred)),
                    "mae_vs_DFT_ref_BoE": float(np.mean(np.abs(err))),
                    "rmse_vs_DFT_ref_BoE": float(np.sqrt(np.mean(err**2))),
                    "bias_vs_DFT_ref_BoE": float(np.mean(err)),
                }
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
