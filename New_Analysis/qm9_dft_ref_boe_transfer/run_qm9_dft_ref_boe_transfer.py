#!/usr/bin/env python3
"""Evaluate QM9 GemNet predictions against DFT_ref,BoE."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = THIS_DIR / "model_manifest.csv"
DEFAULT_OUT = THIS_DIR / "outputs"


@dataclass(frozen=True)
class Spec:
    model_id: str
    display_name: str
    mode: str
    npz_path: Path | None
    sid_lmdb_path: Path | None
    baseline_csv: Path | None
    baseline_columns: tuple[str, ...]
    true_csv: Path | None
    true_column: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--models", default="", help="Optional comma-separated model_id subset.")
    return parser.parse_args()


def resolve(raw: object) -> Path | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def split_cols(raw: object) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ()
    return tuple(part.strip() for part in str(raw).split(";") if part.strip())


def read_manifest(path: Path) -> list[Spec]:
    df = pd.read_csv(path, keep_default_na=False)
    specs = []
    for row in df.to_dict(orient="records"):
        specs.append(
            Spec(
                model_id=row["model_id"],
                display_name=row["display_name"],
                mode=row["mode"],
                npz_path=resolve(row["npz_path"]),
                sid_lmdb_path=resolve(row["sid_lmdb_path"]),
                baseline_csv=resolve(row["baseline_csv"]),
                baseline_columns=split_cols(row["baseline_columns"]),
                true_csv=resolve(row["true_csv"]),
                true_column=row["true_column"],
                notes=row["notes"],
            )
        )
    return specs


def validate(specs: list[Spec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        missing = []
        column_errors = []

        if spec.npz_path is None or not spec.npz_path.exists():
            missing.append(str(spec.npz_path or "missing npz_path"))
        if spec.sid_lmdb_path is None or not spec.sid_lmdb_path.exists():
            missing.append(str(spec.sid_lmdb_path or "missing sid_lmdb_path"))
        if spec.true_csv is None or not spec.true_csv.exists():
            missing.append(str(spec.true_csv or "missing true_csv"))
        elif spec.true_column not in pd.read_csv(spec.true_csv, nrows=0).columns:
            column_errors.append(f"{spec.true_csv} missing {spec.true_column}")

        if spec.mode == "plus_baseline":
            if spec.baseline_csv is None or not spec.baseline_csv.exists():
                missing.append(str(spec.baseline_csv or "missing baseline_csv"))
            else:
                baseline_cols = set(pd.read_csv(spec.baseline_csv, nrows=0).columns)
                for col in spec.baseline_columns:
                    if col not in baseline_cols:
                        column_errors.append(f"{spec.baseline_csv} missing {col}")
        elif spec.mode != "direct":
            column_errors.append(f"unknown mode {spec.mode}")

        rows.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "mode": spec.mode,
                "status": "ready" if not missing and not column_errors else "pending",
                "missing_inputs": " | ".join(missing),
                "column_errors": " | ".join(column_errors),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def load_npz(path: Path) -> pd.DataFrame:
    z = np.load(path)
    if "ids" not in z.files:
        raise ValueError(f"{path} missing ids")
    if "energy" in z.files:
        pred = np.asarray(z["energy"]).reshape(-1).astype(float)
    else:
        pred = None
        for key in z.files:
            arr = np.asarray(z[key])
            if key != "ids" and np.issubdtype(arr.dtype, np.number):
                pred = arr.reshape(-1).astype(float)
                break
        if pred is None:
            raise ValueError(f"{path} has no numeric prediction array")
    ids = np.asarray(z["ids"]).reshape(-1).astype(int)
    n = min(len(ids), len(pred))
    return pd.DataFrame({"sid": ids[:n], "gemnet_prediction": pred[:n]})


def load_lmdb_sid_map(path: Path) -> pd.DataFrame:
    import lmdb

    rows = []
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    with env.begin() as txn:
        for key, value in txn.cursor():
            if key == b"length":
                continue
            data = pickle.loads(value)
            rows.append({"sid": int(getattr(data, "sid")), "mol_id": str(getattr(data, "mol_id", getattr(data, "sid")))})
    env.close()
    return pd.DataFrame(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, shift: float) -> dict[str, float]:
    from scipy.stats import kendalltau

    err = y_pred - y_true
    sse = float(np.sum(err**2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    tau = kendalltau(y_true, y_pred).statistic
    return {
        "n": int(y_true.size),
        "shift": float(shift),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst else np.nan,
        "kendall_tau": float(tau),
        "bias": float(np.mean(y_pred - y_true)),
    }


def evaluate_one(spec: Spec) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pred = load_npz(spec.npz_path)
    sid_map = load_lmdb_sid_map(spec.sid_lmdb_path)
    base = pred.merge(sid_map, on="sid", how="left", validate="many_to_one")

    true_df = pd.read_csv(spec.true_csv, usecols=["mol_id", spec.true_column]).rename(columns={spec.true_column: "y_true"})
    base = base.merge(true_df, on="mol_id", how="inner", validate="many_to_one")

    baseline_expr = ""
    if spec.mode == "plus_baseline":
        baseline_df = pd.read_csv(spec.baseline_csv, usecols=["mol_id", *spec.baseline_columns])
        baseline_df["baseline_prediction"] = baseline_df[list(spec.baseline_columns)].sum(axis=1)
        base = base.merge(baseline_df[["mol_id", "baseline_prediction"]], on="mol_id", how="inner", validate="many_to_one")
        base["y_pred_raw"] = base["baseline_prediction"] + base["gemnet_prediction"]
        baseline_expr = ";".join(spec.baseline_columns)
    else:
        base["baseline_prediction"] = 0.0
        base["y_pred_raw"] = base["gemnet_prediction"]

    raw_shift = 0.0
    mean_shift = float(np.mean(base["y_true"].to_numpy(float) - base["y_pred_raw"].to_numpy(float)))
    base["y_pred_mean_shift"] = base["y_pred_raw"] + mean_shift

    rows = []
    for calibration, pred_col, shift in [
        ("raw", "y_pred_raw", raw_shift),
        ("oracle_mean_shift", "y_pred_mean_shift", mean_shift),
    ]:
        m = metrics(base["y_true"].to_numpy(float), base[pred_col].to_numpy(float), shift)
        rows.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "mode": spec.mode,
                "calibration": calibration,
                "target": "DFT_ref,BoE",
                "prediction_column": pred_col,
                "baseline_columns": baseline_expr,
                **m,
                "notes": spec.notes,
            }
        )

    out = base[
        [
            "mol_id",
            "sid",
            "y_true",
            "gemnet_prediction",
            "baseline_prediction",
            "y_pred_raw",
            "y_pred_mean_shift",
        ]
    ].copy()
    out.insert(0, "model_id", spec.model_id)
    out.insert(1, "display_name", spec.display_name)
    return out, rows


def main() -> None:
    args = parse_args()
    specs = read_manifest(args.manifest)
    if args.models:
        requested = {part.strip() for part in args.models.split(",") if part.strip()}
        specs = [spec for spec in specs if spec.model_id in requested]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation = validate(specs)
    validation.to_csv(args.out_dir / "manifest_validation.csv", index=False)
    print("WROTE:", args.out_dir / "manifest_validation.csv")
    print(validation.to_string(index=False))
    if args.validate_only:
        return

    all_predictions = []
    all_metrics = []
    for spec, status in zip(specs, validation["status"]):
        if status != "ready":
            print("SKIP pending:", spec.model_id)
            continue
        print("RUN:", spec.model_id)
        pred_df, metric_rows = evaluate_one(spec)
        all_predictions.append(pred_df)
        all_metrics.extend(metric_rows)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(args.out_dir / "qm9_DFT_ref_BoE_gemnet_predictions.csv", index=False)
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(args.out_dir / "qm9_DFT_ref_BoE_gemnet_metrics.csv", index=False)
        print("WROTE:", args.out_dir / "qm9_DFT_ref_BoE_gemnet_predictions.csv")
        print("WROTE:", args.out_dir / "qm9_DFT_ref_BoE_gemnet_metrics.csv")
        print(metrics_df.sort_values(["calibration", "mae"]).to_string(index=False))


if __name__ == "__main__":
    main()
