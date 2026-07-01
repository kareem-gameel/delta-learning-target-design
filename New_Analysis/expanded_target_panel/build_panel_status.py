#!/usr/bin/env python3
"""Build current completion/status tables for the expanded 9-target panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from expanded_panel_common import TARGETS, load_split_ids, load_target_frame


ROOT = Path(__file__).resolve().parents[2]

MODEL_ORDER = ["BoB", "PRE", "PRE_plus_A", "GemNet", "SchNet"]

ORIGINAL_FIVE = {"dft", "xtb", "deltaxtb", "deltabob", "deltapre"}

GEMNET_TARGET_MAP = {
    "DFT_ref,BoE": "dft",
    "xTB_ref,BoE": "xtb",
    "DFT_ref,BoE,BoB": "deltabob",
    "DFT_ref,BoE,PRE": "deltapre",
}

SCHNET_TARGET_MAP = {
    "SchNet(DFT_ref,BoE)": "dft",
    "SchNet(xTB_ref,BoE)": "xtb",
    "SchNet(Delta^xTB_ref,BoE)": "deltaxtb",
    "SchNet(Delta^BoB_ref,BoE)": "deltabob",
    "SchNet(Delta^PRE_ref,BoE)": "deltapre",
}


def target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": target.target_id,
                "target_label": target.target_label,
                "short_label": target.short_label,
                "target_family": target.target_family,
            }
            for target in TARGETS
        ]
    )


def val_target_iqrs() -> dict[str, float]:
    split_df = load_split_ids()
    out = {}
    for target in TARGETS:
        df = load_target_frame(target, split_df)
        val = df[df["split"].eq("val")]["target"].to_numpy(float)
        if len(val):
            out[target.target_id] = float(np.subtract(*np.percentile(val, [75, 25])))
    return out


def normalize_metric_rows(df: pd.DataFrame, model_id: str, metric_scope: str) -> pd.DataFrame:
    out = df.copy()
    out["model_id"] = model_id
    out["metric_scope"] = metric_scope
    keep = [
        "model_id",
        "target_id",
        "target_label",
        "metric_scope",
        "split",
        "n",
        "mae",
        "mae_over_iqr",
        "rmse",
        "r2",
        "kendall_tau",
        "target_std",
        "target_iqr",
        "best_alpha",
        "cv_mae",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def load_bob(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    val = df[df["split"].eq("val")].copy()
    return normalize_metric_rows(val, "BoB", "val")


def load_pre_original(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["split"] = "val"
    target_labels = target_frame()[["target_id", "target_label"]]
    df = df.merge(target_labels, on="target_id", how="left", validate="many_to_one")
    return normalize_metric_rows(df, "PRE", "shared_filtered_val")


def load_pre_expanded(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    val = df[df["split"].eq("val")].copy()
    # Original five PRE rows already have the filtered table; prefer that for now.
    original = {"dft", "xtb", "deltaxtb", "deltabob", "deltapre"}
    val = val[~val["target_id"].isin(original)].copy()
    return normalize_metric_rows(val, "PRE", "val")


def load_pre_plus_a(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    val = df[df["split"].eq("val")].copy()
    return normalize_metric_rows(val, "PRE_plus_A", "val")


def load_gemnet(path: Path, delta_path: Path, iqrs: dict[str, float]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    target_meta = target_frame()[["target_id", "target_label"]].set_index("target_id")
    rows = []
    df = pd.read_csv(path)
    for _, rec in df.iterrows():
        target_id = GEMNET_TARGET_MAP.get(str(rec.get("new_target", "")))
        if target_id is None:
            continue
        rows.append(
            {
                "target_id": target_id,
                "target_label": target_meta.loc[target_id, "target_label"],
                "split": "val",
                "n": np.nan,
                "mae": float(rec["best_val_mae"]),
                "mae_over_iqr": float(rec["best_val_mae"]) / iqrs[target_id],
                "rmse": np.nan,
                "r2": np.nan,
                "kendall_tau": np.nan,
                "target_std": np.nan,
                "target_iqr": iqrs[target_id],
                "best_alpha": np.nan,
                "cv_mae": np.nan,
            }
        )

    # Clean Delta^xTB_ref,BoE GemNet run is stored separately from the older
    # aggregate training table, whose xTB residual row predates this target panel.
    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        hit = delta[delta["label"].astype(str).eq("GemNet(Delta^xTB)")]
        if not hit.empty:
            target_id = "deltaxtb"
            rec = hit.iloc[0]
            rows = [row for row in rows if row["target_id"] != target_id]
            rows.append(
                {
                    "target_id": target_id,
                    "target_label": target_meta.loc[target_id, "target_label"],
                    "split": "val",
                    "n": np.nan,
                    "mae": float(rec["best_val_mae_Ha"]),
                    "mae_over_iqr": float(rec["best_val_mae_Ha"]) / iqrs[target_id],
                    "rmse": np.nan,
                    "r2": np.nan,
                    "kendall_tau": np.nan,
                    "target_std": float(rec["target_stdev_Ha"]),
                    "target_iqr": iqrs[target_id],
                    "best_alpha": np.nan,
                    "cv_mae": np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()
    return normalize_metric_rows(pd.DataFrame(rows), "GemNet", "training_log_best_val")


def load_schnet(path: Path, iqrs: dict[str, float]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    target_meta = target_frame()[["target_id", "target_label"]].set_index("target_id")
    rows = []
    df = pd.read_csv(path)
    for _, rec in df.iterrows():
        target_id = SCHNET_TARGET_MAP.get(str(rec.get("model", "")))
        if target_id is None or pd.isna(rec.get("best_val_mae")):
            continue
        rows.append(
            {
                "target_id": target_id,
                "target_label": target_meta.loc[target_id, "target_label"],
                "split": "val",
                "n": np.nan,
                "mae": float(rec["best_val_mae"]),
                "mae_over_iqr": float(rec["best_val_mae"]) / iqrs[target_id],
                "rmse": np.nan,
                "r2": np.nan,
                "kendall_tau": np.nan,
                "target_std": np.nan,
                "target_iqr": iqrs[target_id],
                "best_alpha": np.nan,
                "cv_mae": np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return normalize_metric_rows(pd.DataFrame(rows), "SchNet", "training_log_best_val")


def load_expanded_gnn(path: Path, iqrs: dict[str, float]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    target_meta = target_frame()[["target_id", "target_label"]].set_index("target_id")
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()

    rows = []
    ok = df[df["status"].astype(str).eq("ok")].copy()
    for _, rec in ok.iterrows():
        model_id = str(rec.get("model_id", ""))
        target_id = str(rec.get("target_id", ""))
        if model_id not in {"GemNet", "SchNet"} or target_id not in iqrs:
            continue
        rows.append(
            {
                "model_id": model_id,
                "target_id": target_id,
                "target_label": target_meta.loc[target_id, "target_label"],
                "metric_scope": "training_log_best_val_expanded",
                "split": "val",
                "n": np.nan,
                "mae": float(rec["best_val_mae"]),
                "mae_over_iqr": float(rec["best_val_mae"]) / iqrs[target_id],
                "rmse": np.nan,
                "r2": np.nan,
                "kendall_tau": np.nan,
                "target_std": np.nan,
                "target_iqr": iqrs[target_id],
                "best_alpha": np.nan,
                "cv_mae": np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def completed_rows(args: argparse.Namespace) -> pd.DataFrame:
    iqrs = val_target_iqrs()
    pieces = [
        load_bob(args.bob_metrics),
        load_pre_original(args.pre_original_filtered),
        load_pre_expanded(args.pre_expanded_metrics),
        load_pre_plus_a(args.pre_plus_a_metrics),
        load_gemnet(args.gemnet_metrics, args.gemnet_delta_metrics, iqrs),
        load_schnet(args.schnet_metrics, iqrs),
        load_expanded_gnn(args.expanded_gnn_metrics, iqrs),
    ]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def status_grid(completed: pd.DataFrame) -> pd.DataFrame:
    targets = target_frame()
    grid = targets.assign(_key=1).merge(pd.DataFrame({"model_id": MODEL_ORDER, "_key": 1}), on="_key").drop(columns="_key")
    if completed.empty:
        grid["status"] = "missing"
        return grid
    val = completed[["target_id", "model_id", "mae", "mae_over_iqr", "metric_scope"]].copy()
    out = grid.merge(val, on=["target_id", "model_id"], how="left", validate="one_to_one")
    out["status"] = np.where(out["mae"].notna(), "complete", "missing")
    return out


def write_markdown(status: pd.DataFrame, out_path: Path) -> None:
    summary = (
        status.groupby("model_id")["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(MODEL_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    lines = [
        "# Expanded Panel Status",
        "",
        "This file is generated by `build_panel_status.py`.",
        "",
        "## Completion Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Missing Rows",
        "",
    ]
    missing = status[status["status"].eq("missing")][["model_id", "target_id", "target_label"]]
    if missing.empty:
        lines.append("No missing rows.")
    else:
        lines.append(missing.to_markdown(index=False))
    lines.append("")
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "New_Analysis/expanded_target_panel/outputs/status")
    parser.add_argument(
        "--bob-metrics",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/ridge/bob_target_metrics.csv",
    )
    parser.add_argument(
        "--pre-original-filtered",
        type=Path,
        default=ROOT / "New_Analysis/pre_predict_targets/outputs/pre_target_metrics_shared_filtered_val.csv",
    )
    parser.add_argument(
        "--pre-expanded-metrics",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/ridge/pre_target_metrics.csv",
    )
    parser.add_argument(
        "--pre-plus-a-metrics",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_target_metrics.csv",
    )
    parser.add_argument(
        "--gemnet-metrics",
        type=Path,
        default=ROOT / "New_Analysis/outputs/gemnet_trainings_tmqm_val_mae.csv",
    )
    parser.add_argument(
        "--gemnet-delta-metrics",
        type=Path,
        default=ROOT / "New_Analysis/outputs/gemnet_ref_boe_delta_targets_val_mae_stdev.csv",
    )
    parser.add_argument(
        "--schnet-metrics",
        type=Path,
        default=ROOT / "New_Analysis/schnet_runs/trillium/schnet_tmqm_val_mae_summary_all.csv",
    )
    parser.add_argument(
        "--expanded-gnn-metrics",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/gnn/expanded_gnn_val_mae_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_rows(args)
    status = status_grid(completed)

    completed_path = args.out_dir / "completed_model_target_val_metrics.csv"
    status_path = args.out_dir / "model_target_completion_grid.csv"
    summary_path = args.out_dir / "model_target_completion_summary.md"

    completed.to_csv(completed_path, index=False)
    status.to_csv(status_path, index=False)
    write_markdown(status, summary_path)

    print(f"WROTE: {completed_path}")
    print(f"WROTE: {status_path}")
    print(f"WROTE: {summary_path}")
    print(status.groupby("model_id")["status"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
