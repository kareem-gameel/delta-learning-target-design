#!/usr/bin/env python3
"""Plot baseline-vs-DFT_ref,BoE parity density panels for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "New_Analysis/datasets"
SPLIT_PATH = ROOT / "trillium_project/split_ids/tmqm_val_ids.csv"
OUT_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"


BASELINES = [
    {
        "id": "bob",
        "label": "BoB",
        "path": ROOT / "New_Analysis/bob_predict_targets/outputs/bob_predictions_dft.csv",
        "column": "bob_pred",
        "split_column": "split",
    },
    {
        "id": "pre",
        "label": "PRE",
        "path": ROOT / "New_Analysis/pre_predict_targets/outputs/pre_predictions_dft.csv",
        "column": "pre_pred",
        "split_column": "split",
    },
    {
        "id": "pre_plus_a",
        "label": "PRE+A",
        "path": ROOT / "New_Analysis/pre_plus_a/outputs_high_alpha_cv/pre_plus_a_predictions_dft_ref_boe.csv",
        "column": "pre_plus_a_pred",
        "split_column": "split",
    },
    {
        "id": "xtb_ref_boe",
        "label": "xTB_ref,BoE",
        "path": DATASET_DIR / "tmQM__xTB_ref-BoE.csv",
        "column": "xTB_ref-BoE",
    },
]


def load_validation_targets() -> pd.DataFrame:
    dft = pd.read_csv(DATASET_DIR / "tmQM__DFT_ref-BoE.csv", usecols=["mol_id", "DFT_ref-BoE"])
    val_ids = pd.read_csv(SPLIT_PATH, usecols=["mol_id"])
    mask_path = ROOT / "New_Analysis/split_baseline_val_mae/outputs_dft_ref_boe_only/shared_outlier_mask.csv"
    shared_mask = pd.read_csv(mask_path, usecols=["mol_id", "remove_shared_outlier"])
    shared_mask = shared_mask.loc[~shared_mask["remove_shared_outlier"], ["mol_id"]]
    val_ids["mol_id"] = val_ids["mol_id"].astype(str)
    dft["mol_id"] = dft["mol_id"].astype(str)
    shared_mask["mol_id"] = shared_mask["mol_id"].astype(str)
    return (
        val_ids.merge(shared_mask, on="mol_id", how="inner", validate="one_to_one")
        .merge(dft, on="mol_id", how="inner", validate="one_to_one")
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic),
        "r2": float(1.0 - ss_res / ss_tot),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [0.2, 99.8])
    pad = 0.04 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dft = load_validation_targets()
    panels: list[dict[str, object]] = []
    all_values = [dft["DFT_ref-BoE"].to_numpy(float)]

    for spec in BASELINES:
        baseline = pd.read_csv(spec["path"], usecols=["mol_id", spec["column"]])
        if spec.get("split_column"):
            baseline = pd.read_csv(spec["path"], usecols=["mol_id", spec["split_column"], spec["column"]])
            baseline = baseline.loc[baseline[spec["split_column"]] == "val", ["mol_id", spec["column"]]]
        baseline["mol_id"] = baseline["mol_id"].astype(str)
        merged = dft.merge(baseline, on="mol_id", how="inner", validate="one_to_one")
        y = merged["DFT_ref-BoE"].to_numpy(float)
        x = merged[spec["column"]].to_numpy(float)
        row = {
            "baseline_id": spec["id"],
            "baseline_label": spec["label"],
            **metrics(y, x),
        }
        panels.append({"spec": spec, "data": merged, "metrics": row})
        all_values.append(x)

    panels.sort(key=lambda item: item["metrics"]["pearson_r"])
    metric_table = pd.DataFrame([item["metrics"] for item in panels])
    table_path = TABLE_DIR / "baseline_dft_ref_boe_val_correlations.csv"
    metric_table.to_csv(table_path, index=False)

    lim = robust_limits(np.concatenate(all_values))
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.35), sharex=True, sharey=True, constrained_layout=True)
    cmap = "mako" if "mako" in plt.colormaps() else "viridis"

    for ax, item in zip(axes, panels):
        spec = item["spec"]
        df = item["data"]
        m = item["metrics"]
        x = df[spec["column"]].to_numpy(float)
        y = df["DFT_ref-BoE"].to_numpy(float)

        hb = ax.hexbin(
            x,
            y,
            gridsize=58,
            extent=(lim[0], lim[1], lim[0], lim[1]),
            mincnt=1,
            bins="log",
            cmap=cmap,
            linewidths=0.0,
        )
        ax.plot(lim, lim, color="black", lw=1.0, ls="--", alpha=0.70)
        ax.set_title(f"{spec['label']}", fontsize=11, fontweight="bold")
        ax.text(
            0.04,
            0.96,
            f"r = {m['pearson_r']:.3f}\nMAE = {m['mae']:.4f}\nRMSE = {m['rmse']:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        ax.grid(True, color="0.90", lw=0.6)

    axes[0].set_ylabel("DFT_ref,BoE (Ha)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Baseline target (Ha)", fontsize=10)

    cbar = fig.colorbar(hb, ax=axes, shrink=0.82, pad=0.012)
    cbar.set_label("Molecule count per hexbin (log)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Validation-set baseline correlation with DFT_ref,BoE (shared filtered mask)",
        fontsize=13,
        fontweight="bold",
    )
    out_path = OUT_DIR / "baseline_dft_ref_boe_val_correlation_hexbin.png"
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(metric_table.to_string(index=False))
    print(f"WROTE: {table_path}")
    print(f"WROTE: {out_path}")


if __name__ == "__main__":
    main()
