#!/usr/bin/env python3
"""Plot MLP baseline accuracy and residual scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "New_Analysis/manuscript_support/figures/mlp_baseline_dft_ref_boe_mae_residual_iqr.png"
TABLE = ROOT / "New_Analysis/manuscript_support/tables/mlp_baseline_dft_ref_boe_val_metrics.csv"
DATASET_DIR = ROOT / "New_Analysis/datasets"
SPLIT_PATH = ROOT / "trillium_project/split_ids/tmqm_val_ids.csv"

DATA = {
    "xtb": {
        "baseline": "xTB",
        "delta": r"$\Delta^{\mathrm{xTB}}$",
        "color": "#4C78A8",
        "path": DATASET_DIR / "tmQM__xTB_ref-BoE.csv",
        "column": "xTB_ref-BoE",
    },
    "mlp_bob": {
        "baseline": "BoB",
        "delta": r"$\Delta^{\mathrm{BoB}}$",
        "color": "#F58518",
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/bob/mlp_bob_predictions_dft.csv",
        "column": "MLP_BoB_ref-BoE",
        "split_column": "split",
    },
    "mlp_pre": {
        "baseline": "PRE",
        "delta": r"$\Delta^{\mathrm{PRE}}$",
        "color": "#54A24B",
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/pre/mlp_pre_predictions_dft.csv",
        "column": "MLP_PRE_ref-BoE",
        "split_column": "split",
    },
    "mlp_pre_plus_a": {
        "baseline": "PRE+A",
        "delta": r"$\Delta^{\mathrm{PRE+A}}$",
        "color": "#B279A2",
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/mlp_pre_plus_a_predictions_dft.csv",
        "column": "MLP_PRE_plus_A_ref-BoE",
        "split_column": "split",
    },
}


def load_validation_dft() -> pd.DataFrame:
    dft = pd.read_csv(DATASET_DIR / "tmQM__DFT_ref-BoE.csv", usecols=["mol_id", "DFT_ref-BoE"])
    val = pd.read_csv(SPLIT_PATH, usecols=["mol_id"])
    for frame in (dft, val):
        frame["mol_id"] = frame["mol_id"].astype(str)
    return val.merge(dft, on="mol_id", how="inner", validate="one_to_one")


def load_predictions(spec: dict[str, object]) -> pd.DataFrame:
    cols = ["mol_id", spec["column"]]
    if "split_column" in spec:
        cols.append(spec["split_column"])
    pred = pd.read_csv(spec["path"], usecols=cols)
    if "split_column" in spec:
        pred = pred.loc[pred[spec["split_column"]] == "val", ["mol_id", spec["column"]]]
    pred["mol_id"] = pred["mol_id"].astype(str)
    return pred


def build_metrics() -> pd.DataFrame:
    dft = load_validation_dft()
    rows = []
    for baseline_id, spec in DATA.items():
        pred = load_predictions(spec)
        merged = dft.merge(pred, on="mol_id", how="inner", validate="one_to_one")
        y = merged["DFT_ref-BoE"].to_numpy(float)
        yhat = merged[spec["column"]].to_numpy(float)
        err = y - yhat
        rows.append(
            {
                "baseline_id": baseline_id,
                "baseline_label": spec["baseline"],
                "delta_label": spec["delta"],
                "color": spec["color"],
                "n": int(len(merged)),
                "pearson_r": float(pearsonr(y, yhat).statistic),
                "residual_iqr": float(np.subtract(*np.percentile(err, [75, 25]))),
                "residual_variance": float(np.var(err, ddof=0)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r")


def main() -> None:
    df = build_metrics().sort_values("mae", ascending=False)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["color"]).to_csv(TABLE, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.18, top=0.92, wspace=0.34)

    panels = [
        {
            "ax": axes[0],
            "values": df["mae"].to_numpy(float),
            "labels": df["baseline_label"].tolist(),
            "ylabel": "Validation MAE (Ha)",
            "ylim": (0.0, float(df["mae"].max()) * 1.25),
            "fmt": "{:.4f}",
            "offset": float(df["mae"].max()) * 0.035,
            "letter": "a",
        },
        {
            "ax": axes[1],
            "values": df["residual_iqr"].to_numpy(float),
            "labels": df["delta_label"].tolist(),
            "ylabel": "Residual scale, IQR (Ha)",
            "ylim": (0.0, float(df["residual_iqr"].max()) * 1.22),
            "fmt": "{:.3f}",
            "offset": float(df["residual_iqr"].max()) * 0.025,
            "letter": "b",
        },
    ]

    colors = df["color"].tolist()
    for panel in panels:
        ax = panel["ax"]
        x = np.arange(len(df))
        bars = ax.bar(x, panel["values"], color=colors, edgecolor="#202020", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["labels"], fontsize=9)
        ax.set_ylabel(panel["ylabel"], fontsize=10)
        ax.set_ylim(*panel["ylim"])
        ax.grid(axis="y", color="0.86", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.18,
            1.03,
            f"({panel['letter']})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        for bar, value in zip(bars, panel["values"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + panel["offset"],
                panel["fmt"].format(value),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax_b = axes[1]
    x = np.arange(len(df))
    variance = df["residual_variance"].to_numpy(float)
    variance_color = "#B91C1C"
    ax_var = ax_b.twinx()
    ax_var.plot(
        x,
        variance,
        color=variance_color,
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        zorder=5,
        label="Residual variance",
    )
    ax_var.set_ylabel(r"Residual variance (Ha$^2$)", fontsize=9, color=variance_color)
    ax_var.tick_params(axis="y", labelsize=8, colors=variance_color)
    ax_var.set_ylim(0.0, float(variance.max()) * 1.25)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_color(variance_color)
    for xi, value in zip(x, variance):
        ax_var.text(
            xi,
            value + float(variance.max()) * 0.035,
            f"{value * 1e3:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=variance_color,
            fontweight="bold",
        )
    ax_var.text(
        0.98,
        0.96,
        r"line: variance ($10^{-3}$ Ha$^2$)",
        transform=ax_var.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=variance_color,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(df[["baseline_label", "n", "pearson_r", "residual_iqr", "residual_variance", "mae", "rmse"]].to_string(index=False))
    print(f"WROTE: {TABLE}")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
