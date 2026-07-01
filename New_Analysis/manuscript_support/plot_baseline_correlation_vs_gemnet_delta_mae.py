#!/usr/bin/env python3
"""Scatter: baseline correlation with E_ref,BoE vs GemNet delta-target MAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
BASELINE_METRICS = ROOT / "New_Analysis/manuscript_support/tables/mlp_baseline_dft_ref_boe_val_metrics.csv"
GNN_STATUS = ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv"
MLP_GNN = ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary_all.csv"
OUT = ROOT / "New_Analysis/manuscript_support/figures/baseline_correlation_vs_gemnet_delta_mae.png"
OUT_TABLE = ROOT / "New_Analysis/manuscript_support/tables/baseline_correlation_vs_gemnet_delta_mae.csv"
OUT_RHO = ROOT / "New_Analysis/manuscript_support/figures/baseline_spearman_vs_gemnet_delta_mae.png"
OUT_RHO_TABLE = ROOT / "New_Analysis/manuscript_support/tables/baseline_spearman_vs_gemnet_delta_mae.csv"
DATASET_DIR = ROOT / "New_Analysis/datasets"
SPLIT_PATH = ROOT / "trillium_project/split_ids/tmqm_val_ids.csv"

COLORS = {
    "xTB": "#4C78A8",
    "BoB": "#F58518",
    "PRE": "#54A24B",
    "PRE+A": "#B279A2",
}

DELTA_LABELS = {
    "xTB": r"$\Delta^{xTB}$",
    "BoB": r"$\Delta^{BoB}$",
    "PRE": r"$\Delta^{PRE}$",
    "PRE+A": r"$\Delta^{PRE+A}$",
}

GEMNET_TARGET_IDS = {
    "xTB": "deltaxtb",
    "BoB": "delta_mlp_bob",
    "PRE": "delta_mlp_pre",
    "PRE+A": "delta_mlp_pre_plus_a",
}

BASELINE_PREDICTIONS = {
    "xTB": {
        "path": DATASET_DIR / "tmQM__xTB_ref-BoE.csv",
        "column": "xTB_ref-BoE",
    },
    "BoB": {
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/bob/mlp_bob_predictions_dft.csv",
        "column": "MLP_BoB_ref-BoE",
        "split_column": "split",
    },
    "PRE": {
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/pre/mlp_pre_predictions_dft.csv",
        "column": "MLP_PRE_ref-BoE",
        "split_column": "split",
    },
    "PRE+A": {
        "path": ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/mlp_pre_plus_a_predictions_dft.csv",
        "column": "MLP_PRE_plus_A_ref-BoE",
        "split_column": "split",
    },
}


def load_direct_gemnet_mae() -> float:
    df = pd.read_csv(GNN_STATUS)
    row = df.loc[(df["model_id"] == "GemNet") & (df["target_id"] == "dft")]
    if row.empty:
        raise ValueError(f"Missing direct GemNet E_ref,BoE MAE in {GNN_STATUS}")
    return float(row.iloc[0]["mae"])


def load_gemnet_delta_mae(label: str) -> float:
    target_id = GEMNET_TARGET_IDS[label]
    if target_id == "deltaxtb":
        df = pd.read_csv(GNN_STATUS)
        row = df.loc[(df["model_id"] == "GemNet") & (df["target_id"] == target_id)]
        if row.empty:
            raise ValueError(f"Missing GemNet {target_id} MAE in {GNN_STATUS}")
        return float(row.iloc[0]["mae"])

    df = pd.read_csv(MLP_GNN)
    row = df.loc[(df["model_id"] == "GemNet") & (df["target_id"] == target_id)]
    if row.empty:
        raise ValueError(f"Missing GemNet {target_id} MAE in {MLP_GNN}")
    return float(row.iloc[0]["best_val_mae"])


def load_validation_dft() -> pd.DataFrame:
    dft = pd.read_csv(DATASET_DIR / "tmQM__DFT_ref-BoE.csv", usecols=["mol_id", "DFT_ref-BoE"])
    val = pd.read_csv(SPLIT_PATH, usecols=["mol_id"])
    for frame in (dft, val):
        frame["mol_id"] = frame["mol_id"].astype(str)
    return val.merge(dft, on="mol_id", how="inner", validate="one_to_one")


def load_baseline_prediction(label: str) -> pd.DataFrame:
    spec = BASELINE_PREDICTIONS[label]
    cols = ["mol_id", spec["column"]]
    if "split_column" in spec:
        cols.append(spec["split_column"])
    pred = pd.read_csv(spec["path"], usecols=cols)
    if "split_column" in spec:
        pred = pred.loc[pred[spec["split_column"]].eq("val"), ["mol_id", spec["column"]]]
    pred["mol_id"] = pred["mol_id"].astype(str)
    return pred.rename(columns={spec["column"]: "baseline_pred"})


def baseline_spearman_lookup() -> dict[str, float]:
    truth = load_validation_dft()
    out = {}
    for label in BASELINE_PREDICTIONS:
        pred = load_baseline_prediction(label)
        merged = truth.merge(pred, on="mol_id", how="inner", validate="one_to_one")
        out[label] = float(spearmanr(merged["DFT_ref-BoE"].to_numpy(float), merged["baseline_pred"].to_numpy(float)).statistic)
    return out


def build_table() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_METRICS)
    rho_lookup = baseline_spearman_lookup()
    rows = []
    for row in baseline.to_dict(orient="records"):
        label = row["baseline_label"]
        rows.append(
            {
                "baseline": label,
                "delta_target": DELTA_LABELS[label],
                "baseline_pearson_r_with_dft_ref_boe": float(row["pearson_r"]),
                "baseline_spearman_rho_with_dft_ref_boe": rho_lookup[label],
                "baseline_val_mae": float(row["mae"]),
                "residual_iqr": float(row["residual_iqr"]),
                "residual_variance": float(row["residual_variance"]),
                "gemnet_delta_val_mae": load_gemnet_delta_mae(label),
            }
        )
    return pd.DataFrame(rows).sort_values("baseline_pearson_r_with_dft_ref_boe").reset_index(drop=True)


def plot_scatter(df: pd.DataFrame, x_col: str, x_label: str, out_path: Path, out_table: Path) -> None:
    direct_mae = load_direct_gemnet_mae()
    df = df.sort_values(x_col).reset_index(drop=True)
    x = df[x_col]
    y = df["gemnet_delta_val_mae"]
    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(4.85, 3.55), constrained_layout=True)
    for row in df.to_dict(orient="records"):
        ax.scatter(
            row[x_col],
            row["gemnet_delta_val_mae"],
            s=88,
            color=COLORS[row["baseline"]],
            edgecolor="#202020",
            linewidth=0.85,
            zorder=3,
        )
        dx = 0.004
        dy = 0.00018
        if row["baseline"] == "PRE+A":
            dy = -0.00038
        if row["baseline"] == "PRE":
            dy = 0.00028
        ax.annotate(
            row["delta_target"],
            (row[x_col], row["gemnet_delta_val_mae"]),
            xytext=(dx, dy),
            textcoords="offset fontsize",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLORS[row["baseline"]],
        )

    ax.axhline(direct_mae, color="#1F2933", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        df[x_col].min() + 0.002,
        direct_mae + 0.00018,
        r"GemNet $E_{\mathrm{ref},\mathrm{BoE}}$",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#1F2933",
    )
    ax.text(
        0.03,
        0.97,
        rf"$r={pearson:+.2f}$" + "\n" + rf"$\rho={spearman:+.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#374151",
    )

    ax.set_xlabel(x_label, fontsize=8.8)
    ax.set_ylabel("GemNet delta-target validation MAE (Ha)", fontsize=8.8)
    ax.grid(color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pad = max((float(x.max()) - float(x.min())) * 0.10, 0.01)
    ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
    ax.set_ylim(0.0054, max(float(y.max()), direct_mae) * 1.10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"Pearson r: {pearson:+.4f}")
    print(f"Spearman rho: {spearman:+.4f}")
    print(f"Direct GemNet E_ref,BoE MAE: {direct_mae:.4f}")
    print(f"WROTE: {out_table}")
    print(f"WROTE: {out_path}")


def main() -> None:
    df = build_table()
    plot_scatter(
        df,
        "baseline_pearson_r_with_dft_ref_boe",
        r"Baseline correlation with $E_{\mathrm{ref},\mathrm{BoE}}$ (Pearson $r$)",
        OUT,
        OUT_TABLE,
    )
    plot_scatter(
        df,
        "baseline_spearman_rho_with_dft_ref_boe",
        r"Baseline rank correlation with $E_{\mathrm{ref},\mathrm{BoE}}$ (Spearman $\rho$)",
        OUT_RHO,
        OUT_RHO_TABLE,
    )


if __name__ == "__main__":
    main()
