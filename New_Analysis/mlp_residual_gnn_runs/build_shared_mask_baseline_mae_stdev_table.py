#!/usr/bin/env python3
"""Build shared-mask MAE/residual-std table for linear and MLP baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SPECS = [
    (
        "linear BoB",
        ROOT / "New_Analysis/bob_predict_targets/outputs/bob_predictions_dft.csv",
        "target",
        "bob_pred",
    ),
    (
        "MLP BoB",
        ROOT / "New_Analysis/mlp_baselines/outputs/bob/mlp_bob_predictions_dft.csv",
        "DFT_ref-BoE",
        "MLP_BoB_ref-BoE",
    ),
    (
        "linear PRE",
        ROOT / "New_Analysis/pre_predict_targets/outputs/pre_predictions_dft.csv",
        "target",
        "pre_pred",
    ),
    (
        "MLP PRE",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre/mlp_pre_predictions_dft.csv",
        "DFT_ref-BoE",
        "MLP_PRE_ref-BoE",
    ),
    (
        "linear PRE+A",
        ROOT / "New_Analysis/pre_plus_a/outputs_high_alpha_cv/pre_plus_a_predictions_dft_ref_boe.csv",
        "target",
        "pre_plus_a_pred",
    ),
    (
        "MLP PRE+A",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/mlp_pre_plus_a_predictions_dft.csv",
        "DFT_ref-BoE",
        "MLP_PRE_plus_A_ref-BoE",
    ),
]

SCHNET_BY_MODEL = {
    "linear BoB": ("core", "Delta^BoB_ref,BoE"),
    "linear PRE": ("core", "Delta^PRE_ref,BoE"),
    "linear PRE+A": ("expanded", "deltapre_plus_a"),
    "MLP BoB": ("mlp", "delta_mlp_bob"),
    "MLP PRE": ("mlp", "delta_mlp_pre"),
    "MLP PRE+A": ("mlp", "delta_mlp_pre_plus_a"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-mask",
        type=Path,
        default=ROOT / "New_Analysis/split_baseline_val_mae/outputs_dft_ref_boe_only/shared_outlier_mask.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/comparison_plots",
    )
    parser.add_argument(
        "--schnet-core",
        type=Path,
        default=ROOT / "New_Analysis/schnet_runs/trillium/schnet_tmqm_val_mae_summary_all.csv",
    )
    parser.add_argument(
        "--schnet-expanded",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/gnn/expanded_gnn_val_mae_summary.csv",
    )
    parser.add_argument(
        "--schnet-mlp",
        type=Path,
        default=ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary.csv",
    )
    return parser.parse_args()


def read_kept_ids(path: Path) -> set[str]:
    mask = pd.read_csv(path)
    return set(mask.loc[~mask["remove_shared_outlier"], "mol_id"].astype(str))


def compute_row(model: str, path: Path, target_col: str, pred_col: str, kept_ids: set[str]) -> dict[str, object]:
    df = pd.read_csv(path)
    df = df[(df["split"].eq("val")) & (df["mol_id"].astype(str).isin(kept_ids))].copy()
    residual = df[target_col].to_numpy(float) - df[pred_col].to_numpy(float)
    return {
        "model": model,
        "n": len(df),
        "mae": float(np.mean(np.abs(residual))),
        "residual_stdev": float(np.std(residual, ddof=0)),
    }


def load_schnet_maes(core_path: Path, expanded_path: Path, mlp_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}

    core = pd.read_csv(core_path)
    for model, (source, key) in SCHNET_BY_MODEL.items():
        if source != "core":
            continue
        hit = core[core["model"].astype(str).str.contains(key, regex=False, na=False)]
        if len(hit):
            out[model] = float(hit.iloc[0]["best_val_mae"])

    expanded = pd.read_csv(expanded_path)
    expanded = expanded[(expanded["model_id"].eq("SchNet")) & (expanded["status"].eq("ok"))]
    for model, (source, key) in SCHNET_BY_MODEL.items():
        if source != "expanded":
            continue
        hit = expanded[expanded["target_id"].eq(key)]
        if len(hit):
            out[model] = float(hit.iloc[0]["best_val_mae"])

    mlp = pd.read_csv(mlp_path)
    mlp = mlp[(mlp["model_id"].eq("SchNet")) & (mlp["status"].eq("ok"))]
    for model, (source, key) in SCHNET_BY_MODEL.items():
        if source != "mlp":
            continue
        hit = mlp[mlp["target_id"].eq(key)]
        if len(hit):
            out[model] = float(hit.iloc[0]["best_val_mae"])

    return out


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Shared Outlier Mask",
        "",
        "| model | MAE | residual stdev | SchNet MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(f"| {row.model} | {row.mae:.5f} | {row.residual_stdev:.5f} | {row.schnet_mae:.5f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_table(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.axis("off")
    ax.set_title("Shared Outlier Mask", loc="left", fontsize=16, fontweight="bold", pad=12)
    display = [[row.model, f"{row.mae:.5f}", f"{row.residual_stdev:.5f}", f"{row.schnet_mae:.5f}"] for row in df.itertuples(index=False)]
    table = ax.table(
        cellText=display,
        colLabels=["model", "MAE", "residual stdev", "SchNet MAE"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept_ids = read_kept_ids(args.shared_mask)
    df = pd.DataFrame([compute_row(*spec, kept_ids) for spec in SPECS])
    schnet = load_schnet_maes(args.schnet_core, args.schnet_expanded, args.schnet_mlp)
    df["schnet_mae"] = df["model"].map(schnet)

    csv_path = args.out_dir / "shared_mask_baseline_mae_residual_stdev_table.csv"
    md_path = args.out_dir / "shared_mask_baseline_mae_residual_stdev_table.md"
    png_path = args.out_dir / "shared_mask_baseline_mae_residual_stdev_table.png"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    draw_table(df, png_path)

    print(df.to_string(index=False))
    print(f"WROTE: {csv_path}")
    print(f"WROTE: {md_path}")
    print(f"WROTE: {png_path}")


if __name__ == "__main__":
    main()
