#!/usr/bin/env python3
"""Compare linear vs MLP baseline residual MAE before and after SchNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASELINE_NAME = {
    "linear BoB": ("BoB", "Linear"),
    "MLP BoB": ("BoB", "MLP"),
    "linear PRE": ("PRE", "Linear"),
    "MLP PRE": ("PRE", "MLP"),
    "linear PRE+A": ("PRE+A", "Linear"),
    "MLP PRE+A": ("PRE+A", "MLP"),
}

SCHNET_LINEAR_TARGET = {
    "BoB": ("Delta^BoB_ref,BoE", "deltabob"),
    "PRE": ("Delta^PRE_ref,BoE", "deltapre"),
    "PRE+A": ("Delta^PRE+A_ref,BoE", "deltapre_plus_a"),
}

SCHNET_MLP_TARGET = {
    "BoB": "delta_mlp_bob",
    "PRE": "delta_mlp_pre",
    "PRE+A": "delta_mlp_pre_plus_a",
}

MODEL_ORDER = ["BoB", "PRE", "PRE+A"]
FAMILY_ORDER = ["Linear", "MLP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=ROOT / "New_Analysis/mlp_baselines/outputs/mlp_vs_linear_baseline_summary.csv",
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/comparison_plots",
    )
    return parser.parse_args()


def load_linear_schnet(core_path: Path, expanded_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}

    core = pd.read_csv(core_path)
    for _, rec in core.iterrows():
        label = str(rec.get("model", ""))
        for baseline, (target_label, _target_id) in SCHNET_LINEAR_TARGET.items():
            if target_label in label and pd.notna(rec.get("best_val_mae")):
                out[baseline] = float(rec["best_val_mae"])

    expanded = pd.read_csv(expanded_path)
    expanded = expanded[(expanded["model_id"].eq("SchNet")) & (expanded["status"].eq("ok"))]
    for baseline, (_target_label, target_id) in SCHNET_LINEAR_TARGET.items():
        hit = expanded[expanded["target_id"].eq(target_id)]
        if len(hit):
            out[baseline] = float(hit.iloc[0]["best_val_mae"])
    return out


def load_mlp_schnet(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[(df["model_id"].eq("SchNet")) & (df["status"].eq("ok"))]
    out: dict[str, float] = {}
    for baseline, target_id in SCHNET_MLP_TARGET.items():
        hit = df[df["target_id"].eq(target_id)]
        if len(hit):
            out[baseline] = float(hit.iloc[0]["best_val_mae"])
    return out


def build_summary(args: argparse.Namespace) -> pd.DataFrame:
    baseline = pd.read_csv(args.baseline_summary)
    linear_schnet = load_linear_schnet(args.schnet_core, args.schnet_expanded)
    mlp_schnet = load_mlp_schnet(args.schnet_mlp)

    rows = []
    for _, rec in baseline.iterrows():
        parsed = BASELINE_NAME.get(str(rec["model"]))
        if parsed is None:
            continue
        baseline_name, fit_family = parsed
        schnet_after = linear_schnet.get(baseline_name) if fit_family == "Linear" else mlp_schnet.get(baseline_name)
        if schnet_after is None or not np.isfinite(schnet_after):
            continue
        before = float(rec["mae"])
        rows.append(
            {
                "baseline": baseline_name,
                "fit_family": fit_family,
                "mask": rec["mask"],
                "baseline_val_mae": before,
                "schnet_residual_val_mae": schnet_after,
                "delta_mae": schnet_after - before,
                "relative_change": (schnet_after - before) / before if before > 0 else np.nan,
                "improvement_percent": 100.0 * (before - schnet_after) / before if before > 0 else np.nan,
                "baseline_r2": float(rec["r2"]),
                "baseline_tau": float(rec["tau"]),
                "baseline_max_abs": float(rec["max_abs"]),
                "baseline_gt1_count": int(rec["gt1"]),
                "baseline_gt10_count": int(rec["gt10"]),
                "target_iqr": float(rec["target_iqr"]),
            }
        )
    return pd.DataFrame(rows)


def draw_cleveland(df: pd.DataFrame, out_png: Path, title: str, log_scale: bool, show_percent: bool = True) -> None:
    plot_df = df.copy()
    plot_df["order_base"] = plot_df["baseline"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    plot_df["order_family"] = plot_df["fit_family"].map({name: i for i, name in enumerate(FAMILY_ORDER)})
    plot_df = plot_df.sort_values(["order_base", "order_family"]).reset_index(drop=True)
    labels = [f"{row.baseline} {row.fit_family}" for row in plot_df.itertuples()]
    y = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    before = plot_df["baseline_val_mae"].to_numpy(float)
    after = plot_df["schnet_residual_val_mae"].to_numpy(float)

    for yi, x0, x1 in zip(y, before, after):
        ax.plot([x0, x1], [yi, yi], color="#9ca3af", lw=2.2, zorder=1)
    ax.scatter(before, y, s=82, color="#ef4444", edgecolor="#111827", linewidth=0.8, label="Baseline residual MAE", zorder=3)
    ax.scatter(after, y, s=82, color="#2563eb", edgecolor="#111827", linewidth=0.8, label="SchNet residual MAE", zorder=4)

    if show_percent:
        for yi, x0, x1, imp in zip(y, before, after, plot_df["improvement_percent"].to_numpy(float)):
            x_text = max(x0, x1) * (1.05 if log_scale else 1.02)
            ax.text(x_text, yi, f"{imp:+.0f}%", va="center", fontsize=9, color="#374151")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Validation MAE (Ha)")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Validation MAE (Ha, log scale)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def make_pre_filtered_exception(summary: pd.DataFrame) -> pd.DataFrame:
    """Use robust PRE baseline values while keeping other rows on all-val MAE."""
    all_val = summary[summary["mask"].eq("all val")].copy()
    shared = summary[summary["mask"].isin(["shared mask", "shared outlier mask"])].copy()
    if all_val.empty or shared.empty:
        return all_val

    out = all_val.copy()
    for fit_family in FAMILY_ORDER:
        replace = shared[(shared["baseline"].eq("PRE")) & (shared["fit_family"].eq(fit_family))]
        if replace.empty:
            continue
        idx = out[(out["baseline"].eq("PRE")) & (out["fit_family"].eq(fit_family))].index
        if len(idx) != 1:
            continue
        keep_mask_label = "all val, PRE uses shared-filtered baseline MAE"
        for col in [
            "baseline_val_mae",
            "delta_mae",
            "relative_change",
            "improvement_percent",
            "baseline_r2",
            "baseline_tau",
            "baseline_max_abs",
            "baseline_gt1_count",
            "baseline_gt10_count",
            "target_iqr",
        ]:
            out.loc[idx, col] = replace.iloc[0][col]
        out.loc[idx, "mask"] = keep_mask_label
    out.loc[~out["baseline"].eq("PRE"), "mask"] = "all val"
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    out_csv = args.out_dir / "linear_vs_mlp_residual_schnet_mae_summary.csv"
    summary.to_csv(out_csv, index=False)

    for mask, suffix, title, log_scale in [
        (
            "all val",
            "all_val_log",
            "Baseline residual MAE before and after SchNet (all validation rows)",
            True,
        ),
        (
            "shared mask",
            "shared_mask",
            "Baseline residual MAE before and after SchNet (shared outlier mask)",
            False,
        ),
    ]:
        sub = summary[summary["mask"].eq(mask)].copy()
        if not sub.empty:
            draw_cleveland(
                sub,
                args.out_dir / f"linear_vs_mlp_residual_schnet_cleveland_{suffix}.png",
                title,
                log_scale=log_scale,
            )

    hybrid = make_pre_filtered_exception(summary)
    if not hybrid.empty:
        hybrid_csv = args.out_dir / "linear_vs_mlp_residual_schnet_mae_summary_pre_filtered_exception.csv"
        hybrid.to_csv(hybrid_csv, index=False)
        draw_cleveland(
            hybrid,
            args.out_dir / "linear_vs_mlp_residual_schnet_cleveland_pre_filtered_exception.png",
            "Baseline residual MAE before and after SchNet (PRE uses filtered baseline MAE)",
            log_scale=False,
        )

    shared_all = summary[summary["mask"].isin(["shared mask", "shared outlier mask"])].copy()
    if not shared_all.empty:
        shared_all_csv = args.out_dir / "linear_vs_mlp_residual_schnet_mae_summary_shared_mask_only.csv"
        shared_all.to_csv(shared_all_csv, index=False)
        draw_cleveland(
            shared_all,
            args.out_dir / "linear_vs_mlp_residual_schnet_cleveland_shared_mask_no_percent.png",
            "Baseline residual MAE before and after SchNet (shared outlier mask)",
            log_scale=False,
            show_percent=False,
        )

    print(summary.to_string(index=False))
    print(f"WROTE: {out_csv}")
    print(f"WROTE: {args.out_dir / 'linear_vs_mlp_residual_schnet_cleveland_all_val_log.png'}")
    print(f"WROTE: {args.out_dir / 'linear_vs_mlp_residual_schnet_cleveland_shared_mask.png'}")
    if not hybrid.empty:
        print(f"WROTE: {hybrid_csv}")
        print(f"WROTE: {args.out_dir / 'linear_vs_mlp_residual_schnet_cleveland_pre_filtered_exception.png'}")
    if not shared_all.empty:
        print(f"WROTE: {shared_all_csv}")
        print(f"WROTE: {args.out_dir / 'linear_vs_mlp_residual_schnet_cleveland_shared_mask_no_percent.png'}")


if __name__ == "__main__":
    main()
