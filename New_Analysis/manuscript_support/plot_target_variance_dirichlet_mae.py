#!/usr/bin/env python3
"""Target IQR vs IQR-normalized Dirichlet roughness colored by GNN MAE."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from expanded_panel_common import DEFAULT_SPLIT_DIR, load_feature_space, load_split_ids  # noqa: E402
from run_expanded_smoothness import align_features, dirichlet_metrics, unique_undirected_edges  # noqa: E402


OUT_DIR = ROOT / "New_Analysis/manuscript_support/figures"
TABLE_DIR = ROOT / "New_Analysis/manuscript_support/tables"
OUT_CSV = TABLE_DIR / "target_variance_dirichlet_gnn_mae_points.csv"

GEMNET_OUT = OUT_DIR / "target_iqr_pre_plus_a_dirichlet_gemnet_mae.png"
SCHNET_OUT = OUT_DIR / "target_iqr_pre_dirichlet_schnet_mae.png"
BIPANEL_OUT = OUT_DIR / "target_iqr_dirichlet_gnn_mae_bipanel.png"
GEMNET_VAR_NORM_OUT = OUT_DIR / "target_iqr_pre_plus_a_variance_norm_dirichlet_gemnet_mae.png"
SCHNET_VAR_NORM_OUT = OUT_DIR / "target_iqr_pre_variance_norm_dirichlet_schnet_mae.png"
BIPANEL_VAR_NORM_OUT = OUT_DIR / "target_iqr_variance_norm_dirichlet_gnn_mae_bipanel.png"

TARGET_COLORS = {
    r"$E_{\mathrm{ref}}$": "#374151",
    r"$E_{\mathrm{ref},\mathrm{BoE}}$": "#2563EB",
    r"$\Delta^{xTB}$": "#4C78A8",
    r"$\Delta^{BoB}$": "#F58518",
    r"$\Delta^{PRE}$": "#54A24B",
    r"$\Delta^{PRE+A}$": "#B279A2",
}

MAE_COLORS = ["#D8B70A", "#2A9D8F", "#3A6EA5", "#7B4EA3", "#8C1D40"]


@dataclass(frozen=True)
class PlotTarget:
    target_id: str
    label: str
    csv_path: Path
    column: str


TARGETS = [
    PlotTarget(
        "dft_ref_boe",
        r"$E_{\mathrm{ref},\mathrm{BoE}}$",
        ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv",
        "DFT_ref-BoE",
    ),
    PlotTarget("deltaxtb", r"$\Delta^{xTB}$", ROOT / "New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv", "Delta_xTB-ref-BoE"),
    PlotTarget(
        "delta_mlp_bob",
        r"$\Delta^{BoB}$",
        ROOT / "New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
        "Delta_MLP_BoB-ref-BoE",
    ),
    PlotTarget(
        "delta_mlp_pre",
        r"$\Delta^{PRE}$",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
        "Delta_MLP_PRE-ref-BoE",
    ),
    PlotTarget(
        "delta_mlp_pre_plus_a",
        r"$\Delta^{PRE+A}$",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
        "Delta_MLP_PRE_plus_A-ref-BoE",
    ),
]

GEMNET_MAE = {
    "dft_ref_boe": 0.0101,
    "deltaxtb": 0.0064,
    "delta_mlp_bob": 0.0174,
    "delta_mlp_pre": 0.0181,
    "delta_mlp_pre_plus_a": 0.0168,
}

SCHNET_MAE = {
    "dft_ref_boe": 0.0153,
    "deltaxtb": 0.0093,
    "delta_mlp_bob": 0.0179,
    "delta_mlp_pre": 0.0182,
    "delta_mlp_pre_plus_a": 0.0167,
}


def load_target_panel() -> pd.DataFrame:
    split_df = load_split_ids(DEFAULT_SPLIT_DIR)
    val = split_df[split_df["split"].eq("val")][["mol_id", "split"]].copy()
    panel = val.copy()
    for target in TARGETS:
        df = pd.read_csv(target.csv_path, usecols=["mol_id", target.column]).dropna().drop_duplicates("mol_id")
        df["mol_id"] = df["mol_id"].astype(str)
        df = df.rename(columns={target.column: target.target_id})
        panel = panel.merge(df, on="mol_id", how="inner", validate="one_to_one")
    return panel.reset_index(drop=True)


def build_edges(x, feature_space: str, k: int = 10) -> np.ndarray:
    cache = TABLE_DIR / f"{feature_space}_raw_k{k}_target_variance_dirichlet_edges.npz"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        if int(z["n_rows"]) == x.shape[0] and int(z["n_cols"]) == x.shape[1] and int(z["k"]) == k:
            return z["edges"]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    edges = unique_undirected_edges(ind[:, 1:])
    np.savez_compressed(cache, edges=edges, n_rows=x.shape[0], n_cols=x.shape[1], k=k)
    return edges


def compute_rows(panel: pd.DataFrame, feature_space: str, model_id: str, mae_map: dict[str, float]) -> list[dict[str, object]]:
    x_all, feature_ids, feature_meta = load_feature_space(feature_space)
    x, aligned = align_features(x_all, feature_ids, panel)
    edges = build_edges(x, feature_space=feature_space, k=10)
    rows = []
    for target in TARGETS:
        y = aligned[target.target_id].to_numpy(float)
        metrics = dirichlet_metrics(y, edges)
        rows.append(
            {
                "model_id": model_id,
                "feature_space": feature_meta["feature_space"],
                "feature_scaling": "raw",
                "target_id": target.target_id,
                "target_label": target.label,
                "n_molecules": int(len(y)),
                "target_variance": float(np.var(y, ddof=0)),
                "target_std": float(np.std(y, ddof=0)),
                "target_iqr": metrics["target_iqr"],
                "target_iqr_mha": metrics["target_iqr"] * 1000.0,
                "normalized_dirichlet": metrics["edge_avg_norm_by_population_var"],
                "iqr_normalized_dirichlet": metrics["edge_rmse_over_iqr"] ** 2,
                "edge_rmse": metrics["edge_rmse"],
                "edge_rmse_over_iqr": metrics["edge_rmse_over_iqr"],
                "gnn_val_mae": mae_map.get(target.target_id, np.nan),
            }
        )
    return rows


def annotate_offsets(model_id: str) -> dict[str, tuple[float, float]]:
    if model_id == "GemNet":
        return {
            "dft_ref_boe": (-12, -16),
            "deltaxtb": (8, 6),
            "delta_mlp_bob": (9, 8),
            "delta_mlp_pre": (16, 9),
            "delta_mlp_pre_plus_a": (7, -12),
        }
    return {
        "dft_ref_boe": (-12, -16),
        "deltaxtb": (8, 6),
        "delta_mlp_bob": (8, 8),
        "delta_mlp_pre": (8, -16),
        "delta_mlp_pre_plus_a": (8, -12),
    }


def discrete_mae_scale(values: pd.Series) -> tuple[ListedColormap, BoundaryNorm, list[float]]:
    ticks = sorted(float(v) for v in values.dropna().unique())
    if len(ticks) < 2:
        width = max(abs(ticks[0]) * 0.01, 1e-6)
        boundaries = [ticks[0] - width, ticks[0] + width]
    else:
        mids = [(left + right) / 2.0 for left, right in zip(ticks[:-1], ticks[1:], strict=True)]
        boundaries = [ticks[0] - (mids[0] - ticks[0]), *mids, ticks[-1] + (ticks[-1] - mids[-1])]
    cmap = ListedColormap(MAE_COLORS[: len(ticks)])
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, ticks


def draw_panel(
    ax,
    df: pd.DataFrame,
    model_id: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    y_col: str,
    y_label: str,
):
    sub = df[df["model_id"].eq(model_id)].copy()
    has_mae = sub["gnn_val_mae"].notna()

    scatter = ax.scatter(
        sub.loc[has_mae, "target_iqr_mha"],
        sub.loc[has_mae, y_col],
        c=sub.loc[has_mae, "gnn_val_mae"],
        cmap=cmap,
        norm=norm,
        s=58,
        edgecolor="#111827",
        linewidth=0.7,
        zorder=3,
    )
    missing = sub[~has_mae]
    if not missing.empty:
        ax.scatter(
            missing["target_iqr_mha"],
            missing[y_col],
            s=58,
            marker="x",
            color="#6B7280",
            linewidth=1.6,
            zorder=4,
            label="MAE not available",
        )

    offsets = annotate_offsets(model_id)
    for row in sub.itertuples(index=False):
        dx, dy = offsets.get(row.target_id, (6, 4))
        ax.annotate(
            row.target_label,
            (row.target_iqr_mha, getattr(row, y_col)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.7,
            color=TARGET_COLORS.get(row.target_label, "#111827"),
            clip_on=True,
        )

    ax.set_xscale("log")
    ax.margins(x=0.05, y=0.10)
    ax.xaxis.set_major_locator(FixedLocator([20, 30, 50, 80]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):d}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(15, 92)
    ax.set_xlabel(r"Target IQR (mHa)", fontsize=8.4)
    ax.set_ylabel(y_label, fontsize=8.4)
    ax.tick_params(axis="both", labelsize=7.4)
    ax.grid(color="#E5E7EB", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.98,
        model_id,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        color="#111827",
    )
    return scatter


def plot_panel(df: pd.DataFrame, model_id: str, out_path: Path, y_col: str, y_label: str) -> None:
    sub = df[df["model_id"].eq(model_id)]
    cmap, norm, ticks = discrete_mae_scale(sub["gnn_val_mae"])

    fig, ax = plt.subplots(figsize=(5.65, 3.75), constrained_layout=True)
    scatter = draw_panel(ax, df, model_id, cmap, norm, y_col, y_label)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.012, ticks=ticks)
    cbar.set_label(f"{model_id} validation MAE (Ha)", fontsize=7.9)
    cbar.ax.set_yticklabels([f"{tick:.4f}" for tick in ticks])
    cbar.ax.tick_params(labelsize=7.1)
    if df[df["model_id"].eq(model_id)]["gnn_val_mae"].isna().any():
        ax.legend(frameon=False, loc="lower left", fontsize=6.8, handletextpad=0.4)

    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_bipanel(df: pd.DataFrame, out_path: Path, y_col: str, y_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.20, top=0.91, wspace=0.44)

    for ax, model_id, panel_label in zip(axes, ["GemNet", "SchNet"], ["a", "b"], strict=True):
        sub = df[df["model_id"].eq(model_id)]
        cmap, norm, ticks = discrete_mae_scale(sub["gnn_val_mae"])
        scatter = draw_panel(ax, df, model_id, cmap, norm, y_col, y_label)
        ax.text(
            -0.12,
            1.03,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.018, fraction=0.045, ticks=ticks)
        cbar.set_label(f"{model_id} validation MAE (Ha)", fontsize=7.8)
        cbar.ax.set_yticklabels([f"{tick:.4f}" for tick in ticks])
        cbar.ax.tick_params(labelsize=7.0)

    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_target_panel()
    rows = []
    rows.extend(compute_rows(panel, "pre_plus_a", "GemNet", GEMNET_MAE))
    rows.extend(compute_rows(panel, "pre", "SchNet", SCHNET_MAE))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    iqr_label = r"IQR-normalized Dirichlet roughness, $D_{\mathrm{IQR}}$"
    var_label = r"Variance-normalized Dirichlet roughness, $D_{\mathrm{var}}$"
    plot_bipanel(df, BIPANEL_OUT, "iqr_normalized_dirichlet", iqr_label)
    plot_panel(df, "GemNet", GEMNET_OUT, "iqr_normalized_dirichlet", iqr_label)
    plot_panel(df, "SchNet", SCHNET_OUT, "iqr_normalized_dirichlet", iqr_label)
    plot_bipanel(df, BIPANEL_VAR_NORM_OUT, "normalized_dirichlet", var_label)
    plot_panel(df, "GemNet", GEMNET_VAR_NORM_OUT, "normalized_dirichlet", var_label)
    plot_panel(df, "SchNet", SCHNET_VAR_NORM_OUT, "normalized_dirichlet", var_label)
    print(df.to_string(index=False))
    print(f"WROTE: {OUT_CSV}")
    print(f"WROTE: {BIPANEL_OUT}")
    print(f"WROTE: {GEMNET_OUT}")
    print(f"WROTE: {SCHNET_OUT}")
    print(f"WROTE: {BIPANEL_VAR_NORM_OUT}")
    print(f"WROTE: {GEMNET_VAR_NORM_OUT}")
    print(f"WROTE: {SCHNET_VAR_NORM_OUT}")


if __name__ == "__main__":
    main()
