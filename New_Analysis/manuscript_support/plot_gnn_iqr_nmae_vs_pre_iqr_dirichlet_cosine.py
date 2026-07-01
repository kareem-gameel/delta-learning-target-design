#!/usr/bin/env python3
"""Plot PRE cosine-neighborhood IQR-normalized roughness vs GNN MAE/IQR.

This is the cosine-distance counterpart of
plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet.py. It rebuilds the validation-set
kNN graph in raw PRE feature space using cosine distance and recomputes the
IQR-normalized graph Dirichlet roughness for the same residual target panel.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from expanded_panel_common import DEFAULT_SPLIT_DIR, load_feature_space, load_split_ids  # noqa: E402
from run_expanded_smoothness import dirichlet_metrics, unique_undirected_edges  # noqa: E402

from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet import (  # noqa: E402
    FIG_DIR,
    GEMNET_MAE,
    GEMNET_TARGET_ORDER,
    MARKERS,
    OFFSETS,
    SCHNET_MAE,
    SCHNET_TARGET_ORDER,
    TABLE_DIR,
    TARGET_COLORS,
    TARGET_LABELS,
)


@dataclass(frozen=True)
class TargetSource:
    target_id: str
    csv_path: Path
    column: str


TARGET_SOURCES = [
    TargetSource("dft_ref_boe", ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv", "DFT_ref-BoE"),
    TargetSource("deltaxtb", ROOT / "New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv", "Delta_xTB-ref-BoE"),
    TargetSource(
        "deltabob",
        ROOT / "New_Analysis/datasets/tmQM__Delta_BoB-ref-BoE.csv",
        "E_resid_after_BoE_BoB_tmQM",
    ),
    TargetSource(
        "deltapre",
        ROOT / "New_Analysis/datasets/tmQM__Delta_PRE-ref-BoE.csv",
        "E_resid_after_BoE_PRE_tmQM",
    ),
    TargetSource(
        "deltapre_plus_a",
        ROOT / "New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv",
        "Delta_PRE_plus_A-ref-BoE",
    ),
    TargetSource(
        "delta_mlp_bob",
        ROOT / "New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
        "Delta_MLP_BoB-ref-BoE",
    ),
    TargetSource(
        "delta_mlp_pre",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
        "Delta_MLP_PRE-ref-BoE",
    ),
    TargetSource(
        "delta_mlp_pre_plus_a",
        ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
        "Delta_MLP_PRE_plus_A-ref-BoE",
    ),
]

EDGE_CACHE = TABLE_DIR / "pre_raw_cosine_k10_val_edges.npz"
GEMNET_OUT_TABLE = TABLE_DIR / "gemnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_cosine_linear_mlp.csv"
GEMNET_OUT_FIG = FIG_DIR / "gemnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_cosine_linear_mlp.png"
SCHNET_OUT_TABLE = TABLE_DIR / "schnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_cosine_linear_mlp.csv"
SCHNET_OUT_FIG = FIG_DIR / "schnet_iqr_nmae_vs_pre_iqr_norm_dirichlet_cosine_linear_mlp.png"


def load_target_panel() -> pd.DataFrame:
    split_df = load_split_ids(DEFAULT_SPLIT_DIR)
    frame = split_df[split_df["split"].eq("val")].copy()
    for source in TARGET_SOURCES:
        df = pd.read_csv(source.csv_path, usecols=["mol_id", source.column]).dropna()
        df["mol_id"] = df["mol_id"].astype(str)
        df = df.drop_duplicates("mol_id", keep="first").rename(columns={source.column: source.target_id})
        frame = frame.merge(df, on="mol_id", how="inner", validate="one_to_one")
    return frame.reset_index(drop=True)


def align_pre_features(frame: pd.DataFrame):
    x_all, feature_ids, _feature_meta = load_feature_space("pre")
    rows = pd.DataFrame({"mol_id": feature_ids, "_row": np.arange(len(feature_ids), dtype=int)})
    aligned = frame.merge(rows, on="mol_id", how="inner", validate="one_to_one").reset_index(drop=True)
    x = x_all[aligned["_row"].to_numpy(int)].tocsr()
    return x, aligned.drop(columns=["_row"])


def build_cosine_edges(x, k: int = 10, edge_cache: Path = EDGE_CACHE) -> np.ndarray:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if edge_cache.exists():
        cached = np.load(edge_cache)
        if int(cached["n_rows"]) == x.shape[0] and int(cached["n_features"]) == x.shape[1] and int(cached["k"]) == k:
            return cached["edges"]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    edges = unique_undirected_edges(ind[:, 1:])
    np.savez_compressed(edge_cache, edges=edges, n_rows=x.shape[0], n_features=x.shape[1], k=k)
    return edges


def smoothness_rows(aligned: pd.DataFrame, edges: np.ndarray, mae_map: dict[str, float]) -> pd.DataFrame:
    rows = []
    for source in TARGET_SOURCES:
        target_id = source.target_id
        y = aligned[target_id].to_numpy(float)
        metrics = dirichlet_metrics(y, edges)
        iqr_normalized_dirichlet = float(metrics["edge_rmse_over_iqr"]) ** 2
        target_iqr = float(metrics["target_iqr"])
        rows.append(
            {
                "target_id": target_id,
                "target_label": TARGET_LABELS[target_id],
                "target_variance": float(metrics["target_std"]) ** 2,
                "target_iqr": target_iqr,
                "gnn_val_mae": float(mae_map[target_id]),
                "iqr_normalized_dirichlet": iqr_normalized_dirichlet,
                "gnn_mae_over_iqr": float(mae_map[target_id]) / target_iqr,
            }
        )
    return pd.DataFrame(rows)


def plot_panel(
    df: pd.DataFrame,
    order: list[str],
    model_label: str,
    out_path: Path,
    x_label: str = r"PRE cosine $D_{\mathrm{IQR}}$",
) -> pd.DataFrame:
    order_map = {target_id: i for i, target_id in enumerate(order)}
    df = df[df["target_id"].isin(order)].copy()
    df["plot_order"] = df["target_id"].map(order_map)
    df = df.sort_values("plot_order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.35, 3.9), constrained_layout=True)

    log_x = np.log10(df["iqr_normalized_dirichlet"].to_numpy(float))
    y = df["gnn_mae_over_iqr"].to_numpy(float)
    pearson_log = float(pd.Series(log_x).corr(pd.Series(y), method="pearson"))
    spearman = float(df["iqr_normalized_dirichlet"].corr(df["gnn_mae_over_iqr"], method="spearman"))

    for row in df.to_dict(orient="records"):
        target_id = row["target_id"]
        ax.scatter(
            row["iqr_normalized_dirichlet"],
            row["gnn_mae_over_iqr"],
            s=84,
            marker=MARKERS[target_id],
            color=TARGET_COLORS[target_id],
            edgecolor="#202020",
            linewidth=0.85,
            zorder=3,
        )
        dx, dy = OFFSETS[target_id]
        ax.annotate(
            row["target_label"],
            (row["iqr_normalized_dirichlet"], row["gnn_mae_over_iqr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=TARGET_COLORS[target_id],
            ha="right" if dx < 0 else "left",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=9.2)
    ax.set_ylabel(rf"{model_label} validation n-MAE, MAE/IQR$(y)$", fontsize=9.2)
    ax.grid(color="#E5E7EB", linewidth=0.75, which="both")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_min = df["iqr_normalized_dirichlet"].min()
    x_max = df["iqr_normalized_dirichlet"].max()
    y_pad = (df["gnn_mae_over_iqr"].max() - df["gnn_mae_over_iqr"].min()) * 0.13
    ax.set_xlim(x_min / 1.35, x_max * 1.35)
    ax.set_ylim(0.0, df["gnn_mae_over_iqr"].max() + y_pad)

    stats_handle = Line2D([], [], linestyle="none", marker=None)
    stats_legend = ax.legend(
        [stats_handle],
        [rf"$r_{{\log x}}={pearson_log:.2f}$, $\rho={spearman:.2f}$"],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=8.2,
        handlelength=0,
        handletextpad=0,
        borderpad=0.45,
    )
    ax.add_artist(stats_legend)

    target_ids = set(df["target_id"])
    shape_groups = [
        ({"dft_ref", "dft_ref_boe", "xtb_ref_boe"}, "full/baseline targets", "o"),
        ({"deltaxtb"}, r"$\Delta^{\mathrm{xTB}}$ residual", "D"),
        ({"deltabob", "deltapre", "deltapre_plus_a"}, "linear residuals", "s"),
        ({"delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"}, "MLP residuals", "^"),
    ]
    shape_handles = []
    shape_labels = []
    for ids, label, marker in shape_groups:
        if target_ids.intersection(ids):
            shape_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor="#F3F4F6",
                    markeredgecolor="#202020",
                    markeredgewidth=0.85,
                    markersize=6.2,
                )
            )
            shape_labels.append(label)
    ax.legend(
        shape_handles,
        shape_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=7.4,
        handlelength=1.0,
        handletextpad=0.45,
        borderpad=0.45,
        labelspacing=0.35,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    frame = load_target_panel()
    x, aligned = align_pre_features(frame)
    edges = build_cosine_edges(x, k=10)

    gemnet = smoothness_rows(aligned, edges, GEMNET_MAE)
    gemnet = plot_panel(gemnet, GEMNET_TARGET_ORDER, "GemNet", GEMNET_OUT_FIG)
    gemnet.drop(columns=["plot_order"]).to_csv(GEMNET_OUT_TABLE, index=False)

    schnet = smoothness_rows(aligned, edges, SCHNET_MAE)
    schnet = plot_panel(schnet, SCHNET_TARGET_ORDER, "SchNet", SCHNET_OUT_FIG)
    schnet.drop(columns=["plot_order"]).to_csv(SCHNET_OUT_TABLE, index=False)

    print("GemNet")
    print(gemnet.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {GEMNET_OUT_TABLE}")
    print(f"WROTE: {GEMNET_OUT_FIG}")
    print("\nSchNet")
    print(schnet.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {SCHNET_OUT_TABLE}")
    print(f"WROTE: {SCHNET_OUT_FIG}")


if __name__ == "__main__":
    main()
