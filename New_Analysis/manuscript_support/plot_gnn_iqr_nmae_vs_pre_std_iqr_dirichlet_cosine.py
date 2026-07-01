#!/usr/bin/env python3
"""Plot std-scaled PRE cosine-neighborhood roughness vs GNN MAE/IQR."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from run_expanded_smoothness import scale_features  # noqa: E402

from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet import (  # noqa: E402
    GEMNET_MAE,
    GEMNET_TARGET_ORDER,
    SCHNET_MAE,
    SCHNET_TARGET_ORDER,
    TABLE_DIR,
    FIG_DIR,
)
from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet_cosine import (  # noqa: E402
    align_pre_features,
    build_cosine_edges,
    load_target_panel,
    plot_panel,
    smoothness_rows,
)


EDGE_CACHE = TABLE_DIR / "pre_std_cosine_k10_val_edges.npz"
GEMNET_OUT_TABLE = TABLE_DIR / "gemnet_iqr_nmae_vs_pre_std_iqr_norm_dirichlet_cosine_linear_mlp.csv"
GEMNET_OUT_FIG = FIG_DIR / "gemnet_iqr_nmae_vs_pre_std_iqr_norm_dirichlet_cosine_linear_mlp.png"
SCHNET_OUT_TABLE = TABLE_DIR / "schnet_iqr_nmae_vs_pre_std_iqr_norm_dirichlet_cosine_linear_mlp.csv"
SCHNET_OUT_FIG = FIG_DIR / "schnet_iqr_nmae_vs_pre_std_iqr_norm_dirichlet_cosine_linear_mlp.png"


def main() -> None:
    frame = load_target_panel()
    x_raw, aligned = align_pre_features(frame)
    x_std, n_retained, n_dropped = scale_features(x_raw, "std", 1e-12)
    edges = build_cosine_edges(x_std, k=10, edge_cache=EDGE_CACHE)
    x_label = r"std-scaled PRE cosine $D_{\mathrm{IQR}}$"

    gemnet = smoothness_rows(aligned, edges, GEMNET_MAE)
    gemnet = plot_panel(gemnet, GEMNET_TARGET_ORDER, "GemNet", GEMNET_OUT_FIG, x_label=x_label)
    gemnet.drop(columns=["plot_order"]).to_csv(GEMNET_OUT_TABLE, index=False)

    schnet = smoothness_rows(aligned, edges, SCHNET_MAE)
    schnet = plot_panel(schnet, SCHNET_TARGET_ORDER, "SchNet", SCHNET_OUT_FIG, x_label=x_label)
    schnet.drop(columns=["plot_order"]).to_csv(SCHNET_OUT_TABLE, index=False)

    print(f"std-scaled PRE features retained={n_retained} dropped={n_dropped}")
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
