#!/usr/bin/env python3
"""Plot std-scaled PRE+A cosine-neighborhood roughness vs GemNet MAE/IQR."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from expanded_panel_common import load_feature_space  # noqa: E402
from run_expanded_smoothness import dirichlet_metrics, scale_features  # noqa: E402

from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet import (  # noqa: E402
    FIG_DIR,
    GEMNET_MAE,
    GEMNET_TARGET_ORDER,
    TABLE_DIR,
    TARGET_LABELS,
)
from plot_gnn_iqr_nmae_vs_pre_iqr_dirichlet_cosine import (  # noqa: E402
    build_cosine_edges,
    load_target_panel,
    plot_panel,
    smoothness_rows,
)


EDGE_CACHE = TABLE_DIR / "pre_plus_a_std_cosine_k10_val_edges.npz"
GEMNET_OUT_TABLE = TABLE_DIR / "gemnet_iqr_nmae_vs_pre_plus_a_std_iqr_norm_dirichlet_cosine_linear_mlp.csv"
GEMNET_OUT_FIG = FIG_DIR / "gemnet_iqr_nmae_vs_pre_plus_a_std_iqr_norm_dirichlet_cosine_linear_mlp.png"
GEMNET_TARGET_ORDER_WITH_EXTRA_TARGETS = ["dft_ref", "dft_ref_boe", "xtb_ref_boe", *GEMNET_TARGET_ORDER[1:]]


def align_pre_plus_a_features(frame: pd.DataFrame):
    x_all, feature_ids, feature_meta = load_feature_space("pre_plus_a")
    rows = pd.DataFrame({"mol_id": feature_ids, "_row": np.arange(len(feature_ids), dtype=int)})
    aligned = frame.merge(rows, on="mol_id", how="inner", validate="one_to_one").reset_index(drop=True)
    x = x_all[aligned["_row"].to_numpy(int)].tocsr()
    return x, aligned.drop(columns=["_row"]), feature_meta


def add_extra_targets(frame: pd.DataFrame) -> pd.DataFrame:
    dft_ref = pd.read_csv(ROOT / "New_Analysis/datasets/tmQM__DFT_ref.csv", usecols=["mol_id", "DFT_ref"])
    dft_ref["mol_id"] = dft_ref["mol_id"].astype(str)
    dft_ref = dft_ref.dropna().drop_duplicates("mol_id", keep="first").rename(columns={"DFT_ref": "dft_ref"})

    xtb_ref_boe = pd.read_csv(
        ROOT / "New_Analysis/datasets/tmQM__xTB_ref-BoE.csv",
        usecols=["mol_id", "xTB_ref-BoE"],
    )
    xtb_ref_boe["mol_id"] = xtb_ref_boe["mol_id"].astype(str)
    xtb_ref_boe = (
        xtb_ref_boe.dropna()
        .drop_duplicates("mol_id", keep="first")
        .rename(columns={"xTB_ref-BoE": "xtb_ref_boe"})
    )

    frame = frame.merge(dft_ref, on="mol_id", how="inner", validate="one_to_one")
    frame = frame.merge(xtb_ref_boe, on="mol_id", how="inner", validate="one_to_one")
    return frame.reset_index(drop=True)


def extra_smoothness_row(aligned: pd.DataFrame, edges, target_id: str) -> dict[str, object]:
    y = aligned[target_id].to_numpy(float)
    metrics = dirichlet_metrics(y, edges)
    target_iqr = float(metrics["target_iqr"])
    return {
        "target_id": target_id,
        "target_label": TARGET_LABELS[target_id],
        "target_variance": float(metrics["target_std"]) ** 2,
        "target_iqr": target_iqr,
        "gnn_val_mae": float(GEMNET_MAE[target_id]),
        "iqr_normalized_dirichlet": float(metrics["edge_rmse_over_iqr"]) ** 2,
        "gnn_mae_over_iqr": float(GEMNET_MAE[target_id]) / target_iqr,
    }


def main() -> None:
    frame = add_extra_targets(load_target_panel())
    x_raw, aligned, feature_meta = align_pre_plus_a_features(frame)
    x_std, n_retained, n_dropped = scale_features(x_raw, "std", 1e-12)
    edges = build_cosine_edges(x_std, k=10, edge_cache=EDGE_CACHE)

    gemnet = smoothness_rows(aligned, edges, GEMNET_MAE)
    extra_rows = pd.DataFrame(
        [
            extra_smoothness_row(aligned, edges, "dft_ref"),
            extra_smoothness_row(aligned, edges, "xtb_ref_boe"),
        ]
    )
    gemnet = pd.concat([extra_rows, gemnet], ignore_index=True)
    gemnet = plot_panel(
        gemnet,
        GEMNET_TARGET_ORDER_WITH_EXTRA_TARGETS,
        "GemNet",
        GEMNET_OUT_FIG,
        x_label=r"std-scaled PRE+A cosine $D_{\mathrm{IQR}}$",
    )
    gemnet.drop(columns=["plot_order"]).to_csv(GEMNET_OUT_TABLE, index=False)

    print(
        "std-scaled PRE+A features "
        f"total={feature_meta['n_features']} retained={n_retained} dropped={n_dropped}"
    )
    print("GemNet")
    print(gemnet.drop(columns=["plot_order"]).to_string(index=False))
    print(f"WROTE: {GEMNET_OUT_TABLE}")
    print(f"WROTE: {GEMNET_OUT_FIG}")


if __name__ == "__main__":
    main()
