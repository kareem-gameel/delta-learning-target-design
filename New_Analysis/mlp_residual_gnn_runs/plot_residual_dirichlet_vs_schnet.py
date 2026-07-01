#!/usr/bin/env python3
"""Plot normalized Dirichlet smoothness of residual targets vs SchNet MAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
EXPANDED_DIR = ROOT / "New_Analysis/expanded_target_panel"
sys.path.insert(0, str(EXPANDED_DIR))

from expanded_panel_common import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    TargetSpec,
    load_feature_space,
    load_split_ids,
    load_target_frame,
)
from run_expanded_smoothness import align_features, dirichlet_metrics, unique_undirected_edges  # noqa: E402


TARGETS = [
    {
        "target_id": "deltaxtb",
        "target_label": "Delta^xTB_ref,BoE",
        "plot_label": "Delta xTB",
        "feature_space": "pre",
        "family": "xTB residual",
        "spec": TargetSpec(
            "deltaxtb",
            "Delta^xTB_ref,BoE",
            "Delta^xTB",
            ROOT / "New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv",
            "Delta_xTB-ref-BoE",
            "residual_target",
        ),
    },
    {
        "target_id": "deltabob",
        "target_label": "Delta^BoB_ref,BoE",
        "plot_label": "Delta BoB",
        "feature_space": "bob",
        "family": "linear residual",
        "spec": TargetSpec(
            "deltabob",
            "Delta^BoB_ref,BoE",
            "Delta^BoB",
            ROOT / "New_Analysis/datasets/tmQM__Delta_BoB-ref-BoE.csv",
            "E_resid_after_BoE_BoB_tmQM",
            "residual_target",
        ),
    },
    {
        "target_id": "deltapre",
        "target_label": "Delta^PRE_ref,BoE",
        "plot_label": "Delta PRE",
        "feature_space": "pre",
        "family": "linear residual",
        "spec": TargetSpec(
            "deltapre",
            "Delta^PRE_ref,BoE",
            "Delta^PRE",
            ROOT / "New_Analysis/datasets/tmQM__Delta_PRE-ref-BoE.csv",
            "E_resid_after_BoE_PRE_tmQM",
            "residual_target",
        ),
    },
    {
        "target_id": "deltapre_plus_a",
        "target_label": "Delta^PRE+A_ref,BoE",
        "plot_label": "Delta PRE+A",
        "feature_space": "pre_plus_a",
        "family": "linear residual",
        "spec": TargetSpec(
            "deltapre_plus_a",
            "Delta^PRE+A_ref,BoE",
            "Delta^PRE+A",
            ROOT / "New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv",
            "Delta_PRE_plus_A-ref-BoE",
            "residual_target",
        ),
    },
    {
        "target_id": "delta_mlp_bob",
        "target_label": "Delta^MLP_BoB_ref,BoE",
        "plot_label": "Delta MLP-BoB",
        "feature_space": "bob",
        "family": "MLP residual",
        "spec": TargetSpec(
            "delta_mlp_bob",
            "Delta^MLP_BoB_ref,BoE",
            "Delta^MLP_BoB",
            ROOT / "New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
            "Delta_MLP_BoB-ref-BoE",
            "residual_target",
        ),
    },
    {
        "target_id": "delta_mlp_pre",
        "target_label": "Delta^MLP_PRE_ref,BoE",
        "plot_label": "Delta MLP-PRE",
        "feature_space": "pre",
        "family": "MLP residual",
        "spec": TargetSpec(
            "delta_mlp_pre",
            "Delta^MLP_PRE_ref,BoE",
            "Delta^MLP_PRE",
            ROOT / "New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
            "Delta_MLP_PRE-ref-BoE",
            "residual_target",
        ),
    },
    {
        "target_id": "delta_mlp_pre_plus_a",
        "target_label": "Delta^MLP_PRE+A_ref,BoE",
        "plot_label": "Delta MLP-PRE+A",
        "feature_space": "pre_plus_a",
        "family": "MLP residual",
        "spec": TargetSpec(
            "delta_mlp_pre_plus_a",
            "Delta^MLP_PRE+A_ref,BoE",
            "Delta^MLP_PRE+A",
            ROOT / "New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
            "Delta_MLP_PRE_plus_A-ref-BoE",
            "residual_target",
        ),
    },
]

SCHNET_MAE = {
    "deltaxtb": 0.0093,
    "deltabob": 0.0142,
    "deltapre": 0.0125,
    "deltapre_plus_a": 0.0175,
}

FEATURE_LABEL = {
    "bob": "BoB",
    "pre": "PRE",
    "pre_plus_a": "PRE+A",
}

FAMILY_COLOR = {
    "xTB residual": "#10b981",
    "linear residual": "#2563eb",
    "MLP residual": "#ef4444",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--force-feature-space",
        choices=["bob", "pre", "pre_plus_a"],
        default=None,
        help="If set, compute all targets in this single feature space.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/smoothness",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def build_edges_cached(x, feature_space: str, k: int, n_jobs: int, out_dir: Path) -> np.ndarray:
    cache_dir = out_dir / "edge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{feature_space}_raw_k{k}_n{x.shape[0]}_d{x.shape[1]}_edges.npz"
    if cache_path.exists():
        log(f"Loading cached edges: {cache_path}")
        return np.load(cache_path)["edges"]

    log(f"Building exact kNN edges: feature_space={feature_space} rows={x.shape[0]} dims={x.shape[1]} k={k}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=n_jobs)
    nn.fit(x)
    _dist, ind = nn.kneighbors(x)
    edges = unique_undirected_edges(ind[:, 1:])
    np.savez_compressed(cache_path, edges=edges)
    log(f"WROTE: {cache_path} edges={edges.shape[0]}")
    return edges


def load_mlp_schnet_maes(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[(df["model_id"].eq("SchNet")) & (df["status"].eq("ok"))]
    return dict(zip(df["target_id"].astype(str), df["best_val_mae"].astype(float)))


def load_target_for_val(spec: TargetSpec, split_df: pd.DataFrame) -> pd.DataFrame:
    val_split = split_df[split_df["split"].eq("val")].copy()
    return load_target_frame(spec, val_split)[["mol_id", "target"]].merge(
        val_split[["mol_id", "split"]], on="mol_id", how="inner", validate="one_to_one"
    )


def compute_feature_space_rows(
    feature_space: str,
    targets: list[dict[str, object]],
    split_df: pd.DataFrame,
    k: int,
    n_jobs: int,
    out_dir: Path,
):
    x_all, feature_ids, feature_meta = load_feature_space(feature_space)
    frames = []
    for target in targets:
        frame = load_target_for_val(target["spec"], split_df).rename(columns={"target": target["target_id"]})
        frames.append((target, frame))

    base = frames[0][1][["mol_id", "split"]].copy()
    for target, frame in frames:
        base = base.merge(frame[["mol_id", target["target_id"]]], on="mol_id", how="inner", validate="one_to_one")

    x, aligned = align_features(x_all, feature_ids, base)
    edges = build_edges_cached(x, feature_space, k, n_jobs, out_dir)

    rows = []
    for target, _frame in frames:
        y = aligned[target["target_id"]].to_numpy(float)
        rows.append(
            {
                "target_id": target["target_id"],
                "target_label": target["target_label"],
                "plot_label": target["plot_label"],
                "family": target["family"],
                "feature_space": feature_meta["feature_space"],
                "feature_scaling": "raw",
                "k": int(k),
                "n_molecules": int(len(y)),
                "n_features": int(x.shape[1]),
                **dirichlet_metrics(y, edges),
            }
        )
    return rows


def add_schnet_metrics(df: pd.DataFrame) -> pd.DataFrame:
    schnet = dict(SCHNET_MAE)
    schnet.update(load_mlp_schnet_maes(ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary.csv"))
    df["schnet_mae"] = df["target_id"].map(schnet)
    df["schnet_mae_over_iqr"] = df["schnet_mae"] / df["target_iqr"]
    return df


def corr_text(df: pd.DataFrame, y_col: str) -> str:
    keep = df[["edge_avg_norm_by_population_var", y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(keep) < 3:
        return "n<3"
    r = pearsonr(keep["edge_avg_norm_by_population_var"], keep[y_col]).statistic
    rho = spearmanr(keep["edge_avg_norm_by_population_var"], keep[y_col]).statistic
    return f"r={r:.2f}, rho={rho:.2f}"


def plot_scatter(df: pd.DataFrame, y_col: str, y_label: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    for family, group in df.groupby("family", sort=False):
        ax.scatter(
            group["edge_avg_norm_by_population_var"],
            group[y_col],
            s=78,
            color=FAMILY_COLOR.get(family, "#6b7280"),
            edgecolor="#111827",
            linewidth=0.8,
            label=family,
            zorder=3,
        )

    for row in df.itertuples(index=False):
        label = f"{row.plot_label}\n({row.feature_space})"
        ax.annotate(
            label,
            (row.edge_avg_norm_by_population_var, getattr(row, y_col)),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlabel("Normalized Dirichlet smoothness\nedge mean squared jump / target variance")
    ax.set_ylabel(y_label)
    ax.set_title(f"Residual smoothness vs SchNet performance ({corr_text(df, y_col)})", loc="left", fontsize=12.5)
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_df = load_split_ids(args.split_dir)
    rows = []
    feature_spaces = [args.force_feature_space] if args.force_feature_space else ["bob", "pre", "pre_plus_a"]
    for feature_space in feature_spaces:
        if args.force_feature_space:
            targets = TARGETS
        else:
            targets = [target for target in TARGETS if target["feature_space"] == feature_space]
        rows.extend(compute_feature_space_rows(feature_space, targets, split_df, args.k, args.n_jobs, args.out_dir))

    df = add_schnet_metrics(pd.DataFrame(rows))
    df = df.sort_values(["feature_space", "family", "target_id"]).reset_index(drop=True)

    out_csv = args.out_dir / "residual_normalized_dirichlet_vs_schnet_points.csv"
    mae_png = args.out_dir / "residual_normalized_dirichlet_vs_schnet_mae.png"
    nmae_png = args.out_dir / "residual_normalized_dirichlet_vs_schnet_mae_over_iqr.png"
    df.to_csv(out_csv, index=False)
    plot_scatter(df, "schnet_mae", "SchNet validation MAE (Ha)", mae_png)
    plot_scatter(df, "schnet_mae_over_iqr", "SchNet validation MAE / target IQR", nmae_png)

    print(
        df[
            [
                "target_id",
                "feature_space",
                "edge_avg_norm_by_population_var",
                "target_iqr",
                "schnet_mae",
                "schnet_mae_over_iqr",
            ]
        ].to_string(index=False)
    )
    print(f"WROTE: {out_csv}")
    print(f"WROTE: {mae_png}")
    print(f"WROTE: {nmae_png}")


if __name__ == "__main__":
    main()
