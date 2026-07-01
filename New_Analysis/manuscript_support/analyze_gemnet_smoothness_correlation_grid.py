#!/usr/bin/env python3
"""Grid-search smoothness/MAE normalizations for GemNet target learnability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SMOOTHNESS_DIR = ROOT / "New_Analysis/expanded_target_panel/outputs/smoothness"
STATUS_METRICS = ROOT / "New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv"
MLP_GNN_METRICS = ROOT / "New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary_all.csv"
OUT_DIR = ROOT / "New_Analysis/manuscript_support/tables"
CORR_OUT = OUT_DIR / "gemnet_smoothness_correlation_grid.csv"
POINTS_OUT = OUT_DIR / "gemnet_smoothness_correlation_grid_points.csv"
TOP_OUT = OUT_DIR / "gemnet_smoothness_correlation_grid_top.csv"
REPORT_OUT = OUT_DIR / "gemnet_smoothness_correlation_grid_summary.md"

SMOOTHNESS_FILES = {
    "BoB": [
        SMOOTHNESS_DIR / "bob_space_dirichlet_val.csv",
    ],
    "PRE": [
        SMOOTHNESS_DIR / "pre_space_dirichlet_val.csv",
        SMOOTHNESS_DIR / "pre_space_extra_dirichlet_val.csv",
    ],
    "PRE+A": [
        SMOOTHNESS_DIR / "pre_plus_a_space_dirichlet_val.csv",
        SMOOTHNESS_DIR / "pre_plus_a_space_extra_dirichlet_val.csv",
    ],
}

TARGET_SETS = {
    "manuscript_mlp5": ["dft", "deltaxtb", "delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"],
    "expanded_linear9": ["dft", "xtb", "deltaxtb", "deltabob", "deltapre", "bob", "pre", "pre_plus_a", "deltapre_plus_a"],
    "core_linear5": ["dft", "deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"],
    "delta_mlp4": ["deltaxtb", "delta_mlp_bob", "delta_mlp_pre", "delta_mlp_pre_plus_a"],
    "delta_linear4": ["deltaxtb", "deltabob", "deltapre", "deltapre_plus_a"],
}

X_METRICS = {
    "absolute_dirichlet": "edge_mean_squared_diff",
    "edge_rmse": "edge_rmse",
    "variance_normalized_dirichlet": "edge_avg_norm_by_population_var",
    "random_pair_normalized_dirichlet": "edge_avg_norm_by_random_pair",
    "iqr_normalized_dirichlet": "edge_rmse_over_iqr_sq",
    "edge_rmse_over_iqr": "edge_rmse_over_iqr",
    "sum_normalized_dirichlet": "E_norm_sum",
}

Y_METRICS = {
    "mae": "gemnet_val_mae",
    "mae_over_iqr": "gemnet_val_mae_over_iqr",
    "mae_over_std": "gemnet_val_mae_over_std",
    "mae_over_variance": "gemnet_val_mae_over_variance",
}


def load_smoothness() -> pd.DataFrame:
    pieces = []
    for feature_space, paths in SMOOTHNESS_FILES.items():
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["feature_space_group"] = feature_space
            pieces.append(df)
    if not pieces:
        raise FileNotFoundError("No smoothness CSV files found")
    out = pd.concat(pieces, ignore_index=True)
    out["target_id_original"] = out["target_id"]
    out["edge_rmse_over_iqr_sq"] = out["edge_rmse_over_iqr"].astype(float) ** 2
    out["target_variance"] = out["target_std"].astype(float) ** 2
    return out


def load_gemnet_metrics() -> pd.DataFrame:
    status = pd.read_csv(STATUS_METRICS)
    status = status[(status["model_id"].eq("GemNet")) & (status["split"].eq("val"))].copy()
    status_rows = status[["target_id", "target_label", "mae"]].rename(columns={"mae": "gemnet_val_mae"})

    mlp = pd.read_csv(MLP_GNN_METRICS)
    mlp = mlp[(mlp["model_id"].eq("GemNet")) & (mlp["status"].eq("ok"))].copy()
    mlp_rows = mlp[["target_id", "target_label", "best_val_mae"]].rename(
        columns={"best_val_mae": "gemnet_val_mae"}
    )

    metrics = pd.concat([status_rows, mlp_rows], ignore_index=True)
    metrics = metrics.drop_duplicates("target_id", keep="last")
    return metrics


def build_points() -> pd.DataFrame:
    smooth = load_smoothness()
    metrics = load_gemnet_metrics()
    points = smooth.merge(metrics, on="target_id", how="inner", suffixes=("", "_metric"))
    points["gemnet_val_mae_over_iqr"] = points["gemnet_val_mae"] / points["target_iqr"]
    points["gemnet_val_mae_over_std"] = points["gemnet_val_mae"] / points["target_std"]
    points["gemnet_val_mae_over_variance"] = points["gemnet_val_mae"] / points["target_variance"]
    return points


def corr_pair(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float]:
    keep = df[[x_col, y_col]].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(keep) < 3:
        return float("nan"), float("nan")
    return (
        float(keep[x_col].corr(keep[y_col], method="pearson")),
        float(keep[x_col].corr(keep[y_col], method="spearman")),
    )


def analyze(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_set_name, target_ids in TARGET_SETS.items():
        for feature_space in sorted(points["feature_space_group"].dropna().unique()):
            for feature_scaling in sorted(points["feature_scaling"].dropna().unique()):
                sub = points[
                    points["feature_space_group"].eq(feature_space)
                    & points["feature_scaling"].eq(feature_scaling)
                    & points["target_id"].isin(target_ids)
                    & points["k"].eq(10)
                ].copy()
                present = set(sub["target_id"])
                if not set(target_ids).issubset(present):
                    continue
                sub = sub.set_index("target_id").loc[target_ids].reset_index()
                for x_name, x_col in X_METRICS.items():
                    for y_name, y_col in Y_METRICS.items():
                        r, rho = corr_pair(sub, x_col, y_col)
                        rows.append(
                            {
                                "target_set": target_set_name,
                                "n_targets": len(sub),
                                "feature_space": feature_space,
                                "feature_scaling": feature_scaling,
                                "k": 10,
                                "knn_metric": "euclidean",
                                "x_metric": x_name,
                                "x_column": x_col,
                                "y_metric": y_name,
                                "y_column": y_col,
                                "pearson_r": r,
                                "spearman_rho": rho,
                                "positive_min_r_rho": min(r, rho),
                                "positive_mean_r_rho": (r + rho) / 2.0,
                                "abs_min_r_rho": min(abs(r), abs(rho)),
                                "abs_mean_r_rho": (abs(r) + abs(rho)) / 2.0,
                            }
                        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    show = df.head(n).copy()
    for col in ["pearson_r", "spearman_rho", "positive_min_r_rho", "positive_mean_r_rho", "abs_min_r_rho"]:
        if col in show:
            show[col] = show[col].map(lambda x: f"{x:.3f}")
    return show[columns].to_markdown(index=False)


def write_report(corr: pd.DataFrame) -> None:
    positive = corr.sort_values(
        ["positive_min_r_rho", "positive_mean_r_rho", "pearson_r", "spearman_rho"],
        ascending=False,
    ).reset_index(drop=True)
    absolute = corr.sort_values(
        ["abs_min_r_rho", "abs_mean_r_rho"],
        ascending=False,
    ).reset_index(drop=True)
    mlp5 = positive[positive["target_set"].eq("manuscript_mlp5")]

    columns = [
        "target_set",
        "feature_space",
        "feature_scaling",
        "x_metric",
        "y_metric",
        "pearson_r",
        "spearman_rho",
        "positive_min_r_rho",
    ]
    lines = [
        "# GemNet Smoothness Correlation Grid",
        "",
        "This is an exploratory sensitivity analysis over available local smoothness outputs.",
        "",
        "Fixed choices in this scan:",
        "- split: tmQM validation",
        "- kNN graph: k=10",
        "- kNN distance: Euclidean",
        "- feature scaling: raw or standardized feature columns, using the precomputed smoothness files",
        "- target sets: manuscript MLP-residual five-target set, expanded linear nine-target set, and delta-only subsets",
        "",
        "The main ranking uses `positive_min_r_rho = min(Pearson r, Spearman rho)`, because the hypothesis is that rougher targets should have larger normalized GemNet error.",
        "",
        "## Best Positive Agreements",
        "",
        markdown_table(positive, columns, 15),
        "",
        "## Best Positive Agreements for Manuscript MLP Five-Target Set",
        "",
        markdown_table(mlp5, columns, 15),
        "",
        "## Best Absolute Agreements",
        "",
        markdown_table(
            absolute,
            [
                "target_set",
                "feature_space",
                "feature_scaling",
                "x_metric",
                "y_metric",
                "pearson_r",
                "spearman_rho",
                "abs_min_r_rho",
            ],
            15,
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "A perfect or near-perfect score in this grid should be treated as descriptive, not confirmatory. "
        "With 4-5 targets, many normalization choices can produce high rank agreement by construction. "
        "The defensible result is the stability of a relationship across related choices, not a single best row.",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = build_points()
    corr = analyze(points)
    top = corr.sort_values(
        ["positive_min_r_rho", "positive_mean_r_rho", "pearson_r", "spearman_rho"],
        ascending=False,
    ).reset_index(drop=True)

    points.to_csv(POINTS_OUT, index=False)
    corr.to_csv(CORR_OUT, index=False)
    top.head(100).to_csv(TOP_OUT, index=False)
    write_report(corr)

    print(f"WROTE: {POINTS_OUT}")
    print(f"WROTE: {CORR_OUT}")
    print(f"WROTE: {TOP_OUT}")
    print(f"WROTE: {REPORT_OUT}")
    print("\nTop positive rows:")
    cols = [
        "target_set",
        "feature_space",
        "feature_scaling",
        "x_metric",
        "y_metric",
        "pearson_r",
        "spearman_rho",
        "positive_min_r_rho",
    ]
    print(top[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
