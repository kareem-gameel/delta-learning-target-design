#!/usr/bin/env python3
"""Join learned-embedding roughness rows to held-out GemNet test metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roughness-csv", type=Path, required=True)
    parser.add_argument(
        "--model-metrics-csv",
        type=Path,
        required=True,
        help=(
            "CSV with columns embedding_model_id, trained_target_id, test_mae. "
            "Optional columns such as test_mae_over_iqr/test_rmse/test_r2 are preserved."
        ),
    )
    parser.add_argument("--self-target-only", action="store_true")
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roughness = pd.read_csv(args.roughness_csv)
    metrics = pd.read_csv(args.model_metrics_csv)
    required_roughness = {"embedding_model_id", "target_id"}
    required_metrics = {"embedding_model_id", "trained_target_id", "test_mae"}
    missing_roughness = sorted(required_roughness - set(roughness.columns))
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_roughness:
        raise ValueError(f"{args.roughness_csv} missing columns: {missing_roughness}")
    if missing_metrics:
        raise ValueError(f"{args.model_metrics_csv} missing columns: {missing_metrics}")

    metrics = metrics.drop_duplicates("embedding_model_id", keep="first").copy()
    out = roughness.merge(metrics, on="embedding_model_id", how="left", validate="many_to_one")
    out["is_trained_target"] = out["target_id"].eq(out["trained_target_id"])
    if args.self_target_only:
        out = out[out["is_trained_target"]].copy()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"WROTE: {args.out_csv}")


if __name__ == "__main__":
    main()

