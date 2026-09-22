"""Command-line interface for roughness calculation and baseline screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import align_features_and_targets, load_features
from .roughness import (
    build_weighted_knn_graph,
    compute_graph_roughness,
    scale_features_by_population_std,
)
from .screening import compare_to_parent


def _comma_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated column.")
    return values


def roughness_command(args: argparse.Namespace) -> None:
    features, feature_ids = load_features(args.features, args.id_column, args.feature_ids)
    features, targets = align_features_and_targets(
        features,
        feature_ids,
        args.targets,
        args.target_columns,
        args.id_column,
    )
    scaled, retained = scale_features_by_population_std(features, args.std_eps)
    edges, weights, graph_metadata = build_weighted_knn_graph(scaled, args.k, args.metric)
    rows = []
    for column in args.target_columns:
        result = compute_graph_roughness(targets[column].to_numpy(float), edges, weights)
        rows.append({"target": column, **result.to_dict()})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata = {
        **graph_metadata,
        "feature_scaling": "population_standard_deviation_without_centering",
        "n_features_total": int(features.shape[1]),
        "n_features_retained": int(retained.sum()),
        "n_features_dropped": int((~retained).sum()),
        "target_columns": args.target_columns,
        "features": str(args.features),
        "targets": str(args.targets),
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)
    print(metadata_path)


def screen_command(args: argparse.Namespace) -> None:
    table = pd.read_csv(args.input)
    result = compare_to_parent(
        table,
        args.parent,
        target_column=args.target_column,
        scale_column=args.scale_column,
        roughness_column=args.roughness_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="target-design")
    subparsers = parser.add_subparsers(dest="command", required=True)

    roughness = subparsers.add_parser("roughness", help="Compute D_IQR for one or more targets.")
    roughness.add_argument("--features", type=Path, required=True)
    roughness.add_argument("--feature-ids", type=Path)
    roughness.add_argument("--targets", type=Path, required=True)
    roughness.add_argument("--target-columns", type=_comma_list, required=True)
    roughness.add_argument("--id-column", default="mol_id")
    roughness.add_argument("--k", type=int, default=10)
    roughness.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    roughness.add_argument("--std-eps", type=float, default=1e-12)
    roughness.add_argument("--output", type=Path, required=True)
    roughness.add_argument("--metadata", type=Path)
    roughness.set_defaults(func=roughness_command)

    screen = subparsers.add_parser("screen", help="Compare targets in scale-roughness space.")
    screen.add_argument("--input", type=Path, required=True)
    screen.add_argument("--parent", required=True)
    screen.add_argument("--target-column", default="target")
    screen.add_argument("--scale-column", default="target_iqr")
    screen.add_argument("--roughness-column", default="d_iqr")
    screen.add_argument("--output", type=Path, required=True)
    screen.set_defaults(func=screen_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
