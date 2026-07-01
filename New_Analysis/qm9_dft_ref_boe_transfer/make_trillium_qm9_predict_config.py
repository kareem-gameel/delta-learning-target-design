#!/usr/bin/env python3
"""Create a FairChem QM9 prediction config for a Trillium GemNet run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--qm9-lmdb",
        default="/scratch/gameelka/deltaML_project/lmdb_files/qm9_psi4_totE.lmdb",
    )
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-config", type=Path)
    parser.add_argument(
        "--prediction-split-index",
        type=int,
        default=-1,
        help="Dataset list index FairChem predict should read from. Default: last split.",
    )
    return parser.parse_args()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    source_config = args.source_config or args.run_root / "configs" / f"{args.run_name}.yml"
    checkpoint = args.checkpoint or first_existing(
        [
            args.run_root / "runs" / "seed_101" / "checkpoints" / args.run_name / "best_checkpoint.pt",
            args.run_root / "runs" / args.run_name / "checkpoints" / args.run_name / "best_checkpoint.pt",
        ]
    )
    out_config = args.out_config or args.run_root / "eval" / "configs" / f"{args.run_name}_qm9_predict.yml"

    if not source_config.exists():
        raise FileNotFoundError(f"Missing source config: {source_config}")
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("Missing checkpoint. Pass --checkpoint explicitly.")
    if not Path(args.qm9_lmdb).exists():
        raise FileNotFoundError(f"Missing QM9 LMDB: {args.qm9_lmdb}")

    cfg = yaml.safe_load(source_config.read_text())
    datasets = cfg.get("dataset")
    if not isinstance(datasets, list) or not datasets:
        raise TypeError(f"Expected non-empty dataset list, got {type(datasets)}")

    split_index = args.prediction_split_index
    datasets[split_index]["src"] = args.qm9_lmdb
    cfg["checkpoint"] = str(checkpoint)

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("source_config:", source_config)
    print("out_config:", out_config)
    print("checkpoint:", checkpoint)
    print("qm9_lmdb:", args.qm9_lmdb)
    print("prediction_split_index:", split_index)
    print("prediction_split:", datasets[split_index])


if __name__ == "__main__":
    main()
