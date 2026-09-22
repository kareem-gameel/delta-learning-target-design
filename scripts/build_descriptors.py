#!/usr/bin/env python3
"""Build paper PRE or PRE+A descriptors from an extended-XYZ file."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from scipy.sparse import hstack, save_npz

from target_design.descriptors.angular import build_angular_matrix
from target_design.descriptors.extxyz import iter_extxyz
from target_design.descriptors.pre import build_pre_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--mode", choices=["pre", "pre+a"], default="pre+a")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int)
    args = parser.parse_args()

    def records():
        stream = iter_extxyz(args.extxyz)
        return islice(stream, args.max_molecules) if args.max_molecules is not None else stream

    pre, molecule_ids = build_pre_matrix(records())
    matrix = pre
    metadata = {"descriptor": "PRE", "n_pre_features": int(pre.shape[1])}
    if args.mode == "pre+a":
        angular, angular_ids = build_angular_matrix(records())
        if molecule_ids != angular_ids:
            raise RuntimeError("PRE and angular descriptor rows are not aligned.")
        matrix = hstack((pre, angular), format="csr")
        metadata.update(
            descriptor="PRE+A",
            n_angular_features=int(angular.shape[1]),
            n_total_features=int(matrix.shape[1]),
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_dir / "features.npz", matrix)
    pd.DataFrame({"mol_id": molecule_ids}).to_csv(args.output_dir / "feature_ids.csv", index=False)
    metadata.update(
        n_molecules=len(molecule_ids),
        source_extxyz=str(args.extxyz),
        cutoff_angstrom=6.0,
        radial_shell_edges_angstrom=[0.0, 1.3, 1.8, 2.3, 3.0, 4.2, 6.0],
        angular_bins=12 if args.mode == "pre+a" else 0,
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
