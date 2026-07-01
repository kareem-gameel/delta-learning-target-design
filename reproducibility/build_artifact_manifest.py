#!/usr/bin/env python3
"""Build a checksum manifest for the curated paper release artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INCLUDE_GLOBS = [
    "README.md",
    "RUN_REPRODUCTION.md",
    "DATA_AND_CODE_AVAILABILITY_PLAN.md",
    "MODEL_METHODOLOGY_REPORT.md",
    "environment.yml",
    ".zenodo.json",
    "reproducibility/*.md",
    "reproducibility/*.py",
    "manuscript/**/*.tex",
    "manuscript/**/*.bib",
    "manuscript/figures/*.png",
    "manuscript/tables/*.csv",
    "manuscript/tables/*.tex",
    "manuscript/tables/*.png",
    "New_Analysis/README.md",
    "New_Analysis/*/README.md",
    "New_Analysis/datasets/*.csv",
    "New_Analysis/datasets/*.json",
    "New_Analysis/source_artifacts/**/*.csv",
    "New_Analysis/source_artifacts/**/*.md",
    "New_Analysis/expanded_target_panel/*.md",
    "New_Analysis/expanded_target_panel/target_registry.csv",
    "New_Analysis/expanded_target_panel/target_registry.json",
    "New_Analysis/expanded_target_panel/outputs/final_plots/*.csv",
    "New_Analysis/expanded_target_panel/outputs/final_plots/*.md",
    "New_Analysis/expanded_target_panel/outputs/gnn/*.csv",
    "New_Analysis/expanded_target_panel/outputs/ridge/*_target_metrics.csv",
    "New_Analysis/expanded_target_panel/outputs/ridge/*_alpha_grid_results.json",
    "New_Analysis/expanded_target_panel/outputs/ridge/*_input_validation.json",
    "New_Analysis/expanded_target_panel/outputs/smoothness/*.csv",
    "New_Analysis/expanded_target_panel/outputs/status/*.csv",
    "New_Analysis/expanded_target_panel/outputs/status/*.md",
    "New_Analysis/gemnet_learned_embedding_smoothness/*.md",
    "New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/**/*.csv",
    "New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/**/*.json",
    "New_Analysis/manuscript_support/*.md",
    "New_Analysis/manuscript_support/*.py",
    "New_Analysis/manuscript_support/figures/*.png",
    "New_Analysis/manuscript_support/tables/*.csv",
    "New_Analysis/manuscript_support/tables/*.md",
    "New_Analysis/manuscript_support/key_tables/*.csv",
    "New_Analysis/manuscript_support/key_tables/*.md",
    "New_Analysis/mlp_baselines/outputs/**/*.csv",
    "New_Analysis/mlp_baselines/outputs/**/*.json",
    "New_Analysis/mlp_residual_gnn_runs/outputs/**/*.csv",
    "New_Analysis/mlp_residual_gnn_runs/outputs/**/*.md",
    "New_Analysis/qm9_dft_ref_boe_transfer/model_manifest.csv",
    "New_Analysis/qm9_dft_ref_boe_transfer/outputs/**/*metrics*.csv",
    "New_Analysis/qm9_dft_ref_boe_transfer/outputs/**/*summary*.csv",
    "trillium_project/split_ids/*.csv",
]

EXCLUDED_SUFFIXES = {
    ".lmdb",
    ".parquet",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".sdf",
    ".extxyz",
    ".mol2",
    ".log",
    ".aux",
    ".out",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
}

EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "manuscript/build",
}

EXCLUDED_NAMES = {
    "artifact_manifest.csv",
    "artifact_manifest.json",
    "deltaml_paper_release_artifacts.zip",
}


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if path.name in EXCLUDED_NAMES:
        return True
    if any(part in rel for part in EXCLUDED_PARTS):
        return True
    return any(rel.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and not is_excluded(path):
                files.add(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    out_dir = ROOT / "reproducibility"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for path in collect_files():
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "sha256": sha256(path),
            }
        )

    csv_path = out_dir / "artifact_manifest.csv"
    json_path = out_dir / "artifact_manifest.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "mtime_utc", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "n_files": len(rows),
        "files": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"WROTE: {csv_path} files={len(rows)}")
    print(f"WROTE: {json_path}")


if __name__ == "__main__":
    main()
