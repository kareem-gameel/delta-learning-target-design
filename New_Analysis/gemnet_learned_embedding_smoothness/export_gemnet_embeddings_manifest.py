#!/usr/bin/env python3
"""Export GemNet atom embeddings from a manifest of configs/checkpoints.

This script is intended to run on Trillium next to the existing clean export
wrapper:

    latent_embeddings/CODEX_embeddings_analysis/trillium_clean_export/main_export_clean.py

The manifest must contain:

    tag,config_path,checkpoint_path

Optional columns:

    embedding_model_id,embedding_model_label,output_dir,dataset_role,mol_id_filter_file
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


ROLE_TO_INDEX = {"train": 0, "val": 1, "test": 2}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def get_role_src(cfg: dict, role: str) -> str:
    idx = ROLE_TO_INDEX[role]
    datasets = cfg.get("dataset")
    if not isinstance(datasets, list) or len(datasets) <= idx:
        raise ValueError(f"Config dataset field does not contain split index {idx} for role={role}")
    src = datasets[idx].get("src")
    if not src:
        raise ValueError(f"Config dataset[{idx}] is missing src")
    return str(src)


def prepare_export_config(
    *,
    config_path: Path,
    output_config_path: Path,
    output_dir: Path,
    tag: str,
    role: str,
    flush_every_batches: int,
    mol_id_filter_file: str | None,
) -> str:
    cfg = yaml.safe_load(config_path.read_text())
    src = get_role_src(cfg, role)

    cfg["submit"] = False
    cfg["summit"] = False
    cfg["identifier"] = f"predict_export_clean_{tag}_{role}"
    cfg["dataset"] = [{"src": src}, {"src": src}, {"src": src}]
    cfg["val_dataset"] = [{"src": src}]
    cfg["test_dataset"] = [{"src": src}]

    cfg.setdefault("model", {})
    cfg["model"]["name"] = "gemnet_t_clean_export"
    cfg["model"]["embedding_export"] = {
        "enabled": True,
        "output_dir": str(output_dir),
        "split_name": role,
        "model_tag": tag,
        "flush_every_batches": int(flush_every_batches),
        "mol_id_filter_file": mol_id_filter_file or None,
    }

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return src


def clean_output_dir(output_dir: Path) -> None:
    for path in output_dir.glob("atomic_embeddings_*.parquet"):
        path.unlink()
    meta = output_dir / "export_meta.json"
    if meta.exists():
        meta.unlink()


def run_export(
    *,
    python_bin: str,
    export_root: Path,
    export_config: Path,
    checkpoint: Path,
    timestamp_id: str,
) -> None:
    cmd = [
        python_bin,
        "-u",
        str(export_root / "main_export_clean.py"),
        "--mode",
        "predict",
        "--config-yml",
        str(export_config),
        "--checkpoint",
        str(checkpoint),
        "--timestamp-id",
        timestamp_id,
    ]
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize_output(output_dir: Path) -> dict:
    files = sorted(output_dir.glob("atomic_embeddings_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No atomic embedding parquet shards found in {output_dir}")
    n_rows = 0
    first_columns = None
    first_embedding_dim = None
    for path in files:
        df = pd.read_parquet(path)
        n_rows += len(df)
        if first_columns is None:
            first_columns = list(df.columns)
            first_embedding_dim = len(df["atom_embedding"].iloc[0]) if len(df) else None
    return {
        "output_dir": str(output_dir),
        "n_files": len(files),
        "n_atomic_rows": int(n_rows),
        "first_columns": first_columns,
        "first_embedding_dim": first_embedding_dim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=sorted(ROLE_TO_INDEX), default="test")
    parser.add_argument("--flush-every-batches", type=int, default=100)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config-out-dir", type=Path, default=None)
    parser.add_argument("--model-filter", default=None, help="Comma-separated tags to run")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_file(args.export_root / "main_export_clean.py", "main_export_clean.py")
    require_file(args.manifest, "export manifest")
    config_out_dir = args.config_out_dir or (args.out_root / "configs_export_predict_clean")
    model_filter = None
    if args.model_filter:
        model_filter = {part.strip() for part in args.model_filter.split(",") if part.strip()}

    manifest = pd.read_csv(args.manifest)
    required = {"tag", "config_path", "checkpoint_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{args.manifest} missing required columns: {missing}")

    rows = []
    for item in manifest.to_dict(orient="records"):
        tag = str(item["tag"])
        if model_filter is not None and tag not in model_filter:
            log(f"SKIP: {tag}; not in --model-filter")
            continue

        role = str(item.get("dataset_role") or args.dataset_role)
        if role not in ROLE_TO_INDEX:
            raise ValueError(f"{tag}: unsupported dataset_role={role!r}")

        config_path = Path(str(item["config_path"]))
        checkpoint_path = Path(str(item["checkpoint_path"]))
        require_file(config_path, f"{tag} config")
        require_file(checkpoint_path, f"{tag} checkpoint")

        output_dir_value = item.get("output_dir")
        if pd.notna(output_dir_value) and str(output_dir_value).strip():
            output_dir = Path(str(output_dir_value))
        else:
            output_dir = args.out_root / role / tag

        mol_id_filter = item.get("mol_id_filter_file")
        mol_id_filter_str = None
        if pd.notna(mol_id_filter) and str(mol_id_filter).strip():
            mol_id_filter_path = Path(str(mol_id_filter))
            require_file(mol_id_filter_path, f"{tag} mol_id_filter_file")
            mol_id_filter_str = str(mol_id_filter_path)

        export_config = config_out_dir / f"predict_export_clean_{tag}_{role}.yml"
        src = prepare_export_config(
            config_path=config_path,
            output_config_path=export_config,
            output_dir=output_dir,
            tag=tag,
            role=role,
            flush_every_batches=args.flush_every_batches,
            mol_id_filter_file=mol_id_filter_str,
        )
        if args.clean:
            clean_output_dir(output_dir)

        row = {
            **item,
            "tag": tag,
            "dataset_role": role,
            "prepared_config_path": str(export_config),
            "resolved_output_dir": str(output_dir),
            "resolved_dataset_src": src,
        }
        log(f"PREPARED: {tag} role={role} src={src} out={output_dir}")

        if not args.dry_run:
            run_export(
                python_bin=args.python_bin,
                export_root=args.export_root,
                export_config=export_config,
                checkpoint=checkpoint_path,
                timestamp_id=f"clean_export_{tag}_{role}",
            )
            row.update(summarize_output(output_dir))
        rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "embedding_export_manifest_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    log(f"WROTE: {summary_path}")
    meta_path = args.out_root / "embedding_export_manifest_summary.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "export_root": str(args.export_root),
                "out_root": str(args.out_root),
                "dataset_role": args.dataset_role,
                "flush_every_batches": args.flush_every_batches,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"WROTE: {meta_path}")


if __name__ == "__main__":
    main()

