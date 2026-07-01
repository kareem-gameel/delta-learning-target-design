#!/usr/bin/env python3
"""Prepare GemNet/SchNet runs for MLP-baseline residual targets on Trillium."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
import yaml


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    prose_label: str
    csv_path: Path
    target_column: str
    lmdb_prefix: str
    description: str


PROJECT_ROOT = Path(os.environ.get("DELTAML_PROJECT", "/scratch/gameelka/deltaML_project"))
TRAIN_ROOT = Path(os.environ.get("FAIRCHEM_TRAIN_ROOT", str(PROJECT_ROOT / "train")))
SOURCE_LMDB_ROOT = Path(os.environ.get("SOURCE_LMDB_ROOT", str(PROJECT_ROOT / "lmdb_files/tmqm_totE")))
OUTPUT_LMDB_ROOT = Path(
    os.environ.get("OUTPUT_LMDB_ROOT", str(PROJECT_ROOT / "lmdb_files/tmqm_mlp_residual_targets"))
)

SOURCE_LMDBS = {
    "train": SOURCE_LMDB_ROOT / "tmqm_totE_matched_train.lmdb",
    "val": SOURCE_LMDB_ROOT / "tmqm_totE_matched_val.lmdb",
    "test": SOURCE_LMDB_ROOT / "tmqm_totE_matched_test.lmdb",
}

TARGETS = {
    "delta_mlp_bob": TargetSpec(
        target_id="delta_mlp_bob",
        prose_label="Delta^MLP_BoB_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
        target_column="Delta_MLP_BoB-ref-BoE",
        lmdb_prefix="tmqm_delta_mlp_bob_ref_boe",
        description="DFT_ref,BoE minus MLP_BoB_ref,BoE residual target.",
    ),
    "delta_mlp_pre": TargetSpec(
        target_id="delta_mlp_pre",
        prose_label="Delta^MLP_PRE_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
        target_column="Delta_MLP_PRE-ref-BoE",
        lmdb_prefix="tmqm_delta_mlp_pre_ref_boe",
        description="DFT_ref,BoE minus MLP_PRE_ref,BoE residual target.",
    ),
    "delta_mlp_pre_plus_a": TargetSpec(
        target_id="delta_mlp_pre_plus_a",
        prose_label="Delta^MLP_PRE+A_ref,BoE",
        csv_path=PROJECT_ROOT
        / "New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
        target_column="Delta_MLP_PRE_plus_A-ref-BoE",
        lmdb_prefix="tmqm_delta_mlp_pre_plus_a_ref_boe",
        description="DFT_ref,BoE minus MLP_PRE+A_ref,BoE residual target.",
    ),
}


def normalize_mol_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(value.item())
        return "_".join(str(x) for x in value.detach().cpu().view(-1).tolist())
    return str(value)


def get_mol_id(data: Any) -> str:
    for attr in ("mol_id", "id", "sid"):
        if hasattr(data, attr):
            return normalize_mol_id(getattr(data, attr))
    attrs = sorted(str(k) for k in getattr(data, "keys", lambda: [])())
    raise KeyError(f"Could not find mol_id/id/sid on LMDB object. Available keys: {attrs}")


def set_energy_target(data: Any, value: float) -> Any:
    target = float(value)
    old_y_relaxed = getattr(data, "y_relaxed", None)
    if isinstance(old_y_relaxed, torch.Tensor):
        data.y_relaxed = old_y_relaxed.new_tensor(target)
    else:
        data.y_relaxed = target

    if hasattr(data, "y"):
        old_y = getattr(data, "y")
        if isinstance(old_y, torch.Tensor):
            data.y = old_y.new_tensor([target]) if old_y.ndim > 0 else old_y.new_tensor(target)
        elif old_y is not None:
            data.y = target
    return data


def read_targets(spec: TargetSpec) -> dict[str, float]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {spec.csv_path}")
    df = pd.read_csv(spec.csv_path)
    required = {"mol_id", spec.target_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{spec.csv_path} is missing columns: {sorted(missing)}")
    df = df[["mol_id", spec.target_column]].dropna().copy()
    df["mol_id"] = df["mol_id"].map(normalize_mol_id)
    if df["mol_id"].duplicated().any():
        duplicated = df.loc[df["mol_id"].duplicated(), "mol_id"].head(10).tolist()
        raise ValueError(f"Duplicate mol_id values in {spec.csv_path}: {duplicated}")
    return dict(zip(df["mol_id"], df[spec.target_column].astype(float)))


def read_existing_targets(lmdb_path: Path) -> list[float]:
    values: list[float] = []
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        for key, raw in txn.cursor():
            if key == b"length":
                continue
            data = pickle.loads(raw)
            value = getattr(data, "y_relaxed")
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().view(-1)[0].item())
            values.append(float(value))
    env.close()
    return values


def rewrite_lmdb_split(
    source: Path,
    destination: Path,
    targets: dict[str, float],
    *,
    force: bool,
    strict_target_coverage: bool,
) -> tuple[list[float], list[str]]:
    if not source.exists():
        raise FileNotFoundError(f"Source LMDB not found: {source}")
    if destination.exists():
        if not force:
            return read_existing_targets(destination), []
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    values: list[float] = []
    missing: list[str] = []
    src_env = lmdb.open(
        str(source),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    dst_env = lmdb.open(str(destination), subdir=False, map_size=1 << 40, lock=True, meminit=False, map_async=True)

    with src_env.begin(write=False) as src_txn, dst_env.begin(write=True) as dst_txn:
        count = 0
        for key, raw in src_txn.cursor():
            if key == b"length":
                continue
            data = pickle.loads(raw)
            mol_id = get_mol_id(data)
            if mol_id not in targets:
                missing.append(mol_id)
                continue
            data = set_energy_target(data, targets[mol_id])
            dst_txn.put(str(count).encode("ascii"), pickle.dumps(data, protocol=-1))
            values.append(float(targets[mol_id]))
            count += 1
        dst_txn.put(b"length", pickle.dumps(count, protocol=-1))

    src_env.close()
    dst_env.sync()
    dst_env.close()
    if missing and strict_target_coverage:
        raise KeyError(
            f"{len(missing)} molecules from {source} were missing in target CSV. "
            f"First missing IDs: {missing[:10]}"
        )
    if missing:
        print(
            f"WARNING: skipped {len(missing)} molecules from {source} missing in target CSV. "
            f"First missing IDs: {missing[:10]}"
        )
    return values, missing


def find_template_config() -> Path:
    candidates = [
        Path(os.environ["TEMPLATE_CONFIG"]) if "TEMPLATE_CONFIG" in os.environ else None,
        TRAIN_ROOT / "configs/gemnet_train_on_totE.yml",
        PROJECT_ROOT / "Atomic_latent_embeddings/CODEX_embeddings_analysis/trillium_train_configs/gemnet_train_on_totE.yml",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    tried = [str(path) for path in candidates if path is not None]
    raise FileNotFoundError(f"No template config found. Tried: {tried}")


def schnet_model_block(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "schnet",
        "use_pbc": False,
        "regress_forces": False,
        "otf_graph": True,
        "hidden_channels": args.hidden_channels,
        "num_filters": args.num_filters,
        "num_interactions": args.num_interactions,
        "num_gaussians": args.num_gaussians,
        "cutoff": args.cutoff,
        "readout": "sum",
    }


def default_run_root(model: str) -> Path:
    if model == "gemnet":
        return Path(os.environ.get("RUN_ROOT", "/scratch/gameelka/gemnet_mlp_residuals_2026_06"))
    if model == "schnet":
        return Path(os.environ.get("RUN_ROOT", "/scratch/gameelka/schnet_mlp_residuals_2026_06"))
    raise ValueError(model)


def write_config(
    model: str,
    spec: TargetSpec,
    run_root: Path,
    train_mean: float,
    train_std: float,
    args: argparse.Namespace,
) -> Path:
    template = find_template_config()
    cfg = yaml.safe_load(template.read_text())
    run_name = f"{model}_{spec.target_id}_ref_boe_seed_{args.seed}"

    cfg["dataset"][0]["src"] = str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_train.lmdb")
    cfg["dataset"][0]["normalize_labels"] = True
    cfg["dataset"][0]["target_mean"] = float(train_mean)
    cfg["dataset"][0]["target_std"] = float(train_std)
    cfg["dataset"][1]["src"] = str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_val.lmdb")
    cfg["dataset"][2]["src"] = str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_test.lmdb")

    cfg.setdefault("cmd", {})
    cfg["cmd"]["identifier"] = run_name
    cfg["cmd"]["seed"] = args.seed

    cfg.setdefault("task", {})
    cfg["task"]["description"] = f"{model} run on {spec.prose_label}. {spec.description}"

    if model == "schnet":
        cfg["model"] = schnet_model_block(args)

    cfg.setdefault("optim", {})
    cfg["optim"]["max_epochs"] = args.max_epochs
    cfg["optim"]["eval_every"] = args.eval_every
    cfg["optim"]["batch_size"] = args.batch_size
    cfg["optim"]["eval_batch_size"] = args.eval_batch_size
    cfg["optim"]["num_workers"] = args.num_workers

    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{run_name}.yml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("template config:", template)
    return out


def write_preflight(run_root: Path, config_paths: list[Path]) -> None:
    preflight = run_root / "preflight_check.py"
    config_lines = ",\n    ".join(repr(str(path)) for path in config_paths)
    preflight.write_text(
        f"""from pathlib import Path
import yaml

for cfg_name in [
    {config_lines},
]:
    cfg_path = Path(cfg_name)
    cfg = yaml.safe_load(cfg_path.read_text())
    print("\\nconfig:", cfg_path)
    print("identifier:", cfg.get("cmd", {{}}).get("identifier"))
    print("model:", cfg.get("model", {{}}).get("name"))
    for idx, split in enumerate(["train", "val", "test"]):
        src = Path(cfg["dataset"][idx]["src"])
        print(split + ":", src, src.exists())
    print("target_mean:", cfg["dataset"][0].get("target_mean"))
    print("target_std:", cfg["dataset"][0].get("target_std"))
    print("max_epochs:", cfg["optim"].get("max_epochs"))
    print("eval_every:", cfg["optim"].get("eval_every"))
    print("batch_size:", cfg["optim"].get("batch_size"))
"""
    )
    print("preflight:", preflight)


def write_manifest(run_root: Path, model: str, rows: list[dict[str, Any]]) -> None:
    out_dir = run_root / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model}_mlp_residual_targets_manifest.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("manifest:", out)


def selected_targets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(TARGETS)
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown target(s): {unknown}. Available: {sorted(TARGETS)}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gemnet", "schnet"], required=True)
    parser.add_argument("--targets", default="all", help=f"Comma-separated targets from {', '.join(TARGETS)} or all")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "101")))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Rewrite existing generated LMDB files.")
    parser.add_argument(
        "--strict-target-coverage",
        action="store_true",
        help="Fail if source LMDB molecules are missing from target CSV. Default skips missing rows.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=6.0)
    return parser.parse_args()


def fill_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.model == "gemnet":
        args.max_epochs = args.max_epochs or int(os.environ.get("MAX_EPOCHS", "80"))
        args.eval_every = args.eval_every or int(os.environ.get("EVAL_EVERY", "5000"))
        args.batch_size = args.batch_size or int(os.environ.get("BATCH_SIZE", "16"))
        args.eval_batch_size = args.eval_batch_size or int(os.environ.get("EVAL_BATCH_SIZE", "16"))
    else:
        args.max_epochs = args.max_epochs or int(os.environ.get("MAX_EPOCHS", "80"))
        args.eval_every = args.eval_every or int(os.environ.get("EVAL_EVERY", "5000"))
        args.batch_size = args.batch_size or int(os.environ.get("BATCH_SIZE", "16"))
        args.eval_batch_size = args.eval_batch_size or int(os.environ.get("EVAL_BATCH_SIZE", "16"))
    return args


def main() -> None:
    args = fill_model_defaults(parse_args())
    run_root = args.run_root or default_run_root(args.model)
    requested = selected_targets(args.targets)

    print("project root:", PROJECT_ROOT)
    print("train root:", TRAIN_ROOT)
    print("run root:", run_root)
    print("source lmdb root:", SOURCE_LMDB_ROOT)
    print("output lmdb root:", OUTPUT_LMDB_ROOT)
    print("model:", args.model)
    print("targets:", requested)

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "jobs").mkdir(parents=True, exist_ok=True)
    OUTPUT_LMDB_ROOT.mkdir(parents=True, exist_ok=True)

    config_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for name in requested:
        spec = TARGETS[name]
        print(f"\nRUN TARGET: {spec.prose_label}")
        targets = read_targets(spec)

        split_values: dict[str, list[float]] = {}
        missing_counts: dict[str, int] = {}
        for split, source in SOURCE_LMDBS.items():
            destination = OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_{split}.lmdb"
            values, missing = rewrite_lmdb_split(
                source,
                destination,
                targets,
                force=args.force,
                strict_target_coverage=args.strict_target_coverage,
            )
            split_values[split] = values
            missing_counts[split] = len(missing)
            print(f"{split}: {destination} n={len(values)} skipped_missing={len(missing)}")

        train_values = np.asarray(split_values["train"], dtype=float)
        train_mean = float(train_values.mean())
        train_std = float(train_values.std(ddof=0))
        if train_std <= 0:
            raise ValueError(f"{spec.target_id}: non-positive train std {train_std}")

        config_path = write_config(args.model, spec, run_root, train_mean, train_std, args)
        config_paths.append(config_path)
        manifest_rows.append(
            {
                "model": args.model,
                "target_id": spec.target_id,
                "prose_label": spec.prose_label,
                "run_name": config_path.stem,
                "config_path": str(config_path),
                "target_csv": str(spec.csv_path),
                "target_column": spec.target_column,
                "lmdb_train": str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_train.lmdb"),
                "lmdb_val": str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_val.lmdb"),
                "lmdb_test": str(OUTPUT_LMDB_ROOT / f"{spec.lmdb_prefix}_test.lmdb"),
                "train_mean": train_mean,
                "train_std": train_std,
                "missing_train": missing_counts["train"],
                "missing_val": missing_counts["val"],
                "missing_test": missing_counts["test"],
            }
        )
        print("train target mean:", train_mean)
        print("train target std:", train_std)
        print("config:", config_path)

    write_preflight(run_root, config_paths)
    write_manifest(run_root, args.model, manifest_rows)


if __name__ == "__main__":
    main()
