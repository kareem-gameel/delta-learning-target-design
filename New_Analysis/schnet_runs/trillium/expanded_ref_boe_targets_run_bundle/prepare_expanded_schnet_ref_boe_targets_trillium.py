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
    label: str
    prose_label: str
    csv_path: Path
    target_column: str
    lmdb_prefix: str
    description: str


PROJECT_ROOT = Path(os.environ.get("DELTAML_PROJECT", "/scratch/gameelka/deltaML_project"))
TRAIN_ROOT = Path(os.environ.get("FAIRCHEM_TRAIN_ROOT", str(PROJECT_ROOT / "train")))
RUN_ROOT = Path(os.environ.get("RUN_ROOT", "/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06"))
SOURCE_LMDB_ROOT = Path(os.environ.get("SOURCE_LMDB_ROOT", str(PROJECT_ROOT / "lmdb_files/tmqm_totE")))

SOURCE_LMDBS = {
    "train": SOURCE_LMDB_ROOT / "tmqm_totE_matched_train.lmdb",
    "val": SOURCE_LMDB_ROOT / "tmqm_totE_matched_val.lmdb",
    "test": SOURCE_LMDB_ROOT / "tmqm_totE_matched_test.lmdb",
}

TARGETS = {
    "bob_ref_boe": TargetSpec(
        label="bob_ref_boe",
        prose_label="BoB_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/datasets/tmQM__BoB-ref-BoE.csv",
        target_column="BoB_ref-BoE",
        lmdb_prefix="tmqm_bob_ref_boe",
        description="BoB ridge fitted target after reference correction and BoE removal.",
    ),
    "pre_ref_boe": TargetSpec(
        label="pre_ref_boe",
        prose_label="PRE_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/datasets/tmQM__PRE-ref-BoE.csv",
        target_column="PRE_ref-BoE",
        lmdb_prefix="tmqm_pre_ref_boe",
        description="PRE ridge fitted target after reference correction and BoE removal.",
    ),
    "pre_plus_a_ref_boe": TargetSpec(
        label="pre_plus_a_ref_boe",
        prose_label="PRE+A_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv",
        target_column="PRE_plus_A_ref-BoE",
        lmdb_prefix="tmqm_pre_plus_a_ref_boe",
        description="PRE plus central-angular ridge fitted target after reference correction and BoE removal.",
    ),
    "delta_pre_plus_a_ref_boe": TargetSpec(
        label="delta_pre_plus_a_ref_boe",
        prose_label="Delta^PRE+A_ref,BoE",
        csv_path=PROJECT_ROOT / "New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv",
        target_column="Delta_PRE_plus_A-ref-BoE",
        lmdb_prefix="tmqm_delta_pre_plus_a_ref_boe",
        description="DFT_ref,BoE minus PRE+A_ref,BoE residual target.",
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
        raise KeyError(f"{spec.csv_path} is missing required columns: {sorted(missing)}")
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
        cursor = src_txn.cursor()
        count = 0
        for key, raw in cursor:
            if key == b"length":
                continue
            data = pickle.loads(raw)
            mol_id = get_mol_id(data)
            if mol_id not in targets:
                missing.append(mol_id)
                continue
            target = targets[mol_id]
            data = set_energy_target(data, target)
            # When target CSVs miss a few molecules, retained entries must be
            # renumbered densely. FairChem indexes LMDB keys from 0..length-1.
            dst_txn.put(str(count).encode("ascii"), pickle.dumps(data, protocol=-1))
            values.append(target)
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
    tried = [str(p) for p in candidates if p is not None]
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


def write_config(spec: TargetSpec, lmdb_dir: Path, train_mean: float, train_std: float, args: argparse.Namespace) -> Path:
    template = find_template_config()
    cfg = yaml.safe_load(template.read_text())
    run_name = f"schnet_{spec.label}_seed_{args.seed}"

    cfg["dataset"][0]["src"] = str(lmdb_dir / f"{spec.lmdb_prefix}_train.lmdb")
    cfg["dataset"][0]["normalize_labels"] = True
    cfg["dataset"][0]["target_mean"] = float(train_mean)
    cfg["dataset"][0]["target_std"] = float(train_std)
    cfg["dataset"][1]["src"] = str(lmdb_dir / f"{spec.lmdb_prefix}_val.lmdb")
    cfg["dataset"][2]["src"] = str(lmdb_dir / f"{spec.lmdb_prefix}_test.lmdb")

    cfg.setdefault("cmd", {})
    cfg["cmd"]["identifier"] = run_name
    cfg["cmd"]["seed"] = args.seed

    cfg.setdefault("task", {})
    cfg["task"]["description"] = f"SchNet controlled run on {spec.prose_label}. {spec.description}"

    cfg["model"] = schnet_model_block(args)

    cfg.setdefault("optim", {})
    cfg["optim"]["max_epochs"] = args.max_epochs
    cfg["optim"]["eval_every"] = args.eval_every
    cfg["optim"]["batch_size"] = args.batch_size
    cfg["optim"]["eval_batch_size"] = args.eval_batch_size
    cfg["optim"]["num_workers"] = args.num_workers

    config_dir = RUN_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{run_name}.yml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("template config:", template)
    return out


def write_preflight(config_paths: list[Path]) -> None:
    preflight = RUN_ROOT / "preflight_check.py"
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
    print("model:", cfg.get("model"))
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


def write_manifest(rows: list[dict[str, Any]]) -> None:
    out_dir = RUN_ROOT / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "schnet_expanded_ref_boe_targets_manifest.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("manifest:", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="all", help=f"Comma-separated targets from: {', '.join(TARGETS)} or all")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "101")))
    parser.add_argument("--force", action="store_true", help="Rewrite existing generated LMDB files.")
    parser.add_argument(
        "--strict-target-coverage",
        action="store_true",
        help="Fail if source LMDB molecules are missing from a target CSV. Default is to skip missing rows.",
    )
    parser.add_argument("--max-epochs", type=int, default=int(os.environ.get("MAX_EPOCHS", "80")))
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("EVAL_EVERY", "5000")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=6.0)
    return parser.parse_args()


def selected_targets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(TARGETS)
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown target(s): {unknown}. Available: {sorted(TARGETS)}")
    return requested


def main() -> None:
    args = parse_args()
    requested = selected_targets(args.targets)

    print("project root:", PROJECT_ROOT)
    print("train root:", TRAIN_ROOT)
    print("run root:", RUN_ROOT)
    print("source lmdb root:", SOURCE_LMDB_ROOT)
    print("targets:", requested)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "jobs").mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for name in requested:
        spec = TARGETS[name]
        print(f"\nRUN TARGET: {spec.prose_label}")
        lmdb_dir = PROJECT_ROOT / "lmdb_files" / spec.lmdb_prefix
        lmdb_dir.mkdir(parents=True, exist_ok=True)
        targets = read_targets(spec)

        split_values: dict[str, list[float]] = {}
        missing_counts: dict[str, int] = {}
        for split, source in SOURCE_LMDBS.items():
            destination = lmdb_dir / f"{spec.lmdb_prefix}_{split}.lmdb"
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
            raise ValueError(f"{spec.label}: non-positive train std {train_std}")

        config_path = write_config(spec, lmdb_dir, train_mean, train_std, args)
        config_paths.append(config_path)
        run_name = config_path.stem
        manifest_rows.append(
            {
                "target": spec.label,
                "prose_label": spec.prose_label,
                "run_name": run_name,
                "config_path": str(config_path),
                "target_csv": str(spec.csv_path),
                "target_column": spec.target_column,
                "lmdb_train": str(lmdb_dir / f"{spec.lmdb_prefix}_train.lmdb"),
                "lmdb_val": str(lmdb_dir / f"{spec.lmdb_prefix}_val.lmdb"),
                "lmdb_test": str(lmdb_dir / f"{spec.lmdb_prefix}_test.lmdb"),
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

    write_preflight(config_paths)
    write_manifest(manifest_rows)


if __name__ == "__main__":
    main()
