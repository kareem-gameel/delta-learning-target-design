from __future__ import annotations

import csv
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
import yaml


SEED = int(os.environ.get("SEED", "101"))
LABEL = "b_comp_short_long_residuals"
RUN_NAME = f"{LABEL}_seed_{SEED}"
TARGET_COLUMN = "E_resid_after_BoE_PRE_PRE6p_tmQM"

PROJECT_ROOT = Path(os.environ.get("DELTAML_PROJECT", "/home/kareem/deltaML_project"))
TRAIN_ROOT = Path(os.environ.get("FAIRCHEM_TRAIN_ROOT", str(PROJECT_ROOT / "train")))
RUN_ROOT = Path(
    os.environ.get(
        "RUN_ROOT",
        f"/home/kareem/{LABEL}_reproducibility_2026_05",
    )
)

TARGET_CSV = Path(
    os.environ.get(
        "TARGET_CSV",
        str(
            PROJECT_ROOT
            / "pre_longrange/PRE_6+/processed/b_comp_short_long/"
            "tmQM__b_comp_short_long__pred_and_residual.csv"
        ),
    )
)

SOURCE_LMDB_ROOT = Path(
    os.environ.get("SOURCE_LMDB_ROOT", str(PROJECT_ROOT / "lmdb_files/tmqm_totE"))
)
SOURCE_LMDBS = {
    "train": SOURCE_LMDB_ROOT / "tmqm_totE_matched_train.lmdb",
    "val": SOURCE_LMDB_ROOT / "tmqm_totE_matched_val.lmdb",
    "test": SOURCE_LMDB_ROOT / "tmqm_totE_matched_test.lmdb",
}

OUTPUT_LMDB_DIR = Path(
    os.environ.get(
        "OUTPUT_LMDB_DIR",
        str(PROJECT_ROOT / "lmdb_files/tmqm_b_comp_short_long_residuals"),
    )
)
OUTPUT_LMDB_PREFIX = "tmqm_b_comp_short_long_residuals"

CONFIG_DIR = RUN_ROOT / "configs"
JOBS_DIR = RUN_ROOT / "jobs"
RUNS_DIR = RUN_ROOT / "runs"
SUMMARIES_DIR = RUN_ROOT / "summaries"
EVAL_DIR = RUN_ROOT / "eval"

SCALE_FILE = Path(
    os.environ.get(
        "GEMNET_SCALE_FILE",
        "/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/train/scaling_factors/gemnet-dT.json",
    )
)

TEMPLATE_CONFIG_CANDIDATES = [
    Path(os.environ["TEMPLATE_CONFIG"]) if "TEMPLATE_CONFIG" in os.environ else None,
    Path("/home/kareem/PRE6plus_BoE_residuals_reproducibility_2026_04/configs/PRE6plus_BoE_residuals_seed_101.yml"),
    Path("/home/kareem/PRE6plus_residuals_reproducibility_2026_04/configs/PRE6plus_residuals_seed_101.yml"),
    TRAIN_ROOT / "configs/gemnet_train_on_pre6plus_boe_residuals.yml",
    TRAIN_ROOT / "configs/gemnet_train_on_pre_boe_residuals.yml",
]


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


def read_targets() -> dict[str, float]:
    if not TARGET_CSV.exists():
        raise FileNotFoundError(
            f"Target CSV not found: {TARGET_CSV}\n"
            "Upload tmQM__b_comp_short_long__pred_and_residual.csv before running this script."
        )
    df = pd.read_csv(TARGET_CSV)
    required = {"mol_id", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{TARGET_CSV} is missing required columns: {sorted(missing)}")
    df = df[["mol_id", TARGET_COLUMN]].dropna().copy()
    df["mol_id"] = df["mol_id"].map(normalize_mol_id)
    if df["mol_id"].duplicated().any():
        duplicated = df.loc[df["mol_id"].duplicated(), "mol_id"].head(10).tolist()
        raise ValueError(f"Duplicate mol_id values in {TARGET_CSV}: {duplicated}")
    return dict(zip(df["mol_id"], df[TARGET_COLUMN].astype(float)))


def rewrite_lmdb_split(source: Path, destination: Path, targets: dict[str, float]) -> list[float]:
    if not source.exists():
        raise FileNotFoundError(f"Source LMDB not found: {source}")
    if destination.exists():
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
    dst_env = lmdb.open(
        str(destination),
        subdir=False,
        map_size=1 << 40,
        lock=True,
        meminit=False,
        map_async=True,
    )

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
            dst_txn.put(key, pickle.dumps(data, protocol=-1))
            values.append(target)
            count += 1
        dst_txn.put(b"length", pickle.dumps(count, protocol=-1))

    src_env.close()
    dst_env.sync()
    dst_env.close()

    if missing:
        raise KeyError(
            f"{len(missing)} molecules from {source} were missing in target CSV. "
            f"First missing IDs: {missing[:10]}"
        )
    return values


def find_template_config() -> Path:
    for candidate in TEMPLATE_CONFIG_CANDIDATES:
        if candidate is not None and candidate.exists():
            return candidate
    tried = [str(p) for p in TEMPLATE_CONFIG_CANDIDATES if p is not None]
    raise FileNotFoundError(f"No template config found. Tried: {tried}")


def write_config(train_mean: float, train_std: float) -> Path:
    template = find_template_config()
    cfg = yaml.safe_load(template.read_text())

    cfg["dataset"][0]["src"] = str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_train.lmdb")
    cfg["dataset"][0]["normalize_labels"] = True
    cfg["dataset"][0]["target_mean"] = float(train_mean)
    cfg["dataset"][0]["target_std"] = float(train_std)
    cfg["dataset"][1]["src"] = str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_val.lmdb")
    cfg["dataset"][2]["src"] = str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_test.lmdb")

    cfg.setdefault("cmd", {})
    cfg["cmd"]["identifier"] = RUN_NAME
    cfg["cmd"]["seed"] = SEED

    cfg.setdefault("model", {})
    cfg["model"]["scale_file"] = str(SCALE_FILE)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"{RUN_NAME}.yml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("template config:", template)
    return out


def write_helpers(config_path: Path) -> None:
    for path in (JOBS_DIR, RUNS_DIR, SUMMARIES_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)

    manifest = SUMMARIES_DIR / f"{LABEL}_single_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "seed",
                "run_name",
                "config_path",
                "root",
                "lmdb_train",
                "lmdb_val",
                "lmdb_test",
                "target_column",
                "target_csv",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": LABEL,
                "seed": SEED,
                "run_name": RUN_NAME,
                "config_path": str(config_path),
                "root": str(RUN_ROOT),
                "lmdb_train": str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_train.lmdb"),
                "lmdb_val": str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_val.lmdb"),
                "lmdb_test": str(OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_test.lmdb"),
                "target_column": TARGET_COLUMN,
                "target_csv": str(TARGET_CSV),
            }
        )

    check = RUN_ROOT / "preflight_check.py"
    check.write_text(
        f"""from pathlib import Path
import yaml

cfg_path = Path("{config_path}")
cfg = yaml.safe_load(cfg_path.read_text())
print("config:", cfg_path)
print("seed:", cfg.get("cmd", {{}}).get("seed"))
print("identifier:", cfg.get("cmd", {{}}).get("identifier"))
print("train:", cfg["dataset"][0]["src"], Path(cfg["dataset"][0]["src"]).exists())
print("val:", cfg["dataset"][1]["src"], Path(cfg["dataset"][1]["src"]).exists())
print("test:", cfg["dataset"][2]["src"], Path(cfg["dataset"][2]["src"]).exists())
print("target_mean:", cfg["dataset"][0].get("target_mean"))
print("target_std:", cfg["dataset"][0].get("target_std"))
print("scale_file:", cfg.get("model", {{}}).get("scale_file"), Path(cfg.get("model", {{}}).get("scale_file", "")).exists())
print("max_epochs:", cfg["optim"]["max_epochs"])
"""
    )

    submit = RUN_ROOT / "run_nohup.sh"
    submit.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="{RUN_ROOT}"
TRAIN_ROOT="{TRAIN_ROOT}"
CFG_PATH="{config_path}"
RUN_NAME="{RUN_NAME}"
SEED="{SEED}"
PYTHON_BIN="${{PYTHON_BIN:-$HOME/miniconda3/envs/fairchem-v110/bin/python}}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate fairchem-v110
unset PYTHONPATH
export PATH="$HOME/miniconda3/envs/fairchem-v110/bin:$PATH"
hash -r

unset WORLD_SIZE RANK LOCAL_RANK SLURM_PROCID SLURM_LOCALID
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG_PATH="$RUN_DIR/$RUN_NAME.train.log"
NOHUP_LOG="$RUN_DIR/$RUN_NAME.nohup.log"
PID_PATH="$RUN_DIR/$RUN_NAME.pid"

mkdir -p "$RUN_DIR"
cd "$TRAIN_ROOT"

nohup "$PYTHON_BIN" -u main.py \\
  --mode train \\
  --config-yml "$CFG_PATH" \\
  --identifier "$RUN_NAME" \\
  --seed "$SEED" \\
  --run-dir "$RUN_DIR" \\
  --timestamp-id "$RUN_NAME" \\
  > "$NOHUP_LOG" 2>&1 &

echo "$!" > "$PID_PATH"
echo "pid:      $(cat "$PID_PATH")"
echo "nohup:    $NOHUP_LOG"
echo "train:    $LOG_PATH"
echo "config:   $CFG_PATH"
echo
echo "Follow with:"
echo "tail -f '$NOHUP_LOG'"
"""
    )
    submit.chmod(0o750)


def main() -> None:
    print("run name:", RUN_NAME)
    print("project root:", PROJECT_ROOT)
    print("train root:", TRAIN_ROOT)
    print("run root:", RUN_ROOT)
    print("target csv:", TARGET_CSV)
    print("target column:", TARGET_COLUMN)
    print("source lmdb root:", SOURCE_LMDB_ROOT)
    print("output lmdb dir:", OUTPUT_LMDB_DIR)
    print("scale file:", SCALE_FILE)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_LMDB_DIR.mkdir(parents=True, exist_ok=True)

    targets = read_targets()
    split_values: dict[str, list[float]] = {}
    for split, source in SOURCE_LMDBS.items():
        destination = OUTPUT_LMDB_DIR / f"{OUTPUT_LMDB_PREFIX}_{split}.lmdb"
        values = rewrite_lmdb_split(source, destination, targets)
        split_values[split] = values
        print(f"wrote {split}: {destination} n={len(values)}")

    train_values = np.asarray(split_values["train"], dtype=float)
    train_mean = float(train_values.mean())
    train_std = float(train_values.std(ddof=0))
    if train_std <= 0:
        raise ValueError(f"Non-positive train target std: {train_std}")

    config_path = write_config(train_mean, train_std)
    write_helpers(config_path)

    print("train target mean:", train_mean)
    print("train target std:", train_std)
    print("config:", config_path)
    print("preflight:", RUN_ROOT / "preflight_check.py")
    print("launcher:", RUN_ROOT / "run_nohup.sh")


if __name__ == "__main__":
    main()
