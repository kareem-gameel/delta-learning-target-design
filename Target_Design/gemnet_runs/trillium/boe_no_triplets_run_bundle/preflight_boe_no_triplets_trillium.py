#!/usr/bin/env python
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("/scratch/gameelka/BoE_residuals_reproducibility_2026_04/configs/BoE_residuals_seed_101.yml")
DEFAULT_OVERLAY = Path("/scratch/gameelka/fairchem_no_triplets_overlay")


def main() -> None:
    config = Path(os.environ.get("BOE_NO_TRIPLETS_CONFIG", DEFAULT_CONFIG))
    overlay = Path(os.environ.get("FAIRCHEM_OVERLAY_PATH", DEFAULT_OVERLAY))

    print("python:", sys.executable)
    print("config:", config, config.exists())
    print("overlay:", overlay, overlay.exists())

    if overlay.exists():
        sys.path.insert(0, str(overlay))

    import fairchem.core.models.gemnet.layers.interaction_block as ib

    ib_path = Path(inspect.getfile(ib))
    ib_text = ib_path.read_text()
    print("fairchem interaction_block:", ib_path)
    print("no_triplets_patch_present:", "SW_NO_TRIPLETS_ABLATION" in ib_text)

    if not config.exists():
        raise FileNotFoundError(config)

    cfg = yaml.safe_load(config.read_text())
    dataset = cfg.get("dataset")

    if isinstance(dataset, list):
        entries = dataset
    else:
        entries = [
            cfg.get("dataset", {}),
            cfg.get("val_dataset", {}),
            cfg.get("test_dataset", {}),
        ]

    split_names = ["train", "val", "test"]
    for split, entry in zip(split_names, entries):
        src = Path(entry["src"])
        print(f"{split}: {src} {src.exists()}")
        if not src.exists():
            raise FileNotFoundError(src)

    train_entry = entries[0]
    print("target_mean:", train_entry.get("target_mean"))
    print("target_std:", train_entry.get("target_std"))
    print("max_epochs:", cfg.get("optim", {}).get("max_epochs"))
    print("scale_file:", cfg.get("model", {}).get("scale_file"), Path(cfg.get("model", {}).get("scale_file", "")).exists())


if __name__ == "__main__":
    main()
