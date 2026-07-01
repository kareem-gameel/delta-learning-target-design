#!/usr/bin/env python3
"""Create QM9 predict configs and evaluation manifest for linear Delta^PRE+A.

This prepares, but does not run, QM9 prediction for:

- GemNet(Delta^PRE+A_ref,BoE)
- SchNet(Delta^PRE+A_ref,BoE)

It expects the trained expanded-target runs to already exist on Trillium.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


RUNS = {
    "gemnet": {
        "model_id": "gemnet_Delta_PRE_plus_A_ref_BoE",
        "display_name": "PRE+A_ref,BoE + GemNet(Delta^PRE+A_ref,BoE)",
        "run_name": "gemnet_delta_pre_plus_a_ref_boe_seed_101",
        "main": "/scratch/gameelka/deltaML_project/train/main.py",
        "run_root_arg": "gemnet_run_root",
        "checkpoint_candidates": [
            "runs/delta_pre_plus_a_ref_boe_seed_101/checkpoints/gemnet_delta_pre_plus_a_ref_boe_seed_101/best_checkpoint.pt",
            "runs/seed_101/checkpoints/gemnet_delta_pre_plus_a_ref_boe_seed_101/best_checkpoint.pt",
        ],
    },
    "schnet": {
        "model_id": "schnet_Delta_PRE_plus_A_ref_BoE",
        "display_name": "PRE+A_ref,BoE + SchNet(Delta^PRE+A_ref,BoE)",
        "run_name": "schnet_delta_pre_plus_a_ref_boe_seed_101",
        "main": "/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/main_schnet.py",
        "run_root_arg": "schnet_run_root",
        "checkpoint_candidates": [
            "runs/seed_101/checkpoints/schnet_delta_pre_plus_a_ref_boe_seed_101/best_checkpoint.pt",
            "runs/schnet_delta_pre_plus_a_ref_boe_seed_101/checkpoints/schnet_delta_pre_plus_a_ref_boe_seed_101/best_checkpoint.pt",
        ],
    },
}


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def replace_dataset_srcs(cfg: dict, qm9_lmdb: str) -> None:
    for key in ["dataset", "val_dataset", "test_dataset"]:
        value = cfg.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "src" in item:
                    item["src"] = qm9_lmdb
        elif isinstance(value, dict) and "src" in value:
            value["src"] = qm9_lmdb


def first_existing(root: Path, rels: list[str]) -> Path | None:
    for rel in rels:
        p = root / rel
        if p.exists():
            return p
    return None


def make_config(model_key: str, run_root: Path, qm9_lmdb: Path, out_root: Path) -> dict[str, str]:
    spec = RUNS[model_key]
    run_name = spec["run_name"]
    source_config = run_root / "configs" / f"{run_name}.yml"
    require(source_config, f"{model_key} source config")
    require(qm9_lmdb, "QM9 LMDB")
    checkpoint = first_existing(run_root, spec["checkpoint_candidates"])
    if checkpoint is None:
        candidates = "\n".join(str(run_root / rel) for rel in spec["checkpoint_candidates"])
        raise FileNotFoundError(f"Missing {model_key} checkpoint. Checked:\n{candidates}")

    cfg = yaml.safe_load(source_config.read_text())
    replace_dataset_srcs(cfg, str(qm9_lmdb))
    cfg["checkpoint"] = str(checkpoint)

    pred_id = f"{run_name}_qm9"
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_config = cfg_dir / f"{pred_id}.yml"
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    pred_root = out_root / "predict"
    npz_path = pred_root / "results" / pred_id / "is2re_predictions.npz"
    return {
        "model_key": model_key,
        "model_id": spec["model_id"],
        "display_name": spec["display_name"],
        "run_name": run_name,
        "prediction_identifier": pred_id,
        "source_config": str(source_config),
        "predict_config": str(out_config),
        "checkpoint": str(checkpoint),
        "main": str(spec["main"]),
        "predict_run_dir": str(pred_root),
        "expected_npz_path": str(npz_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemnet-run-root", type=Path, default=Path("/scratch/gameelka/gemnet_expanded_ref_boe_targets_2026_06"))
    parser.add_argument("--schnet-run-root", type=Path, default=Path("/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06"))
    parser.add_argument(
        "--qm9-lmdb",
        type=Path,
        default=Path("/scratch/gameelka/deltaML_project/lmdb_files/qm9_psi4_totE/qm9_psi4_dft_ref-corr_neutral.lmdb"),
    )
    parser.add_argument(
        "--pre-plus-a-baseline-csv",
        type=Path,
        default=Path(
            "/scratch/gameelka/deltaML_project/New_Analysis/pre_plus_a/outputs_qm9/qm9_pre_plus_a_ridge_predictions_dft_ref_boe.csv"
        ),
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("/scratch/gameelka/deltaML_project/New_Analysis/source_artifacts/boe/QM9__BoE_tmQM_ridge__residuals.csv"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("/scratch/gameelka/deltaML_project/New_Analysis/linear_prea_qm9_transfer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [
        make_config("gemnet", args.gemnet_run_root, args.qm9_lmdb, args.out_root),
        make_config("schnet", args.schnet_run_root, args.qm9_lmdb, args.out_root),
    ]
    config_df = pd.DataFrame(rows)
    config_path = args.out_root / "linear_prea_qm9_predict_manifest.csv"
    config_df.to_csv(config_path, index=False)

    eval_rows = []
    for row in rows:
        eval_rows.append(
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "mode": "plus_baseline",
                "npz_path": row["expected_npz_path"],
                "sid_lmdb_path": str(args.qm9_lmdb),
                "baseline_csv": str(args.pre_plus_a_baseline_csv),
                "baseline_columns": "PRE_plus_A_ref-BoE",
                "true_csv": str(args.truth_csv),
                "true_column": "E_resid_BoE_tmQM",
                "notes": "QM9 transfer for linear Delta^PRE+A_ref,BoE.",
            }
        )
    eval_path = args.out_root / "linear_prea_qm9_eval_manifest.csv"
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False)

    print("WROTE:", config_path)
    print("WROTE:", eval_path)
    print("\nPrediction commands:")
    for row in rows:
        print(
            "\n"
            f"mkdir -p {row['predict_run_dir']}\n"
            "cd /scratch/gameelka/deltaML_project/train\n"
            f"\"$PY\" -u {row['main']} \\\n"
            "  --mode predict \\\n"
            f"  --config-yml {row['predict_config']} \\\n"
            f"  --checkpoint {row['checkpoint']} \\\n"
            f"  --identifier {row['prediction_identifier']} \\\n"
            f"  --run-dir {row['predict_run_dir']} \\\n"
            f"  --timestamp-id {row['prediction_identifier']} \\\n"
            f"  2>&1 | tee {row['predict_run_dir']}/{row['prediction_identifier']}.log"
        )


if __name__ == "__main__":
    main()
