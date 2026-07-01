#!/usr/bin/env python3
"""Create QM9 predict configs for SchNet linear delta targets.

This prepares, but does not run, QM9 prediction/evaluation for:

- SchNet(Delta^xTB_ref,BoE) + xTB_ref,BoE
- SchNet(Delta^BoB_ref,BoE) + BoB_ref,BoE
- SchNet(Delta^PRE_ref,BoE) + PRE_ref,BoE

The output evaluation manifest is compatible with
New_Analysis/qm9_dft_ref_boe_transfer/run_qm9_dft_ref_boe_transfer.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


TARGETS = {
    "deltaxtb": {
        "model_id": "schnet_Delta_xTB_ref_BoE",
        "display_name": "xTB_ref,BoE + SchNet(Delta^xTB_ref,BoE)",
        "run_name": "schnet_Delta_xTB_ref_BoE_seed_101",
        "baseline_csv": "/scratch/gameelka/deltaML_project/New_Analysis/source_artifacts/boe_xtb/QM9__BoE_xTB_tmQM_ridge__residuals.csv",
        "baseline_columns": "E_resid_BoE_xTB_tmQM_on_QM9",
        "notes": "QM9 transfer for linear Delta^xTB_ref,BoE.",
    },
    "deltabob": {
        "model_id": "schnet_Delta_BoB_ref_BoE",
        "display_name": "BoB_ref,BoE + SchNet(Delta^BoB_ref,BoE)",
        "run_name": "schnet_Delta_BoB_ref_BoE_seed_101",
        "baseline_csv": "/scratch/gameelka/deltaML_project/New_Analysis/source_artifacts/bob/QM9__BoB_tmQM_ridge_on_BoEresid__pred_and_residual.csv",
        "baseline_columns": "E_BoE_plus_BoB_tmQM_on_QM9",
        "notes": "QM9 transfer for linear Delta^BoB_ref,BoE.",
    },
    "deltapre": {
        "model_id": "schnet_Delta_PRE_ref_BoE",
        "display_name": "PRE_ref,BoE + SchNet(Delta^PRE_ref,BoE)",
        "run_name": "schnet_Delta_PRE_ref_BoE_seed_101",
        "baseline_csv": "/scratch/gameelka/deltaML_project/New_Analysis/source_artifacts/pre/QM9__PRE_tmQM_ridge_on_BoEresid__residuals.csv",
        "baseline_columns": "E_BoE_plus_PRE_tmQM",
        "notes": "QM9 transfer for linear Delta^PRE_ref,BoE.",
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


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def checkpoint_candidates(run_root: Path, run_name: str) -> list[Path]:
    return [
        run_root / "runs" / run_name / "checkpoints" / run_name / "best_checkpoint.pt",
        run_root / "runs" / "seed_101" / "checkpoints" / run_name / "best_checkpoint.pt",
        run_root / "checkpoints" / run_name / "best_checkpoint.pt",
    ]


def make_one(target_key: str, args: argparse.Namespace) -> dict[str, str]:
    spec = TARGETS[target_key]
    run_name = spec["run_name"]
    source_config = args.schnet_run_root / "configs" / f"{run_name}.yml"
    require(source_config, f"{target_key} source config")
    require(args.qm9_lmdb, "QM9 LMDB")

    checkpoint = first_existing(checkpoint_candidates(args.schnet_run_root, run_name))
    if checkpoint is None:
        checked = "\n".join(str(p) for p in checkpoint_candidates(args.schnet_run_root, run_name))
        raise FileNotFoundError(f"Missing checkpoint for {target_key}. Checked:\n{checked}")

    baseline_csv = Path(spec["baseline_csv"])
    require(baseline_csv, f"{target_key} QM9 baseline CSV")

    cfg = yaml.safe_load(source_config.read_text())
    replace_dataset_srcs(cfg, str(args.qm9_lmdb))
    cfg["checkpoint"] = str(checkpoint)

    pred_id = f"{run_name}_qm9"
    cfg_dir = args.out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_config = cfg_dir / f"{pred_id}.yml"
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    predict_root = args.out_root / "predict"
    npz_path = predict_root / "results" / pred_id / "is2re_predictions.npz"
    return {
        "target_key": target_key,
        "model_id": spec["model_id"],
        "display_name": spec["display_name"],
        "run_name": run_name,
        "prediction_identifier": pred_id,
        "source_config": str(source_config),
        "predict_config": str(out_config),
        "checkpoint": str(checkpoint),
        "main": str(args.main_schnet),
        "predict_run_dir": str(predict_root),
        "expected_npz_path": str(npz_path),
        "baseline_csv": str(baseline_csv),
        "baseline_columns": spec["baseline_columns"],
        "notes": spec["notes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="all", help="Comma-separated subset: deltaxtb,deltabob,deltapre or all.")
    parser.add_argument("--schnet-run-root", type=Path, default=Path("/scratch/gameelka/schnet_ref_boe_core_2026_06"))
    parser.add_argument(
        "--main-schnet",
        type=Path,
        default=Path("/scratch/gameelka/schnet_ref_boe_core_2026_06/main_schnet.py"),
    )
    parser.add_argument(
        "--qm9-lmdb",
        type=Path,
        default=Path("/scratch/gameelka/deltaML_project/lmdb_files/qm9_psi4_totE/qm9_psi4_dft_ref-corr_neutral.lmdb"),
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("/scratch/gameelka/deltaML_project/New_Analysis/source_artifacts/boe/QM9__BoE_tmQM_ridge__residuals.csv"),
    )
    parser.add_argument("--truth-column", default="E_resid_BoE_tmQM")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/scratch/gameelka/deltaML_project/New_Analysis/schnet_linear_delta_qm9_transfer"),
    )
    return parser.parse_args()


def selected_targets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(TARGETS)
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Valid: {sorted(TARGETS)}")
    return keys


def main() -> None:
    args = parse_args()
    require(args.main_schnet, "SchNet predict entrypoint")
    require(args.truth_csv, "QM9 truth CSV")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = [make_one(key, args) for key in selected_targets(args.targets)]
    predict_manifest = args.out_root / "schnet_linear_delta_qm9_predict_manifest.csv"
    pd.DataFrame(rows).to_csv(predict_manifest, index=False)

    eval_rows = []
    for row in rows:
        eval_rows.append(
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "mode": "plus_baseline",
                "npz_path": row["expected_npz_path"],
                "sid_lmdb_path": str(args.qm9_lmdb),
                "baseline_csv": row["baseline_csv"],
                "baseline_columns": row["baseline_columns"],
                "true_csv": str(args.truth_csv),
                "true_column": args.truth_column,
                "notes": row["notes"],
            }
        )
    eval_manifest = args.out_root / "schnet_linear_delta_qm9_eval_manifest.csv"
    pd.DataFrame(eval_rows).to_csv(eval_manifest, index=False)

    print("WROTE:", predict_manifest)
    print("WROTE:", eval_manifest)
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
