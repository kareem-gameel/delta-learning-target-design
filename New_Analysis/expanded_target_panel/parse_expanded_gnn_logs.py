#!/usr/bin/env python3
"""Parse expanded GemNet/SchNet validation MAEs from training logs.

This parser is intentionally independent of the Trillium directory layout. It
searches recursively for ``*.train.log`` files under the provided run roots and
extracts the best validation ``energy_mae`` for the four expanded GNN targets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TARGET_BY_RUN_TOKEN = {
    "bob_ref_boe": ("bob", "BoB_ref,BoE"),
    "pre_ref_boe": ("pre", "PRE_ref,BoE"),
    "pre_plus_a_ref_boe": ("pre_plus_a", "PRE+A_ref,BoE"),
    "delta_pre_plus_a_ref_boe": ("deltapre_plus_a", "Delta^PRE+A_ref,BoE"),
}

OUTPUT_COLUMNS = [
    "model_id",
    "target_id",
    "target_label",
    "status",
    "n_val_evals",
    "best_epoch",
    "best_step",
    "best_val_mae",
    "last_epoch",
    "last_step",
    "last_val_mae",
    "log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemnet-run-root",
        type=Path,
        default=ROOT / "New_Analysis/gemnet_runs/trillium/expanded_ref_boe_targets_results",
    )
    parser.add_argument(
        "--schnet-run-root",
        type=Path,
        default=ROOT / "New_Analysis/schnet_runs/trillium/expanded_ref_boe_targets_results",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "New_Analysis/expanded_target_panel/outputs/gnn/expanded_gnn_val_mae_summary.csv",
    )
    return parser.parse_args()


def identify_target(log_path: Path) -> tuple[str, str] | None:
    name = log_path.stem
    if name.endswith(".train"):
        name = name[: -len(".train")]
    lower = name.lower()

    # Check longest/specific tokens first so PRE+A is not mistaken for PRE.
    for token in sorted(TARGET_BY_RUN_TOKEN, key=len, reverse=True):
        if token in lower:
            return TARGET_BY_RUN_TOKEN[token]
    return None


def parse_one(model_id: str, path: Path) -> dict[str, object]:
    target = identify_target(path)
    target_id = target[0] if target else None
    target_label = target[1] if target else None

    vals: list[dict[str, float | None]] = []
    mode = None
    last_epoch = None
    last_step = None

    for line in path.read_text(errors="ignore").splitlines():
        epoch_match = re.search(r"epoch:\s*([0-9.eE+-]+)", line)
        step_match = re.search(r"step:\s*([0-9.eE+-]+)", line)
        if epoch_match:
            last_epoch = float(epoch_match.group(1))
        if step_match:
            last_step = float(step_match.group(1))

        low = line.lower()
        if "evaluating on val" in low or "validating" in low:
            mode = "val"
            continue
        if "evaluating on test" in low or "predicting on test" in low:
            mode = "test"
            continue

        mae_match = re.search(r"energy_mae:\s*([0-9.eE+-]+)", line)
        if mae_match and mode == "val":
            vals.append(
                {
                    "epoch": last_epoch,
                    "step": last_step,
                    "val_mae": float(mae_match.group(1)),
                }
            )
            mode = None

    if target_id is None:
        return {
            "model_id": model_id,
            "target_id": None,
            "target_label": None,
            "status": "unknown target",
            "log": str(path),
        }
    if not vals:
        return {
            "model_id": model_id,
            "target_id": target_id,
            "target_label": target_label,
            "status": "no parsed val energy_mae",
            "log": str(path),
        }

    best = min(vals, key=lambda rec: float(rec["val_mae"]))
    last = vals[-1]
    return {
        "model_id": model_id,
        "target_id": target_id,
        "target_label": target_label,
        "status": "ok",
        "n_val_evals": len(vals),
        "best_epoch": best["epoch"],
        "best_step": best["step"],
        "best_val_mae": best["val_mae"],
        "last_epoch": last["epoch"],
        "last_step": last["step"],
        "last_val_mae": last["val_mae"],
        "log": str(path),
    }


def find_logs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.train.log"))


def parse_model(model_id: str, root: Path) -> list[dict[str, object]]:
    return [parse_one(model_id, path) for path in find_logs(root)]


def deduplicate_ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ok = df[df["status"].eq("ok")].copy()
    not_ok = df[~df["status"].eq("ok")].copy()
    if ok.empty:
        return df

    # Keep the run with the most validation evaluations if duplicate logs were
    # downloaded from smoke and full runs into the same folder.
    ok = (
        ok.sort_values(["model_id", "target_id", "n_val_evals", "best_val_mae"], ascending=[True, True, False, True])
        .drop_duplicates(["model_id", "target_id"], keep="first")
        .sort_values(["model_id", "target_id"])
    )
    return pd.concat([ok, not_ok], ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(parse_model("GemNet", args.gemnet_run_root))
    rows.extend(parse_model("SchNet", args.schnet_run_root))
    df = deduplicate_ok_rows(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if df.empty:
        print("No expanded GNN logs found.")
    else:
        cols = [
            "model_id",
            "target_id",
            "target_label",
            "status",
            "n_val_evals",
            "best_epoch",
            "best_step",
            "best_val_mae",
            "last_val_mae",
        ]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"WROTE: {args.out_csv}")


if __name__ == "__main__":
    main()
