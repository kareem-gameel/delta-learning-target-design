#!/usr/bin/env python3
"""Create canonical target CSVs and registry for the expanded target panel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from expanded_panel_common import DATASET_DIR, DEFAULT_PRE_A_OUTPUT, TARGETS, write_registry


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_two_column(src: Path, src_col: str, dst: Path, dst_col: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source target file: {src}")
    df = pd.read_csv(src, usecols=["mol_id", src_col]).dropna().drop_duplicates("mol_id", keep="first")
    df["mol_id"] = df["mol_id"].astype(str)
    out = df.rename(columns={src_col: dst_col})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    log(f"WROTE: {dst} rows={len(out)} column={dst_col}")


def promote_pre_plus_a(pre_plus_a_dir: Path, *, required: bool) -> None:
    source = pre_plus_a_dir / "tmQM__PRE_plus_A_ref_BoE__pred_and_residual.csv"
    delta_source = pre_plus_a_dir / "tmQM__Delta_PRE_plus_A-ref-BoE.csv"
    if not source.exists():
        message = (
            "PRE+A source is not available locally yet: "
            f"{source}. Run the rsync command in the README, then rerun this script."
        )
        if required:
            raise FileNotFoundError(message)
        log(f"SKIP: {message}")
        return

    df = pd.read_csv(source)
    required_cols = {"mol_id", "E_PRE_plus_A_ref_BoE", "Delta_PRE_plus_A-ref-BoE"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")

    baseline = (
        df[["mol_id", "E_PRE_plus_A_ref_BoE"]]
        .dropna()
        .drop_duplicates("mol_id", keep="first")
        .rename(columns={"E_PRE_plus_A_ref_BoE": "PRE_plus_A_ref-BoE"})
    )
    baseline["mol_id"] = baseline["mol_id"].astype(str)
    baseline_path = DATASET_DIR / "tmQM__PRE_plus_A-ref-BoE.csv"
    baseline.to_csv(baseline_path, index=False)
    log(f"WROTE: {baseline_path} rows={len(baseline)}")

    if delta_source.exists():
        delta = pd.read_csv(delta_source)
    else:
        delta = df[["mol_id", "Delta_PRE_plus_A-ref-BoE"]].copy()
    if "Delta_PRE_plus_A-ref-BoE" not in delta.columns:
        raise ValueError(f"{delta_source} missing Delta_PRE_plus_A-ref-BoE")
    delta = delta[["mol_id", "Delta_PRE_plus_A-ref-BoE"]].dropna().drop_duplicates("mol_id", keep="first")
    delta["mol_id"] = delta["mol_id"].astype(str)
    delta_path = DATASET_DIR / "tmQM__Delta_PRE_plus_A-ref-BoE.csv"
    delta.to_csv(delta_path, index=False)
    log(f"WROTE: {delta_path} rows={len(delta)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plus-a-dir", type=Path, default=DEFAULT_PRE_A_OUTPUT)
    parser.add_argument("--require-pre-plus-a", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    write_two_column(
        DATASET_DIR / "tmQM__Delta_BoB-ref-BoE.csv",
        "E_BoE_plus_BoB_tmQM",
        DATASET_DIR / "tmQM__BoB-ref-BoE.csv",
        "BoB_ref-BoE",
    )
    write_two_column(
        DATASET_DIR / "tmQM__Delta_PRE-ref-BoE.csv",
        "E_BoE_plus_PRE_tmQM",
        DATASET_DIR / "tmQM__PRE-ref-BoE.csv",
        "PRE_ref-BoE",
    )
    promote_pre_plus_a(args.pre_plus_a_dir, required=args.require_pre_plus_a)

    registry = write_registry()
    log("Target registry:")
    print(registry[["target_id", "target_label", "column", "exists"]].to_string(index=False))
    missing = registry.loc[~registry["exists"], "target_id"].tolist()
    if missing:
        log(f"Missing targets remain: {missing}")
    expected = {target.target_id for target in TARGETS}
    got = set(registry["target_id"])
    if expected != got:
        raise AssertionError(f"Registry target mismatch: expected={sorted(expected)} got={sorted(got)}")


if __name__ == "__main__":
    main()
