#!/usr/bin/env python3
"""Target definitions for learned GemNet embedding smoothness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "New_Analysis/datasets"
MLP_BASELINE_DIR = ROOT / "New_Analysis/mlp_baselines/outputs"


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    target_label: str
    short_label: str
    csv_path: Path
    column: str
    target_family: str
    optional: bool = False


TARGETS: list[TargetSpec] = [
    TargetSpec(
        "dft_ref",
        "DFT_ref",
        "DFT_ref",
        DATASET_DIR / "tmQM__DFT_ref.csv",
        "DFT_ref",
        "reference",
    ),
    TargetSpec(
        "dft_ref_boe",
        "DFT_ref,BoE",
        "DFT",
        DATASET_DIR / "tmQM__DFT_ref-BoE.csv",
        "DFT_ref-BoE",
        "reference_boe",
    ),
    TargetSpec(
        "xtb_ref_boe",
        "xTB_ref,BoE",
        "xTB",
        DATASET_DIR / "tmQM__xTB_ref-BoE.csv",
        "xTB_ref-BoE",
        "baseline_target",
    ),
    TargetSpec(
        "delta_xtb_ref_boe",
        "Delta^xTB_ref,BoE",
        "Delta^xTB",
        DATASET_DIR / "tmQM__Delta_xTB-ref-BoE.csv",
        "Delta_xTB-ref-BoE",
        "residual_target",
    ),
    TargetSpec(
        "delta_bob_ref_boe",
        "Delta^BoB_ref,BoE",
        "Delta^BoB",
        DATASET_DIR / "tmQM__Delta_BoB-ref-BoE.csv",
        "E_resid_after_BoE_BoB_tmQM",
        "residual_target",
    ),
    TargetSpec(
        "delta_pre_ref_boe",
        "Delta^PRE_ref,BoE",
        "Delta^PRE",
        DATASET_DIR / "tmQM__Delta_PRE-ref-BoE.csv",
        "E_resid_after_BoE_PRE_tmQM",
        "residual_target",
    ),
    TargetSpec(
        "bob_ref_boe",
        "BoB_ref,BoE",
        "BoB",
        DATASET_DIR / "tmQM__BoB-ref-BoE.csv",
        "BoB_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "pre_ref_boe",
        "PRE_ref,BoE",
        "PRE",
        DATASET_DIR / "tmQM__PRE-ref-BoE.csv",
        "PRE_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "pre_plus_a_ref_boe",
        "PRE+A_ref,BoE",
        "PRE+A",
        DATASET_DIR / "tmQM__PRE_plus_A-ref-BoE.csv",
        "PRE_plus_A_ref-BoE",
        "fitted_baseline_target",
    ),
    TargetSpec(
        "delta_pre_plus_a_ref_boe",
        "Delta^PRE+A_ref,BoE",
        "Delta^PRE+A",
        DATASET_DIR / "tmQM__Delta_PRE_plus_A-ref-BoE.csv",
        "Delta_PRE_plus_A-ref-BoE",
        "residual_target",
    ),
    TargetSpec(
        "delta_mlp_bob_ref_boe",
        "Delta^MLP_BoB_ref,BoE",
        "Delta^MLP-BoB",
        MLP_BASELINE_DIR / "bob/tmQM__Delta_MLP_BoB-ref-BoE.csv",
        "Delta_MLP_BoB-ref-BoE",
        "mlp_residual_target",
        optional=True,
    ),
    TargetSpec(
        "delta_mlp_pre_ref_boe",
        "Delta^MLP_PRE_ref,BoE",
        "Delta^MLP-PRE",
        MLP_BASELINE_DIR / "pre/tmQM__Delta_MLP_PRE-ref-BoE.csv",
        "Delta_MLP_PRE-ref-BoE",
        "mlp_residual_target",
        optional=True,
    ),
    TargetSpec(
        "delta_mlp_pre_plus_a_ref_boe",
        "Delta^MLP_PRE+A_ref,BoE",
        "Delta^MLP-PRE+A",
        MLP_BASELINE_DIR / "pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv",
        "Delta_MLP_PRE_plus_A-ref-BoE",
        "mlp_residual_target",
        optional=True,
    ),
]

TARGET_BY_ID = {target.target_id: target for target in TARGETS}


def parse_target_ids(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return [target.target_id for target in TARGETS]
    if normalized in {"core", "ref_boe"}:
        return [
            "dft_ref_boe",
            "xtb_ref_boe",
            "delta_xtb_ref_boe",
            "delta_bob_ref_boe",
            "delta_pre_ref_boe",
            "bob_ref_boe",
            "pre_ref_boe",
            "pre_plus_a_ref_boe",
            "delta_pre_plus_a_ref_boe",
        ]
    if normalized == "deltas":
        return [
            "delta_xtb_ref_boe",
            "delta_bob_ref_boe",
            "delta_pre_ref_boe",
            "delta_pre_plus_a_ref_boe",
        ]
    if normalized == "mlp_deltas":
        return [
            "delta_mlp_bob_ref_boe",
            "delta_mlp_pre_ref_boe",
            "delta_mlp_pre_plus_a_ref_boe",
        ]
    target_ids = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(target_ids) - set(TARGET_BY_ID))
    if unknown:
        raise ValueError(f"Unknown target IDs: {unknown}. Choices: {sorted(TARGET_BY_ID)}")
    return target_ids


def selected_targets(value: str, *, skip_missing_optional: bool = True) -> list[TargetSpec]:
    out: list[TargetSpec] = []
    missing: list[str] = []
    for target_id in parse_target_ids(value):
        target = TARGET_BY_ID[target_id]
        if not target.csv_path.exists():
            if target.optional and skip_missing_optional:
                continue
            missing.append(f"{target.target_id}: {target.csv_path}")
            continue
        out.append(target)
    if missing:
        raise FileNotFoundError("Missing target CSV(s):\n" + "\n".join(missing))
    if not out:
        raise ValueError(f"No targets selected from --targets={value!r}")
    return out


def load_target_values(target: TargetSpec) -> pd.DataFrame:
    df = pd.read_csv(target.csv_path)
    missing = sorted({"mol_id", target.column} - set(df.columns))
    if missing:
        raise ValueError(f"{target.csv_path} is missing required columns: {missing}")
    out = df[["mol_id", target.column]].dropna().drop_duplicates("mol_id", keep="first").copy()
    out["mol_id"] = out["mol_id"].astype(str)
    out = out.rename(columns={target.column: target.target_id})
    return out.reset_index(drop=True)

