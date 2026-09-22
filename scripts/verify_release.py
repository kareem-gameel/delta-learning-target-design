#!/usr/bin/env python3
"""Audit the GitHub release candidate for scientific completeness and hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "manuscript/si/data/test_protocol_2026-09-22"
FORBIDDEN_SUFFIXES = {
    ".lmdb", ".pt", ".pth", ".ckpt", ".onnx", ".parquet", ".npz", ".npy",
    ".h5", ".hdf5", ".sdf", ".mol2", ".joblib", ".pkl", ".log", ".aux",
    ".fls", ".fdb_latexmk", ".synctex.gz",
}
MAX_FILE_BYTES = 50 * 1024 * 1024
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "outputs",
    "historical_inherited_si",
}
ALLOWED_NUMPY_ARCHIVES = {
    "manuscript/si/data/test_protocol_2026-09-22/graph_gemnet_dftb_valid.npz",
    "manuscript/si/data/test_protocol_2026-09-22/graph_gemnet_primary.npz",
    "manuscript/si/data/test_protocol_2026-09-22/graph_schnet_dftb_valid.npz",
    "manuscript/si/data/test_protocol_2026-09-22/graph_schnet_primary.npz",
    "manuscript/si/data/normalization_controls_test_2026-09-22/permuted_roughness.npz",
}
STALE_ACTIVE_PATHS = [
    ROOT / "manuscript/si/data/corrected_2026-09-22",
    ROOT / "manuscript/si/figures/corrected_2026-09-22",
    ROOT / "manuscript/si/tables/corrected_2026-09-22",
]


def forbidden(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative.as_posix() in ALLOWED_NUMPY_ARCHIVES:
        return False
    return any(relative.as_posix().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return path.is_file() and not any(part in EXCLUDED_PARTS for part in relative.parts)


def main() -> None:
    symlinks = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_symlink()]
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if release_file(path) and forbidden(path)
    ]
    oversized = [
        {"file": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size}
        for path in ROOT.rglob("*")
        if release_file(path) and path.stat().st_size > MAX_FILE_BYTES
    ]

    controls = {"DFTB_ref-BoE", "Delta_DFTB-ref-BoE"}
    errors: list[str] = []
    main_targets: set[str] = set()
    tmqm = pd.DataFrame()
    qm9 = pd.DataFrame()
    score_audit = pd.DataFrame()
    roughness = pd.DataFrame()
    correlations = pd.DataFrame()
    required_data = {
        "registry": "target_registry.csv",
        "tmqm": "tmqm_final_ledger.csv",
        "qm9": "qm9_final_ledger.csv",
        "scores": "prediction_score_audit.csv",
        "roughness": "roughness.csv",
        "correlations": "correlations.csv",
    }
    missing_data = [name for name in required_data.values() if not (DATA / name).is_file()]
    if missing_data:
        errors.append(f"Current test-protocol files are missing: {missing_data}")
    else:
        registry = pd.read_csv(DATA / required_data["registry"])
        tmqm = pd.read_csv(DATA / required_data["tmqm"])
        qm9 = pd.read_csv(DATA / required_data["qm9"])
        score_audit = pd.read_csv(DATA / required_data["scores"])
        roughness = pd.read_csv(DATA / required_data["roughness"])
        correlations = pd.read_csv(DATA / required_data["correlations"])
        main_targets = set(registry["target_id"])
        if len(main_targets) != 13 or main_targets & controls:
            errors.append("The corrected primary target registry is invalid")
        if len(tmqm) != 82 or not tmqm["test_status"].eq("complete").all():
            errors.append("The final tmQM ledger is incomplete")
        if len(qm9) != 82 or not qm9["qm9_score_status"].eq("complete").all():
            errors.append("The final QM9 ledger is incomplete")
        if not qm9["checkpoint_finality"].eq("final").all():
            errors.append("At least one QM9 checkpoint is provisional")
        if len(score_audit) != 164 or score_audit["absolute_difference"].max() >= 1e-12:
            errors.append("Prediction-score reproduction failed")
        primary_roughness = roughness.loc[roughness["cohort"].eq("primary")]
        dftb_roughness = roughness.loc[roughness["cohort"].eq("dftb_valid")]
        if (
            len(roughness) != 32
            or not roughness["split"].eq("test").all()
            or len(primary_roughness) != 26
            or not primary_roughness["n_molecules"].eq(7104).all()
            or len(dftb_roughness) != 6
            or not dftb_roughness["n_molecules"].eq(7068).all()
        ):
            errors.append("The test-graph roughness contract is invalid")
        if correlations["dftb_included"].astype(bool).any():
            errors.append("DFTB entered pooled correlation or fit statistics")
    stale_active = [path.relative_to(ROOT).as_posix() for path in STALE_ACTIVE_PATHS if path.exists()]
    if stale_active:
        errors.append(f"Validation-era SI snapshots remain active: {stale_active}")
    if symlinks:
        errors.append(f"Symlinks are not allowed in the release candidate: {symlinks}")
    if forbidden_files:
        errors.append(f"Forbidden generated artifacts are present: {forbidden_files}")
    if oversized:
        errors.append(f"Files larger than 50 MiB are present: {oversized}")
    if not (ROOT / "LICENSES.md").is_file():
        errors.append("Path-scoped licensing documentation is missing")

    report = {
        "release_status": "verified" if not errors else "failed",
        "public_release_ready": not errors,
        "public_release_blockers": errors,
        "primary_targets": len(main_targets),
        "tmqm_final_rows": len(tmqm),
        "qm9_final_rows": len(qm9),
        "prediction_score_checks": len(score_audit),
        "dftb_seeds": [101],
        "primary_seeds": [101, 102, 103],
        "roughness_split": "test",
        "roughness_rows": len(roughness),
        "dftb_excluded_from_pooled_statistics": not correlations.empty and not correlations["dftb_included"].astype(bool).any(),
        "stale_active_paths": stale_active,
        "symlinks": symlinks,
        "forbidden_files": forbidden_files,
        "oversized_files": oversized,
        "errors": errors,
    }
    (ROOT / "RELEASE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
