#!/usr/bin/env python3
"""Create a zip archive from reproducibility/artifact_manifest.csv."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "artifact_manifest.csv"
OUT = ROOT / "reproducibility" / "deltaml_paper_release_artifacts.zip"


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST}. Run reproducibility/build_artifact_manifest.py first."
        )

    rows = list(csv.DictReader(MANIFEST.open()))
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(MANIFEST, MANIFEST.relative_to(ROOT).as_posix())
        json_manifest = MANIFEST.with_suffix(".json")
        if json_manifest.exists():
            zf.write(json_manifest, json_manifest.relative_to(ROOT).as_posix())
        for row in rows:
            path = ROOT / row["relative_path"]
            if not path.exists():
                raise FileNotFoundError(path)
            zf.write(path, row["relative_path"])

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"WROTE: {OUT} size={size_mb:.2f} MiB files={len(rows)}")


if __name__ == "__main__":
    main()

