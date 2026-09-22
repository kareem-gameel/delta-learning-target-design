#!/usr/bin/env python3
"""Write deterministic SHA256 manifests for the release candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "outputs",
    "historical_inherited_si",
}
EXCLUDED_NAMES = {
    "MANIFEST.sha256",
    "repository_manifest.csv",
    "repository_manifest.json",
    "SOURCE_DATA_MANIFEST.csv",
    "SOURCE_DATA_MANIFEST.json",
    "RELEASE_AUDIT.json",
    "verification.json",
    "toy_roughness.csv",
    "toy_roughness.metadata.json",
    "toy_screening.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path.name not in EXCLUDED_NAMES
        and path.name != ".DS_Store"
    )


def rows_for(paths: list[Path]) -> list[dict[str, str | int]]:
    return [
        {
            "file": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())
    ]


def write_csv_json(stem: str, rows: list[dict[str, str | int]]) -> None:
    with (ROOT / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    source_roots = [ROOT / "manuscript/data", ROOT / "manuscript/si/data"]
    source_paths = [path for source_root in source_roots for path in source_root.rglob("*") if included(path)]
    write_csv_json("SOURCE_DATA_MANIFEST", rows_for(source_paths))

    repository_paths = [path for path in ROOT.rglob("*") if included(path)]
    repository_rows = rows_for(repository_paths)
    write_csv_json("repository_manifest", repository_rows)
    (ROOT / "MANIFEST.sha256").write_text(
        "".join(f"{row['sha256']}  {row['file']}\n" for row in repository_rows)
    )
    print(f"WROTE: {ROOT / 'MANIFEST.sha256'} files={len(repository_rows)}")


if __name__ == "__main__":
    main()
