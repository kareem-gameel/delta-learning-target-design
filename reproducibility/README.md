# Reproducibility Release Notes

This directory contains metadata for preparing a publication-ready release.

## Files

- `build_artifact_manifest.py`: scans curated release paths and writes checksums.
- `artifact_manifest.csv`: generated manifest with file size and SHA256 checksums.
- `artifact_manifest.json`: JSON equivalent of the generated manifest.

## Usage

From the repository root:

```bash
python reproducibility/build_artifact_manifest.py
```

The manifest intentionally skips large binary artifacts such as LMDB databases,
checkpoints, `.npz` prediction arrays, parquet embedding exports, and raw
structure files. Those should be handled explicitly in the Zenodo bundle only if
the final paper requires exact reruns without regeneration.
