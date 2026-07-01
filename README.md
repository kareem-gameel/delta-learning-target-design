# Delta Learning as Target Design for Transferable ML Interatomic Potentials

This repository contains the manuscript source, analysis scripts, training
configuration generators, and lightweight result tables used for the paper:

> Delta Learning as Target Design for Transferable Machine-Learning Interatomic
> Potentials

The study compares target transformations for machine-learning interatomic
potentials on tmQM and QM9, focusing on corrected reference energies,
baseline-residual targets, and the relationship between target roughness and
learnability for GemNet and SchNet.

## Repository Scope

This Git repository is intended to contain reproducible code and lightweight
publication artifacts only. Large generated artifacts are excluded from Git:

- LMDB training databases
- neural-network checkpoints
- raw structure files
- exported embedding parquet files
- large prediction arrays
- local logs and scratch outputs

The frozen data artifact bundle for publication should be deposited on Zenodo
to provide a stable DOI. GitHub remains the code repository.

## Main Locations

- `manuscript/`: LaTeX manuscript source, bibliography, paper figures, and paper tables.
- `New_Analysis/`: target construction, baseline fitting, smoothness analysis, transfer analysis, and manuscript-support scripts.
- `Target_Design/`: GemNet training bundle generators and target-design experiments.
- `trillium_project/`: selected split IDs, historical run utilities, and Trillium-specific helper scripts.
- `reproducibility/`: release metadata, artifact manifest generation, and packaging notes.

## Reproduction

Start with [RUN_REPRODUCTION.md](RUN_REPRODUCTION.md). The artifact list and
checksums are generated with:

```bash
python reproducibility/build_artifact_manifest.py
```

The generated `reproducibility/artifact_manifest.csv` records file size and
SHA256 checksums for the curated lightweight and release-bound artifacts.

## Data Availability

The source datasets are public and should be cited from their original
providers: tmQM and QM9. Processed split identifiers, target tables, evaluation
manifests, figure source data, and selected trained-output summaries should be
archived in a versioned Zenodo record and cited by DOI in the final manuscript.
