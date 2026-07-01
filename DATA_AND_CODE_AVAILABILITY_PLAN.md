# Data and Code Availability Plan

This file records what should be released with the manuscript and what should
not be pushed directly to GitHub. It assumes GitHub for code and Zenodo for the
versioned data/artifact DOI.

## Recommended Availability Statement

Use this in the manuscript until permanent links are minted:

> The source datasets used in this work are publicly available from their
> original providers: tmQM for transition-metal complexes and QM9 for small
> organic molecules. The processed split identifiers, target-generation tables,
> baseline fits, model-configuration files, evaluation manifests, and
> figure/table source data needed to reproduce the analyses in this manuscript
> will be deposited in a versioned Zenodo archive at publication. The analysis
> and training code will be released in a public GitHub repository, with the
> corresponding Zenodo DOI cited in the final manuscript. Large
> generated artifacts, including LMDB training databases, neural-network
> checkpoints, prediction arrays, and learned embedding exports, are not tracked
> directly in Git; when required for exact reproduction, they will be provided
> through the Zenodo archive or can be regenerated from the released scripts and
> the public source datasets.

Replace the provisional wording after release with:

> Code and data artifacts are available at `https://github.com/<user>/<repo>`
> and archived at `https://doi.org/<zenodo-doi>`.

## What Goes to GitHub

GitHub should contain the clean, source-first reproducibility repo:

- Manuscript source:
  - `manuscript/`
- Analysis scripts:
  - `New_Analysis/`
  - `Target_Design/` scripts that are still referenced by the manuscript
  - `scripts/`
- Trillium/cluster run bundles and configuration generators:
  - `New_Analysis/gemnet_*`
  - `New_Analysis/schnet_*`
  - `New_Analysis/mlp_*`
  - `trillium_project/configs/` if still needed for historical runs
- Lightweight documentation:
  - `README.md`
  - `MODEL_METHODOLOGY_REPORT.md`
  - this file
  - methods notes under `New_Analysis/manuscript_support/methods_notes/`
- Small, human-readable result tables used directly by the manuscript, if not
  too large:
  - `New_Analysis/manuscript_support/tables/*.csv`
  - selected `*.md` summaries

Do not commit the whole current working tree without cleaning it. The current
repo contains many large local artifacts and many tracked deletions from older
work. Create a clean release branch or a new clean repo.

## What Goes to Zenodo

Use Zenodo for the frozen data artifact because it provides a DOI and a stable
citation target for the publication.

Minimum archive contents needed to reproduce manuscript figures/tables:

- Split identifiers:
  - `trillium_project/split_ids/tmqm_train_ids.csv`
  - `trillium_project/split_ids/tmqm_val_ids.csv`
  - `trillium_project/split_ids/tmqm_test_ids.csv`
- Processed target CSVs:
  - `New_Analysis/datasets/tmQM__DFT_ref-BoE.csv`
  - `New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv`
  - `New_Analysis/datasets/tmQM__BoB-ref-BoE.csv`
  - `New_Analysis/datasets/tmQM__PRE-ref-BoE.csv`
  - `New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv`
  - `New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv`
- Source-artifact CSVs needed for QM9 reconstruction:
  - `New_Analysis/source_artifacts/boe/`
  - `New_Analysis/source_artifacts/boe_xtb/`
  - `New_Analysis/source_artifacts/bob/`
  - `New_Analysis/source_artifacts/pre/`
- Final manuscript tables:
  - `New_Analysis/manuscript_support/tables/`
- Final manuscript figures:
  - `New_Analysis/manuscript_support/figures/`
- Model/evaluation manifests:
  - GemNet and SchNet target manifests
  - QM9 prediction/evaluation manifests
- Optional but useful:
  - Trained checkpoints for final GemNet and SchNet runs
  - Final prediction arrays/CSVs used for QM9 and tmQM parity plots
  - Learned embedding roughness CSVs, not raw parquet exports unless needed

Avoid depositing raw third-party datasets if their own repositories are already
public and citation is sufficient. Instead, provide scripts and checksums that
map from the public source data to the processed artifacts.

## What Not to Put in Git

Keep these out of Git:

- `*.lmdb`
- `*.parquet`
- `*.npz`
- `*.npy`
- `*.pt`, `*.pth`, `*.ckpt`
- raw `*.extxyz`, `*.sdf`, `*.mol2`
- large prediction CSVs
- tensorboard logs
- generated PDFs and PNGs, except final selected manuscript figures if desired

The existing `.gitignore` already excludes most of these.

## Immediate Action Checklist

1. Clean the repo for release:
   - create a new branch or new repo from a clean checkout;
   - copy only source code, manuscript files, configs, and small tables;
   - do not carry over old deleted-file state or local generated artifacts.
2. Create a `reproducibility/` or `release_artifacts/` folder containing:
   - `artifact_manifest.csv`;
   - `environment.yml` or `requirements.txt`;
   - `RUN_REPRODUCTION.md`;
   - checksums for all uploaded data artifacts.
3. Push the clean code repo to GitHub.
4. Upload the data artifact to a Zenodo draft.
5. Mint the Zenodo DOI when the artifact is final.
6. Replace the placeholder URLs/DOIs in `manuscript/main.tex`.

## Suggested Public Repo Name

`target-design-delta-mlips`

Alternative:

`delta-learning-target-design`

## Suggested Zenodo Artifact Name

`Data and analysis artifacts for: Delta Learning as Target Design for Transferable Machine-Learning Interatomic Potentials`
