# Reproduction Guide

This guide describes the clean reproduction path for the manuscript analyses.
It separates lightweight Git-tracked code from larger generated artifacts that
belong in a versioned Zenodo archive.

## 1. Environment

For local plotting and table generation, use the local analysis wrapper:

```bash
New_Analysis/local_analysis_python.sh <script.py> [args...]
```

On Trillium, activate the FairChem environment:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"
```

The environment specification in `environment.yml` is a lightweight record of
the Python-side dependencies needed for the analysis scripts. Exact FairChem,
PyTorch, CUDA, and cluster module versions should be recorded from the cluster
environment used for final model training.

## 2. Source Data

Raw source data are not redistributed through Git.

- tmQM structures and reference quantities should be obtained from the tmQM
  source.
- QM9 structures and reference quantities should be obtained from the QM9
  source.
- Large derived LMDB files should be regenerated from scripts or restored from
  the Zenodo artifact bundle if included in a later archive version.

## 3. Processed Targets

Processed target tables are in `New_Analysis/datasets/`. Important targets
include:

- `tmQM__DFT_ref.csv`
- `tmQM__DFT_ref-BoE.csv`
- `tmQM__xTB_ref-BoE.csv`
- `tmQM__Delta_xTB-ref-BoE.csv`
- `tmQM__Delta_BoB-ref-BoE.csv`
- `tmQM__Delta_PRE-ref-BoE.csv`
- `tmQM__Delta_PRE_plus_A-ref-BoE.csv`

The target registry is in:

```text
New_Analysis/expanded_target_panel/target_registry.csv
```

## 4. Baseline Models

Linear ridge and MLP baseline workflows are organized under:

```text
New_Analysis/expanded_target_panel/
New_Analysis/pre_plus_a/
New_Analysis/mlp_baselines/
```

PRE+A is a pair-radial PRE baseline augmented with central-element angular
environment histograms. Its documentation is in:

```text
New_Analysis/pre_plus_a/README.md
```

## 5. GemNet and SchNet Training

GemNet run bundle generators are under:

```text
Target_Design/gemnet_runs/
```

SchNet run bundle generators are under:

```text
New_Analysis/schnet_runs/trillium/
```

Training itself is cluster-side and requires GPU resources. Do not expect the
full GemNet and SchNet training workflow to run locally.

## 6. Smoothness and Roughness Analyses

Feature-space smoothness analyses are under:

```text
New_Analysis/expanded_target_panel/
New_Analysis/gemnet_learned_embedding_smoothness/
New_Analysis/manuscript_support/
```

The roughness metric used in the manuscript is based on k-nearest-neighbor
graphs in a molecular feature space. For an undirected edge set `E`, the raw
edge roughness is:

```text
D(y) = (1 / |E|) sum_(i,j in E) (y_i - y_j)^2
```

Normalized forms divide this quantity by a target-scale estimate such as
population variance or IQR squared.

## 7. Manuscript Figures and Tables

Paper figures are stored in:

```text
manuscript/figures/
```

Paper tables are stored in:

```text
manuscript/tables/
```

The source scripts and intermediate table inputs are under:

```text
New_Analysis/manuscript_support/
```

## 8. Artifact Manifest

Generate the release artifact manifest with:

```bash
python reproducibility/build_artifact_manifest.py
```

This writes:

```text
reproducibility/artifact_manifest.csv
reproducibility/artifact_manifest.json
```

These files record relative paths, sizes, modification times, and SHA256
checksums for the curated release artifacts.

## 9. Zenodo Artifact Bundle

For the first public release, create a Zenodo artifact bundle containing:

- processed target CSVs
- split ID CSVs
- manuscript figure/table source data
- final model metric summaries
- target smoothness/roughness summary tables
- training/evaluation manifests
- optional large checkpoints and prediction arrays needed for exact
  regeneration of selected figures

Do not upload raw tmQM or QM9 data unless their source licenses explicitly
permit redistribution. Prefer citing the original providers.

After Zenodo mints the DOI, replace the manuscript placeholder with the Zenodo
DOI and GitHub repository URL.
