# Zenodo Upload Checklist

This repository is now prepared for a Zenodo-first data/artifact release.

## Upload File

Upload this zip file to Zenodo:

```text
reproducibility/deltaml_paper_release_artifacts.zip
```

Current local size:

```text
124.6 MiB
```

The zip includes:

- `reproducibility/artifact_manifest.csv`
- `reproducibility/artifact_manifest.json`
- processed target tables
- split identifiers
- manuscript figures/tables
- manuscript-support result tables
- model/evaluation manifests
- reproduction documentation

The zip intentionally excludes:

- LMDB databases
- raw tmQM/QM9 structures
- parquet embedding exports
- NPZ/NPY prediction arrays
- neural-network checkpoints
- local logs and build outputs

## Suggested Zenodo Metadata

Title:

```text
Delta Learning as Target Design for Transferable Machine-Learning Interatomic Potentials
```

Upload type:

```text
Dataset
```

Description:

```text
Processed split identifiers, target-generation tables, baseline fits,
model-configuration files, evaluation manifests, figure/table source data, and
reproducibility metadata accompanying the manuscript "Delta Learning as Target
Design for Transferable Machine-Learning Interatomic Potentials". Raw tmQM and
QM9 source datasets are not redistributed and should be obtained from their
original providers.
```

Creator:

```text
Gameel, Kareem
University of Toronto
```

License:

```text
MIT
```

Keywords:

```text
delta learning; machine-learning interatomic potentials; GemNet; SchNet; tmQM;
QM9; target design
```

Related identifier:

```text
https://github.com/kareem-gameel/Delta_Learning_GemNet
```

Relationship:

```text
Is supplement to
```

## Local Preparation Commands

Regenerate the manifest and zip before the final upload:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
python3 reproducibility/build_artifact_manifest.py
python3 reproducibility/make_release_bundle.py
```

## After Zenodo DOI Is Minted

Replace the manuscript placeholders in:

```text
manuscript/main.tex
```

with:

```text
Code and data artifacts are available at https://github.com/kareem-gameel/Delta_Learning_GemNet and archived at https://doi.org/<zenodo-doi>.
```

