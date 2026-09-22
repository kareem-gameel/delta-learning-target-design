# Data provenance

## Public source datasets

- **tmQM** supplies the transition-metal molecular structures and primary
  in-domain chemical space.
- **QM9** supplies the organic-molecule out-of-domain transfer set.

The raw datasets are not redistributed. Obtain them from their original
providers and use the citations in the manuscript.

## Split identifiers

The `splits/` directory contains the fixed tmQM molecule IDs used throughout
the paper:

| Split | Molecules |
|---|---:|
| train | 56,830 |
| validation | 7,104 |
| test | 7,104 |

Every merge in the released code is performed by molecule ID. CSV, sparse
matrix, LMDB, or extxyz row order must never be assumed to match.

## Source data

`manuscript/data/` contains the machine-readable main-figure values. The active
SI evidence is restricted to
`manuscript/si/data/test_protocol_2026-09-22/` and
`manuscript/si/data/normalization_controls_test_2026-09-22/`. These include the
exact values used in the main and supplementary figures, completed in-domain
multiseed summaries, cached test graphs, normalization controls, and DFTB
controls. SHA256 manifests bind the copied files to the audited evidence.

## Protocol distinction

The reported graph-roughness panel is computed on newly constructed fixed tmQM
test graphs. The primary graphs contain the same 7,104 test molecules for all
13 targets; the matched DFTB control graphs contain the 7,068 DFTB-valid test
molecules. Every learned BoE, BoB, PRE, and PRE+A baseline in the corrected
target registry is fitted only on the tmQM training split. Its frozen
parameters are then applied to validation and test molecules. Validation is
used for checkpoint selection and learning-rate scheduling only. GNN
performance, target IQR, and roughness are evaluated on the untouched tmQM test
split using best-validation checkpoints.

The primary panel uses GemNet and SchNet seeds 101, 102, and 103. QM9 transfer
is also evaluated for all three primary seeds. DFTB uses seed 101 only and is
reported as supporting evidence, not included in pooled correlations or fitted
lines. Historical full-dataset-fit residual targets are not part of this
corrected release.

The repository presents `D_IQR` as an architecture-informed screening
diagnostic and the relationship as an empirical association for the studied
target panel, not as a universal error law. Because the strongest normalized
comparison shares IQR normalization across both axes, raw-error,
raw-roughness, and decoupled-normalization controls are included.

## Descriptor metadata

The machine-readable metadata copied from the canonical PRE and angular caches
are provided in `configs/pre_feature_metadata.json` and
`configs/angular_feature_metadata.json`. These files take precedence over
informal shorthand descriptions of radial-grid spacing.
