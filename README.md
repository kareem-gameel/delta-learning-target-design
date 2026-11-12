# Delta Target Design

Code, processed evidence, manuscript source, and reproducibility checks for
**The Mechanics of Delta Learning: Target Design for Generalizable Scientific
Machine Learning**.

Manuscript revision **v2** is available in release `v0.3.0`. Read the [combined manuscript and SI](manuscript/build/main.pdf) or the [standalone SI](manuscript/supplement.pdf). Supplementary Table S3 compares final-epoch training MAE with validation and test MAE at the validation-selected checkpoint. The September [v0.2.0-corrected release](https://github.com/kareem-gameel/delta-learning-target-design/releases/tag/v0.2.0-corrected) preserves the original manuscript and corrected test protocol. The files under `manuscript/submission/` document that September submission.

## Evidence contract

- The primary panel contains 13 targets, GemNet and SchNet, and independent
  training seeds 101, 102, and 103: 78 trained models.
- The DFTB supporting panel contains two targets, both architectures, and seed
  101 only: four trained models.
- In-domain results are evaluated on the untouched tmQM test split using each
  run's best-validation checkpoint.
- QM9 transfer is evaluated for all three primary seeds. DFTB remains a
  one-seed supporting control.
- Reported uncertainty is the sample standard deviation of seed-level metrics,
  not an ensemble-prediction error.
- Weighted Dirichlet roughness is computed on newly built fixed test graphs:
  7,104 common primary molecules, or 7,068 DFTB-valid molecules for the
  matched supporting control. Test labels are used only after every baseline,
  checkpoint, and training decision is frozen.
- Raw-error and decoupled-normalization controls are included because the
  shared-IQR normalization can strengthen the apparent association.

The frozen corrected evidence contains 82 final tmQM evaluations and 82 final
QM9 evaluations. Validation selects checkpoints; reported in-domain
performance, IQR normalization, and roughness use the test split. See
`manuscript/si/README_CORRECTED_SI.md` for the statistical contract and
`RELEASE_STATUS.md` for release state.

## Repository layout

```text
src/target_design/       Reusable roughness and descriptor implementation
scripts/                 Verification, descriptor, figure, and manifest tools
tests/                   Numerical and release-contract tests
configs/                 Machine-readable method and training metadata
examples/                Small CPU-only examples
manuscript/              Sep15 manuscript, corrected SI, frozen data, and PDFs
docs/                    Method, provenance, reproduction, and publishing notes
```

Raw tmQM and QM9 structures, LMDB databases, checkpoints, full prediction
arrays, embeddings, and training logs are intentionally excluded. The complete
49 GB scientific workspace is maintained separately on Trillium and is audited
by a standalone manifest.

## Installation

```bash
conda env create -f environment.yml
conda activate delta-target-design
python -m pip install -e .
```

## Verify the release

```bash
make verify
```

This runs the numerical tests, regenerates the reproducible main-figure assets
from bundled source data, executes the small descriptor example, audits the
release contents, and rebuilds SHA256 manifests.

## Compute weighted roughness

```bash
target-design roughness \
  --features path/to/pre_plus_a_features.npz \
  --feature-ids path/to/feature_ids.csv \
  --targets path/to/targets.csv \
  --target-columns parent,residual_a,residual_b \
  --k 10 \
  --output outputs/roughness.csv
```

Rows are aligned by molecule ID. The primary convention uses population-SD
feature scaling without centering, cosine neighbours, an undirected union
graph, locally scaled Gaussian edge weights, and squared-IQR target
normalization.

## Manuscript status

The corrected manuscript and SI are complete and included in `manuscript/`.
The combined PDF is `manuscript/build/main.pdf`, the standalone SI is
`manuscript/supplement.pdf`, and the historical v1 arXiv source archive is
`manuscript/submission/Sep15_arxiv_source.zip`. The manuscript license is
CC BY-NC-ND 4.0. This release does not itself submit the paper to arXiv.

## Citation and license

Citation metadata are in `CITATION.cff`. The MIT license applies only to the
software code. The manuscript/preprint uses the author-selected CC BY-NC-ND
4.0 license. No separate reuse license is granted for bundled publication
data. See `LICENSES.md` for the path-scoped terms.
