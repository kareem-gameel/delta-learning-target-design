# Reproducibility

## Level 1: Release verification

Requirements: the dependencies in `environment.yml`, including SciPy and
scikit-learn. The independent SI numerical check additionally requires a
local LaTeX build; see `manuscript/si/README_CORRECTED_SI.md`.

```bash
make verify
```

This validates the final tmQM and QM9 ledgers, seed contracts, summary
statistics, bundled checksums, repository contents, reusable package, and the
main figures that are currently reproducible from frozen manuscript data.

## Level 2: Roughness on supplied descriptors

Install the package and run `target-design roughness` on a descriptor matrix
and target table aligned by `mol_id`. This reproduces the central diagnostic
without training a GNN.

The paper's primary test graph uses:

```text
feature scaling = population SD, no centering
distance = cosine
k = 10
graph = undirected union of directed neighbours
edge weighting = locally scaled Gaussian
normalization = squared target IQR
```

## Level 3: Descriptor construction

Run `scripts/build_descriptors.py` on tmQM structures obtained from the original
provider. The script writes a sparse matrix, row IDs, and descriptor metadata.
The released split files then select the train, validation, or test molecules.
The reported graph is built only after training and checkpoint selection are
frozen, using the fixed test cohort.

## Level 4: Baselines and GNN training

Full GemNet and SchNet reproduction requires GPU training, FairChem-compatible
LMDB generation, baseline target construction, and the archival model
manifests. These larger artifacts are intentionally separated from the
lightweight Git repository. Their provenance is recorded here; they are not
claimed to be deposited in a separate public archival release.

The corrected targets use fixed tmQM train, validation, and test splits. Every
learned baseline is fitted on training molecules only. Each GNN target is
standardized by its training-set mean and standard deviation. The primary tmQM
test and QM9 transfer results use independent seeds 101, 102, and 103, and
uncertainty is the sample standard deviation of seed-level metrics. DFTB is a
one-seed supporting control and is excluded from pooled correlations and fitted
lines. Validation metrics select checkpoints; they are not the reported
performance or roughness values.

## Verification commands

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_main_figures.py
python scripts/verify_release.py
make example
```
