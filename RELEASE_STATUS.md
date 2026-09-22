# Corrected Release

Version: `v0.2.0-corrected`, September 22, 2026.

This release supersedes the July snapshot in the active working tree while
preserving Git history. It includes the finalized corrected manuscript, SI,
test-protocol inputs, and reproducibility tools. No DOI is claimed.

There are 13 primary targets with three seeds for both architectures, and two
single-seed DFTB controls per architecture: 82 tmQM test evaluations and 82 QM9
evaluations. IQR normalization and graph roughness use test cohorts. Validation
selects checkpoints and controls scheduling. DFTB remains excluded from pooled
fits and correlations.

The canonical manuscript-ready handoff was received and applied. Main and SI
compile from the isolated submission ZIP without warnings. The numerical audit
checks 164 stored score records, four test graphs, and 32 roughness values;
normalization controls check 10,000 permutation correlations.

Software is MIT-licensed. The manuscript/preprint uses CC BY-NC-ND 4.0.
Publication data are not assigned a separate reuse license. See `LICENSES.md`.

The release is not an arXiv submission. No identifier or server acceptance is
invented; submission metadata and the source ZIP are included for the authors.
