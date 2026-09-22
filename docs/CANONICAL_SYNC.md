# Canonical manuscript handoff

The canonical working manuscript is maintained outside this release.
The `MANUSCRIPT READY` handoff was received and applied on September 22, 2026.
The following allowlist documents that completed transfer and future updates.

## Allowed current scientific inputs

The corrected public SI may use only these active data trees:

- `manuscript/si/data/test_protocol_2026-09-22/`
- `manuscript/si/data/normalization_controls_test_2026-09-22/`

The first tree contains the complete 82-row tmQM and 82-row QM9 ledgers, 164
prediction-score checks, four cached test graphs, all 32 roughness rows, and
the primary and DFTB matched-cohort statistics. The second tree contains the
normalization sensitivity controls.

## Explicit manuscript allowlist

After the handoff, copy the canonical root TeX sources, bibliography, section
files, current main figures and source tables, current figure generators and
auditors, SI TeX sources, SI scripts, the two data trees above, the
`test_protocol_2026-09-22` SI figures and tables, and the freshly built main
and SI PDFs.

Do not copy revision audits, submission staging directories, `node_modules`,
LaTeX temporary files, caches, or historical root-level SI figures and tables.
Do not copy any directory named `corrected_2026-09-22`; that name belongs to
the superseded validation-graph snapshot.

## Required checks after copying

1. Search the prospective public tree for validation-era scientific claims.
2. Run the independent SI numerical checker, including rescoring all 32 rows
   from the cached test graphs.
3. Rebuild the main manuscript and SI from source.
4. Run `make verify` and validate `MANIFEST.sha256`.
5. Confirm that DFTB remains seed 101 only and excluded from every pooled fit
   and correlation.
6. Compare the release candidate against a fresh clone of the existing public
   repository before committing.
