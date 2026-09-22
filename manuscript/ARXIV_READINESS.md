# arXiv Submission Verification

Reviewed September 22, 2026.

## Confirmed

- Coauthor approval: confirmed by Kareem.
- Selected manuscript license: CC BY-NC-ND 4.0. Software remains MIT.
- AI-assisted writing disclosure removed as requested.
- Main PDF: 41 pages, including the complete 24-page SI.
- Final main and SI builds have no undefined citations, unresolved references,
  or layout warnings.
- Both documents also compile independently from the final ZIP in a fresh
  temporary directory. Rendered page overviews were checked for layout defects.
- Numerical audit passed: 82 runs, 164 stored prediction-score checks,
  four independently rescored test graphs, and 32 roughness rows.
- Normalization controls passed: 10,000 architecture-specific permutation
  correlations and all similar-IQR pairs. DFTB is excluded from pooled fits.
- Four main figures regenerated from corrected test-protocol data.
- Wording distinguishes descriptive evidence from mechanism and explicitly
  preserves DFTB's roughness exception.
- Availability and bibliography identify corrected release
  `v0.2.0-corrected`, not the July snapshot. No DOI is claimed.

## Submission Files

- Combined manuscript and SI: `build/main.pdf`.
- Standalone SI: `supplement.pdf`.
- Source archive: `submission/Sep15_arxiv_source.zip`.
- Archive member checksums: `submission/manifest.json`.
- Metadata and license: `ARXIV_METADATA.md`.
- Entry point: `main.tex`, pdfLaTeX; compiled bibliography included.

The archive excludes historical datasets, logs, checkpoints, and audit
backups. Research originals and Git history are preserved, not deleted.

These checks validate packaged evidence and the local build; they are not a
new inference run or an arXiv submission. The submitter must inspect arXiv's
server-generated preview and select the appropriate submission or replacement
record in its interface.

## Reproduction

From the research repository root, use `New_Analysis/local_analysis_python.sh`
with these scripts:

- `coauthor_share/Sep15_arxiv/si/scripts/check_corrected_si.py --numerical-only`
- `coauthor_share/Sep15_arxiv/si/scripts/check_test_normalization_controls.py`
- `coauthor_share/Sep15_arxiv/scripts/build_arxiv_candidate.py`

Official guidance: https://info.arxiv.org/help/submit_tex.html and
https://info.arxiv.org/help/00README.html.
