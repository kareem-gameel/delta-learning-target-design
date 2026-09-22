# Release Verification Contract

The corrected release must satisfy these reproducible checks:

- Complete 82-row tmQM and QM9 ledgers; correct target and seed membership.
- Test IQR, test roughness, and test MAE; no active validation-era SI snapshots.
- DFTB supporting controls excluded from pooled statistics.
- Canonical finalized manuscript and SI with matching current figure inputs.
- Coauthor approval and selected CC BY-NC-ND 4.0 manuscript license recorded.
- MIT limited to software; no third-party data license inferred.
- Unit tests, figure regeneration, example workflow, and release-content audit.
- Repository SHA256 manifest and clean source-ZIP LaTeX builds.
- Normal Git update preserving prior history; no force push.
- Public tag `v0.2.0-corrected` and matching PDF/source release assets.

`RELEASE_AUDIT.json` records machine-checked content results. GitHub supplies
the immutable tag commit and asset digests. No arXiv identifier or Zenodo DOI
is claimed. ArXiv upload and its server preview are separate from this release.
