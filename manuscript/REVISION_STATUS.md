# Corrected September Manuscript

Reviewed September 22, 2026. The main manuscript and complete SI use corrected
train-only targets and final best-validation-checkpoint evaluations.

- In-domain normalized errors use test MAE divided by matching test IQR.
  Roughness uses matching test graphs and test IQR. Validation selects
  checkpoints and controls training; it is not the reported test score.
- All 13 primary targets have three seeds per architecture. DFTB is a
  single-seed, valid-subset control, displayed separately and excluded from
  pooled correlations and permutation controls.
- QM9 transfer uses reconstructed DFT-BoE energies and absolute Ha errors,
  with matched cohorts and explicitly oracle mean-shift diagnostics.
- All four main figures, eight SI figures, and 18 SI tables use the current
  protocol. Figure 3 intentionally has no error bars.
- All manuscript sections are reconciled with the corrected evidence.
  DFTB's small roughness increase remains visible; mechanistic interpretations
  are distinguished from established results.
- Coauthor approval and CC BY-NC-ND 4.0 manuscript licensing are confirmed.
  Software remains MIT-licensed. No data license is inferred from either.

See `FINAL_SUBMISSION_CHANGELOG.md`, `ARXIV_READINESS.md`, and
`ARXIV_METADATA.md` for changes, verification, and submission metadata.
Historical revision notes are preserved in the research-only audit directory
and are not current evidence or submission instructions.

Canonical inputs: `si/data/test_protocol_2026-09-22/` and
`si/data/normalization_controls_test_2026-09-22/`.
