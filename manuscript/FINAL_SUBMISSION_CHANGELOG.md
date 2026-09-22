# Final Submission Consistency Review

September 22, 2026. Supersedes the earlier section-limited review status.

- All main results retain the corrected test metrics and test-IQR roughness.
  QM9 errors remain absolute Ha, as explicitly requested.
- Abstract, Introduction, Discussion, Conclusion, and the final Section 2.5
  interpretation distinguish xTB's roughness reduction from DFTB's small
  roughness increase. Neither roughness reduction nor baseline accuracy is
  described as a necessary or sufficient condition for lower absolute error.
- Association is not described as proof of a causal physical mechanism.
  Shared-normalization contributions and the retrospective nature of the
  test-set diagnostic are stated consistently with SI Note 7.
- Removed the unsupported exclusion of optimization/capacity effects and
  the assertion that learned representations themselves were improved.
- Fixed element-wise atomic references are distinguished from train-only
  molecular-label-fitted composition references; this avoids describing
  standard-molecule reference conventions as direct isolated-atom calculations.
- Author approval is confirmed. The AI-assisted writing disclosure is absent.
- Main and SI availability sections cite corrected release v0.2.0-corrected;
  no Zenodo DOI is invented. Excluded large artifacts are stated explicitly.
- arXiv license selected by Kareem: CC BY-NC-ND 4.0. Software remains MIT.

Before-edit prose is preserved under
`revision_audit/final_submission_2026-09-22/before/`. Historical sources and
audit snapshots are excluded from the submission archive and active public
release, not deleted from the research workspace or rewritten in Git history.
