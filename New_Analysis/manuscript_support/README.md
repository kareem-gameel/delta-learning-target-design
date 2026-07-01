# Manuscript Support Registry

This folder is the source-of-truth layer for manuscript writing. It should contain curated, manuscript-ready links to saved results, methods notes, figures, and tables.

Workflow rule: do not recompute analyses from the manuscript-writing workflow. If a result, table, or figure is missing or ambiguous, record what is needed and ask the analysis workflow to generate the artifact.

## Contents

- `results_index.md`: numerical results and source files.
- `methods_index.md`: concise method descriptions and implementation provenance.
- `figure_index.md`: publication-ready figures, captions, and source files.
- `key_tables/`: manuscript-ready CSV or Markdown tables.
- `figure_exports/`: publication-ready figure exports.
- `methods_notes/`: longer technical notes.

## Priority Order For Manuscript Claims

1. Use artifacts in this folder.
2. If absent, search current `New_Analysis/` outputs.
3. If still absent or ambiguous, request a saved artifact from the analysis workflow.

