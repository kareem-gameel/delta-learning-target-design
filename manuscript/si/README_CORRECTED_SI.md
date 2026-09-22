# Corrected Test-Protocol SI

Current sources are `corrected_content.tex` and `test_normalization_control.tex`.
Current data are in `data/test_protocol_2026-09-22/` and
`data/normalization_controls_test_2026-09-22/`. Older validation-graph snapshots
are not part of this release.

All normalized in-domain errors use test MAE/test IQR. Weighted roughness
uses fixed test graphs and squared test IQR. The 13 primary targets use three
seeds per architecture; DFTB is a single-seed, reduced-cohort control excluded
from pooled correlations. QM9 errors are absolute Ha.

From `manuscript/`, build with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=si/build si/supplementary_information.tex
python si/scripts/check_corrected_si.py --numerical-only
python si/scripts/check_test_normalization_controls.py
```

The independent numerical checker validates all 82 runs, 164 archived score
checks, and 32 roughness values from four saved graphs. This verifies bundled
evidence; it does not rerun model inference from excluded large checkpoints.
