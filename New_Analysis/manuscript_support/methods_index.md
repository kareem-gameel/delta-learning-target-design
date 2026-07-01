# Methods Index

Curated method descriptions and implementation provenance for manuscript writing.

## Entry Template

### Method Name

- Manuscript section:
- Short description:
- Inputs:
- Outputs:
- Split definition:
- Model/baseline:
- Important hyperparameters:
- Source script/config:
- Saved output files:
- Notes/caveats:

## Terminology

- `DFT_ref`: reference-corrected DFT energy.
- `DFT_ref,BoE`: `DFT_ref` after removing fitted bond-order/composition baseline.
- `xTB_ref,BoE`: `xTB_ref` after the same kind of BoE correction.
- `Delta^xTB_ref,BoE`: `DFT_ref,BoE - xTB_ref,BoE`.
- `Delta^BoB_ref,BoE`: residual after fitting BoB on `DFT_ref,BoE`.
- `Delta^PRE_ref,BoE`: residual after fitting PRE on `DFT_ref,BoE`.
- `Delta^PRE+A_ref,BoE`: residual after fitting PRE+A on `DFT_ref,BoE`.
- MLP residuals: residuals after nonlinear MLP baselines fitted to `DFT_ref,BoE`.

