# Model Methodology Report

This document consolidates the training, fitting, target definitions, split protocol, and error-calculation procedures used across the baseline and GemNet models in this project. It is intended as a writing aid for the paper methodology section.

## 1. High-level modeling structure

The project compares three broad model classes:

1. Physical baseline models.
   These are not fit on tmQM in the same way as the statistical baselines. The main physical baseline used in the paper is xTB.

2. Statistical baseline models.
   These are linear ridge-regression models fit on tmQM using handcrafted features. The main families are `BoE`, `BoB`, and `PRE`, together with composite/stacked variants such as `BoB+BoE` and `PRE+BoE`.

3. GemNet neural models.
   These are GemNet-T graph neural networks trained either directly on the total corrected energy target or on residual targets defined relative to a baseline model.

The main conceptual comparison is between:
- direct learning of the total corrected energy, and
- residual learning, where a simpler baseline is first evaluated and a learned model predicts the remaining correction.

## 2. Common energy target used throughout

Unless stated otherwise, the project uses the tmQM `reference_corrected_energy` as the reference supervised energy target. This is the DFT-level corrected total energy used consistently across the baseline fitting workflows, the residual CSV generation, the LMDB construction, and the GemNet training/evaluation pipeline.

Residual targets are defined as:

- `target_residual = reference_corrected_energy - baseline_prediction`

The exact meaning of the baseline prediction depends on the baseline family:
- for xTB residual learning: `baseline_prediction = E_xTB`
- for BoE residual learning: `baseline_prediction = E_BoE_tmQM`
- for BoB+BoE residual learning: `baseline_prediction = E_BoE_tmQM`, while the linear second-stage model uses BoB features to model the BoE residual
- for PRE+BoE residual learning: `baseline_prediction = E_BoE_tmQM`, while PRE features model the BoE residual

## 3. Dataset split protocol used for fair comparison

### 3.1 GemNet split

The tmQM train/validation/test split used for the Trillium GemNet experiments was exported directly from the LMDB files used for GemNet training. The split membership files are:

- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/split_ids/tmqm_train_ids.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/split_ids/tmqm_val_ids.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/split_ids/tmqm_test_ids.csv`

The split sizes are:
- train: 56,830 molecules
- validation: 7,104 molecules
- test: 7,104 molecules

This is the canonical split used for the split-aware baseline refits and for the validation-vs-validation comparisons to GemNet.

### 3.2 Role of each split

The three splits have distinct roles.

- Train split.
  Used to fit model parameters.
  For ridge baselines, train is also the only split used for internal cross-validation during alpha selection.

- Validation split.
  Used as the first truly unseen set for comparing held-out performance during model comparison.
  For GemNet, this is the split used for checkpoint selection.
  For the split-aware linear baselines, this is the correct analogue of GemNet validation MAE.

- Test split.
  Reserved for final held-out reporting after the modeling protocol has been fixed.
  Test should not be used to choose alpha, choose architecture, or choose checkpoints.

## 4. Physical baseline: xTB

### 4.1 Model role

xTB serves as a physics-based baseline rather than a fitted tmQM statistical model. It provides:
- a direct baseline prediction for corrected total energy comparisons, and
- the reference baseline for `Δ_xTB` residual learning.

### 4.2 Error calculation

When xTB is evaluated as a standalone baseline on tmQM validation, the error is computed by merging:
- tmQM DFT corrected energies from
  `/Users/kareem/PycharmProjects/deltaML_Paper/Original_Data/ref-corrected_data/tmqm_dft_ref-corr_neutral.csv`
- xTB corrected energies from
  `/Users/kareem/PycharmProjects/deltaML_Paper/Original_Data/ref-corrected_data/tmqm_xtb_ref-corr.csv`

The merge is done by `mol_id`, and for the fair validation comparison only molecules in `tmqm_val_ids.csv` are kept.

The validation MAE is then:
- mean absolute difference between DFT corrected energy and xTB corrected energy on the tmQM validation split.

## 5. Statistical baseline family

All statistical baselines are ridge-regression models. They differ in feature representation and in the target they are fit on.

### 5.1 Shared fitting pattern

The shared baseline fitting philosophy is:
- construct a fixed feature representation per molecule
- fit a ridge regression model
- standardize features before fitting
- select regularization with `RidgeCV` over a logarithmic alpha grid
- save fitted predictions and residuals back to CSV for downstream use

For the split-aware refits, the same baseline definition is preserved, but only the train split is used for fitting and internal alpha selection.

### 5.2 Alpha selection

The split-aware ridge baselines use:
- `alphas = np.logspace(-6, 3, 20)`

The regularization strength is selected only by cross-validation inside the train split.
The validation and test molecules are not used in alpha selection.

## 6. BoE baseline

### 6.1 Definition

`BoE` is the composition-only baseline. It uses element-count features over the fixed tmQM element universe.

### 6.2 Feature representation

- One feature per element count.
- The feature dimensionality is 44.
- These are coarse compositional descriptors with no bond or geometry information.

### 6.3 Target

Standalone BoE is fit directly to:
- `reference_corrected_energy`

It also defines the residual target:
- `E_resid_BoE_tmQM = reference_corrected_energy - E_BoE_tmQM`

This residual is reused by:
- `BoB+BoE`
- `PRE+BoE`
- GemNet-on-BoE-residual models

### 6.4 Original full-data fit

The original BoE fit is a full-dataset tmQM ridge fit. Its MAE is an in-sample fit error and should not be compared directly to GemNet validation MAE.

### 6.5 Split-aware BoE refit

A fair split-aware BoE refit was performed using the GemNet train/validation/test split. Outputs include:
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/boe/boe_train_val_test_metrics.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/boe/boe_val_predictions_train_only.csv`

The BoE validation MAE used in later validation-vs-validation figures is taken from the `val` row of `boe_train_val_test_metrics.csv`.

## 7. BoB baseline

### 7.1 Definition

`BoB` is the bond-topology baseline. It uses bond-count features derived from tmQM `.mol2` bond graphs.

### 7.2 Feature representation

- Unordered element-pair bond counts
- Five harmonized bond channels: `{1, 2, 3, ar, un}`
- 990 unordered element-pair combinations over the 44-element universe
- 4,950 total bond-count features

### 7.3 Targets

Two linear models are used with the same BoB feature space:

- `BoB`:
  target = `reference_corrected_energy`

- `BoB+BoE`:
  target = `E_resid_BoE_tmQM`
  final prediction = `E_BoE_tmQM + predicted_BoB_residual`

### 7.4 Split-aware BoB refits

Outputs:
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/bob/bob_train_val_test_metrics.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/bob_boe/bob_boe_train_val_test_metrics.csv`

These provide held-out validation and test MAE after train-only fitting.

## 8. PRE baseline

### 8.1 Definition

`PRE` is the geometry-aware linear baseline. It uses pairwise radial expansion features derived from tmQM structures stored in `extxyz` format.

### 8.2 Feature representation

- Pairwise radial basis features accumulated over unordered element pairs
- 35 Gaussian radial channels up to 6.0 Å
- 44-element universe
- 34,650 total features

### 8.3 Targets

Two PRE fits are used:

- `PRE`:
  target = `reference_corrected_energy`

- `PRE+BoE`:
  target = `E_resid_BoE_tmQM`
  final prediction = `E_BoE_tmQM + predicted_PRE_residual`

### 8.4 Original Trillium fitting workflow

The PRE fitting workflow on Trillium is documented in:
- `/Users/kareem/PycharmProjects/deltaML_Paper/4_Cross_Domain/PRE_fitting/trillium_cpu_jobs/README.md`

Core files:
- `pre_common.py`: shared feature and fitting utilities
- `build_pre_feature_cache.py`: builds sparse PRE feature cache from `tmQM_centered.extxyz`
- `fit_pre_energy.py`: fits PRE on `reference_corrected_energy`
- `fit_pre_boe_resid.py`: fits PRE on `E_resid_BoE_tmQM`

### 8.5 PRE split-aware validation issue and fair shared-mask evaluation

PRE produced a very small number of catastrophic outliers under naive split-aware validation scoring. A PRE-only cleanup would have been methodologically weak, so a shared outlier mask was constructed across all baseline families.

That summary is stored in:
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/baseline_val_outlier_fairness_summary.csv`

The key idea is:
- align all baselines on the same validation molecules
- identify extreme outliers using a shared robust rule
- remove the union of those flagged molecules from every baseline equally

This preserves fairness while preventing a tiny pathological tail from dominating PRE aggregate metrics.

## 9. GemNet model family on tmQM

### 9.1 Shared GemNet architecture and optimizer template

The available Trillium configs and logs show that the tmQM GemNet models were trained using a common GemNet-T template with target-specific LMDBs and target-specific label normalization.

Common architecture/optimizer pattern:
- model: `gemnet_t`
- `num_spherical: 7`
- `num_radial: 128`
- `num_blocks: 3`
- `emb_size_atom: 512`
- `emb_size_edge: 512`
- `emb_size_trip: 64`
- `cutoff: 6.0`
- `max_neighbors: 50`
- `rbf: gaussian`
- `cbf: spherical_harmonics`
- `activation: silu`
- `use_pbc: false`
- `otf_graph: true`
- trainer: energy only
- forces not regressed
- optimizer: `AdamW`
- initial learning rate: `5e-4`
- scheduler: `ReduceLROnPlateau`
- factor: `0.8`
- patience: `3`
- max epochs: `80`
- batch size: `4`
- eval batch size: `4`
- gradient clipping: `10`
- EMA decay: `0.999`
- loss on energy: MSE

The configs also normalize labels using train-split mean and standard deviation, which are encoded in the YAML files or in the converted training config printed in the logs.

### 9.2 GemNet target families used in tmQM

The tmQM GemNet family currently includes at least the following run types, as documented by local log names and saved test prediction files:

- `totE`
  direct GemNet trained on the total corrected energy target

- `xtb_resid`
  GemNet trained on `reference_corrected_energy - E_xTB`

- `boe`
  GemNet trained on `E_resid_BoE_tmQM`

- `bob`
  GemNet trained on BoB residual target

- `bob_boe`
  GemNet trained on the BoE-residual target defined for the BoB+BoE composite family

- `boe_xtb`
  GemNet trained on a residual target associated with the combined BoE+xTB baseline family

- `dg_totE`
  `Delta-GemNet(TotE)` family, treated in project naming as a distinct residual formulation relative to the total-energy GemNet baseline

- `pre_energy`
  GemNet associated with the PRE energy target family

- `pre_boe`
  GemNet associated with the PRE-on-BoE-residual family

The active local logs are:
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_bob_295927.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_bob_boe_295926.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_boe_295928.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_boe_xtb_307331.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_dg_totE_308228.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_pre_boe_315064.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_pre_energy_315065.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_totE_297079.log`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/gemnet_xtb_resid_297080.log`

### 9.3 GemNet train/val/test datasets

GemNet is trained on LMDBs specific to the target family. Each config points to three LMDBs:
- train LMDB
- validation LMDB
- test LMDB

For example, the BoB residual config uses:
- `tmqm_bob_residuals_train.lmdb`
- `tmqm_bob_residuals_val.lmdb`
- `tmqm_bob_residuals_test.lmdb`

and the BoB+BoE residual config uses:
- `tmqm_bob_on_boe_residuals_train.lmdb`
- `tmqm_bob_on_boe_residuals_val.lmdb`
- `tmqm_bob_on_boe_residuals_test.lmdb`

The same pattern is used for other residual families.

### 9.4 GemNet validation error calculation

GemNet validation MAE is taken from the held-out validation evaluations recorded during training.

Operationally, the local analysis code extracts the best validation energy MAE from the training logs by scanning for validation blocks and selecting the minimum validation `energy_mae` encountered during training.

This means the GemNet validation MAE used in comparisons is:
- an out-of-sample metric on the tmQM validation split
- associated with the best validation checkpoint
- not an in-sample training error

### 9.5 GemNet test prediction workflow

For final test evaluation, the best checkpoint for each GemNet model is used in `predict` mode on the corresponding tmQM test LMDB. The saved `.npz` outputs are stored in:
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/tmqm_test_predictions_best/`

Current files include:
- `bob_best_is2re_predictions.npz`
- `bob_boe_best_is2re_predictions.npz`
- `boe_best_is2re_predictions.npz`
- `boe_xtb_best_is2re_predictions.npz`
- `dg_totE_best_is2re_predictions.npz`
- `pre_boe_best_is2re_predictions.npz`
- `pre_energy_best_is2re_predictions.npz`
- `totE_best_is2re_predictions.npz`
- `xtb_resid_best_is2re_predictions.npz`

These `.npz` files contain predicted energies and IDs and are aligned back to the reference LMDB truths for final test scoring.

## 10. Composite-model construction

A composite model combines:
- a baseline prediction, and
- a learned residual prediction

The generic formula is:
- `E_composite = E_baseline + E_residual_model`

Examples:
- `xTB + ΔML`:
  `E_pred = E_xTB + predicted_(DFT - xTB residual)`

- `BoE + GemNet`:
  `E_pred = E_BoE_tmQM + predicted_(DFT - BoE residual)`

- `BoB+BoE + GemNet`:
  `E_pred = E_BoE_tmQM + predicted_(BoB second-stage residual or GemNet residual, depending on model family)`

- `PRE+BoE`:
  `E_pred = E_BoE_tmQM + predicted_PRE_residual`

For parity plots and summary tables, the composite prediction is always compared against the total corrected energy unless the panel is explicitly a residual-space panel.

## 11. How validation and test errors are computed

### 11.1 Linear baseline validation MAE

For split-aware linear baselines:
- fit only on train molecules
- choose alpha only by train-internal CV
- predict on the held-out validation split
- compute MAE, RMSE, residual standard deviation, and `R^2` on validation

This is the correct baseline-side counterpart to GemNet validation MAE.

### 11.2 GemNet validation MAE

For GemNet:
- the model is trained on train LMDB
- validation is run periodically during training
- the best validation `energy_mae` is used as the model-selection metric

### 11.3 Final test metrics

For both baseline and GemNet families, final test metrics are computed only after the protocol is fixed. For GemNet this means:
- select best checkpoint by validation MAE
- run prediction on the test LMDB
- align `.npz` predictions to LMDB truths
- compute final MAE, RMSE, residual standard deviation, `R^2`, and normalized metrics if needed

### 11.4 Normalized metrics

Where reported, normalized MAE/RMSE/Stdev are computed by dividing by the IQR of the corresponding target distribution.

This matters because residual targets and total-energy targets occupy very different numeric scales. A raw MAE comparison between direct-total-energy learning and residual learning can therefore be misleading unless the target scale is taken into account.

## 12. Which quantities are appropriate for which comparison

- Use full-data fitted baseline MAE only for:
  - defining residual targets
  - describing how well a baseline fits the entire tmQM dataset in-sample
  - cross-domain baseline construction where the full fitted baseline is intentionally used

- Use split-aware baseline validation MAE for:
  - fair comparison against GemNet validation MAE
  - statements about held-out tmQM generalization under the GemNet split

- Use test MAE/test ranking metrics for:
  - final held-out reporting
  - final model comparison tables

## 13. Practical file map for paper writing

### Baseline summaries and split-aware results
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/PRE_trillium_fitting/baseline_models_summary.md`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/boe/boe_train_val_test_metrics.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/bob/bob_train_val_test_metrics.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/bob_boe/bob_boe_train_val_test_metrics.csv`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation/baseline_val_outlier_fairness_summary.csv`

### GemNet training provenance
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/logs/`
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/configs/`

### GemNet final test predictions
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/tmqm_test_predictions_best/`

### Split membership
- `/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/split_ids/`

## 14. Suggested condensed wording for the paper

A concise methodology summary that is faithful to the implementation would be:

> We compared physical, linear statistical, and graph-neural models on tmQM using a common corrected total-energy target. The physical baseline was xTB. The statistical baselines were ridge-regression models fit on composition-only (`BoE`), bond-topology (`BoB`), and pairwise radial expansion (`PRE`) features, together with stacked variants that learned the residual to the BoE baseline (`BoB+BoE`, `PRE+BoE`). For fair comparison to GemNet validation performance, these linear baselines were refit on the tmQM training split only, with regularization selected by cross-validation inside the training set, and evaluated on the held-out tmQM validation split. GemNet-T models were trained either directly on the total corrected energy or on residual targets defined relative to a baseline model. The best validation checkpoint was selected using the held-out validation energy MAE, and final test metrics were obtained by running the selected checkpoint on the corresponding test LMDB and aligning the resulting predictions back to reference energies. Composite predictions were formed by adding the learned residual prediction back to the corresponding baseline estimate.

## 15. Caveats and naming notes

A few model-family names in the Trillium logs, especially `boe_xtb` and `dg_totE`, follow project-internal naming conventions. Their existence and training provenance are clear from the logs and prediction files, but if those models are discussed in detail in the paper text, their residual formulas should be verified against the exact LMDB-generation scripts used to create their train/val/test datasets.

For the core methodology sections, however, the baseline families `BoE`, `BoB`, `PRE`, the physical baseline `xTB`, the direct `TotE` GemNet model, and the standard residual GemNet families are already well specified by the local artifacts summarized above.
