# Expanded Target Panel Implementation Status

Last updated: 2026-06-19 08:22.

## Implemented Locally

New folder:

```text
New_Analysis/expanded_target_panel/
```

Scripts:

```text
expanded_panel_common.py
prepare_expanded_targets.py
run_expanded_ridge_targets.py
run_expanded_smoothness.py
README.md
```

The shared registry defines nine targets:

```text
DFT_ref,BoE
xTB_ref,BoE
Delta^xTB_ref,BoE
Delta^BoB_ref,BoE
Delta^PRE_ref,BoE
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

## Canonical Targets Created Locally

Created:

```text
New_Analysis/datasets/tmQM__BoB-ref-BoE.csv
New_Analysis/datasets/tmQM__PRE-ref-BoE.csv
New_Analysis/expanded_target_panel/target_registry.csv
New_Analysis/expanded_target_panel/target_registry.json
```

Created after Trillium rsync:

```text
New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv
New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv
```

All nine target CSVs are now present locally.

## Verified Locally

All scripts compile with the local fair-chem wrapper:

```bash
New_Analysis/local_analysis_python.sh -m py_compile ...
```

Smoke-tested ridge fitting for seven available targets:

```text
BoB -> 7 targets
PRE -> 7 targets
PRE+A -> 7 targets
```

Smoke-tested Dirichlet smoothness for seven available targets:

```text
BoB space, raw/std
PRE space, raw/std
PRE+A space, raw/std
```

## Full Local Outputs Created

Full BoB ridge fit for all nine targets:

```text
New_Analysis/expanded_target_panel/outputs/ridge/bob_target_metrics.csv
New_Analysis/expanded_target_panel/outputs/ridge/bob_alpha_grid_results.json
New_Analysis/expanded_target_panel/outputs/ridge/bob_predictions_*.csv
```

Full smoothness for all nine targets:

```text
New_Analysis/expanded_target_panel/outputs/smoothness/bob_space_dirichlet_val.csv
New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv
New_Analysis/expanded_target_panel/outputs/smoothness/pre_plus_a_space_dirichlet_val.csv
```

Primary smoothness metric:

```text
edge_avg_norm_by_population_var =
mean_edges((y_i - y_j)^2) / Var_population(y)
```

Auxiliary scale-aware metric:

```text
edge_rmse_over_iqr =
sqrt(mean_edges((y_i - y_j)^2)) / IQR(y)
```

## Pending Statistical Ridge Fits

Full PRE ridge over all nine targets was not completed locally. PRE has already been fit for the original five targets, and only these four expanded targets are missing for a complete nine-target panel:

```text
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

On Trillium, a broad `-6:3:20` PRE run completed `BoB_ref,BoE` and `PRE_ref,BoE`, then stalled on `PRE+A_ref,BoE` before writing the combined metrics table. It should be rerun cleanly with the focused PRE grid.

Run these on Trillium:

```text
PRE ridge: bob,pre,pre_plus_a,deltapre_plus_a with 100,335.9818286283781,1000,3000
PRE+A ridge: all nine targets with 1000,3000,10000
```

The scripts are validated locally with smoke tests.

## Trillium Work Needed

Sync the expanded panel scripts and new target CSVs to Trillium manually.

Then run:

```text
PRE ridge for bob,pre,pre_plus_a,deltapre_plus_a
PRE+A ridge for nine targets
GemNet on BoB_ref,BoE; PRE_ref,BoE; PRE+A_ref,BoE
SchNet on BoB_ref,BoE; PRE_ref,BoE; PRE+A_ref,BoE
```

Use the existing protocols:

```text
GemNet: 24 h
SchNet: 2 h
```

No QM9/OOD logic is part of this in-domain expanded target panel until these in-domain metrics are complete.
