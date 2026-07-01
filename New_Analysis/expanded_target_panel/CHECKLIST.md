# Expanded 9-Target Panel Checklist

Last updated: 2026-06-19 13:20.

## Target Panel

Current target IDs:

- [x] `dft` = `DFT_ref,BoE`
- [x] `xtb` = `xTB_ref,BoE`
- [x] `deltaxtb` = `Delta^xTB_ref,BoE`
- [x] `deltabob` = `Delta^BoB_ref,BoE`
- [x] `deltapre` = `Delta^PRE_ref,BoE`
- [x] `bob` = `BoB_ref,BoE`
- [x] `pre` = `PRE_ref,BoE`
- [x] `pre_plus_a` = `PRE+A_ref,BoE`
- [x] `deltapre_plus_a` = `Delta^PRE+A_ref,BoE`

Registry:

- [x] `New_Analysis/expanded_target_panel/target_registry.csv`
- [x] `New_Analysis/expanded_target_panel/target_registry.json`

## Canonical Dataset Files

- [x] `New_Analysis/datasets/tmQM__DFT_ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__xTB_ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__Delta_BoB-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__Delta_PRE-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__BoB-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__PRE-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv`
- [x] `New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv`

## Statistical Ridge Fits on GemNet Splits

### BoB Ridge

- [x] Fit/evaluate on all 9 targets.
- [x] Used broad alpha grid: `-6:3:20`.
- [x] Output: `New_Analysis/expanded_target_panel/outputs/ridge/bob_target_metrics.csv`
- [x] Output: `New_Analysis/expanded_target_panel/outputs/ridge/bob_predictions_*.csv`

### PRE Ridge

Original five targets already exist from previous PRE analysis:

- [x] `dft`
- [x] `xtb`
- [x] `deltaxtb`
- [x] `deltabob`
- [x] `deltapre`

Expanded targets completed locally on 2026-06-19:

- [x] `bob`
- [x] `pre`
- [x] `pre_plus_a`
- [x] `deltapre_plus_a`

Output:

- [x] `New_Analysis/expanded_target_panel/outputs/ridge/pre_target_metrics.csv`
- [x] `New_Analysis/expanded_target_panel/outputs/ridge/pre_predictions_*.csv`
- [x] `New_Analysis/expanded_target_panel/outputs/ridge/pre_alpha_grid_results.json`

Use focused alpha grid:

```text
100,335.9818286283781,1000,3000
```

Rationale: previous PRE high-alpha check for `DFT_ref,BoE` selected alpha near `335.9818`, and broad `-6:3:20` is unnecessary for the remaining targets.

Command used locally:

```bash
New_Analysis/local_analysis_python.sh New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre \
  --targets bob,pre,pre_plus_a,deltapre_plus_a \
  --pre-cache New_Analysis/features/PRE/trillium_cache \
  --alphas=100,335.9818286283781,1000,3000 \
  --n-jobs 4 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
```

### PRE+A Ridge

- [ ] Fit/evaluate on all 9 targets.

Note: attempted locally on 2026-06-19 with the full PRE+A design, but the first target did not complete an alpha evaluation in practical time. Run this part on Trillium.

Use focused high-alpha grid:

```text
1000,3000,10000
```

Rationale: PRE+A `DFT_ref,BoE` check selected `alpha=3000`, with nearby alphas close enough for a focused scan.

Command on Trillium:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

time "$PY" New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre_plus_a \
  --targets all \
  --pre-cache /scratch/gameelka/deltaML_project/pre_trillium_cpu_jobs/cache \
  --angular-cache New_Analysis/features/PRE_A_debug_full \
  --alphas=1000,3000,10000 \
  --n-jobs 3 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
```

## Smoothness Diagnostics

Computed on tmQM validation split with k=10.

Primary metric:

```text
edge_avg_norm_by_population_var =
mean_edges((y_i - y_j)^2) / Var_population(y)
```

Auxiliary metric:

```text
edge_rmse_over_iqr =
sqrt(mean_edges((y_i - y_j)^2)) / IQR(y)
```

Feature spaces:

- [x] BoB raw/std for all 9 targets.
- [x] PRE raw/std for all 9 targets.
- [x] PRE+A raw/std for all 9 targets.

Outputs:

- [x] `New_Analysis/expanded_target_panel/outputs/smoothness/bob_space_dirichlet_val.csv`
- [x] `New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv`
- [x] `New_Analysis/expanded_target_panel/outputs/smoothness/pre_plus_a_space_dirichlet_val.csv`

## GNN Training

Existing:

- [x] GemNet on original five targets.
- [x] SchNet on original five targets.

Missing added targets:

- [ ] GemNet on `BoB_ref,BoE`
- [ ] GemNet on `PRE_ref,BoE`
- [ ] GemNet on `PRE+A_ref,BoE`
- [ ] GemNet on `Delta^PRE+A_ref,BoE` if we decide the residual is part of the GNN target panel.
- [ ] SchNet on `BoB_ref,BoE`
- [ ] SchNet on `PRE_ref,BoE`
- [ ] SchNet on `PRE+A_ref,BoE`
- [ ] SchNet on `Delta^PRE+A_ref,BoE` if we decide the residual is part of the GNN target panel.

Expected walltime:

- GemNet: 24 h per target.
- SchNet: 2 h per target.

## Alpha Policy

- Do not assume the same alpha must win across targets.
- Same feature matrix does not imply same regularization optimum because each target projects onto different feature modes.
- Ridge CV curves are often broad; exact alpha precision is not scientifically important when CV MAE differences are small.
- Use train-only CV, predeclared grids, and report selected alpha plus CV MAE.
- Do not tune alpha using validation/test metrics.

## Next Immediate Steps

1. Run the PRE+A command for all nine targets on Trillium.
2. Download `New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_*` and `pre_plus_a_predictions_*.csv`.
3. Re-run `build_panel_status.py`.
4. Decide whether to train GemNet/SchNet on the four expanded GNN targets.
5. Merge/update final model-target comparison tables and plots.
