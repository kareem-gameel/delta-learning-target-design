# Expanded Target Panel

This folder coordinates the next target-space analysis across five model classes:

- `GemNet`
- `SchNet`
- `BoB` ridge
- `PRE` ridge
- `PRE+A` ridge

The target panel is:

1. `DFT_ref,BoE`
2. `xTB_ref,BoE`
3. `Delta^xTB_ref,BoE`
4. `Delta^BoB_ref,BoE`
5. `Delta^PRE_ref,BoE`
6. `BoB_ref,BoE`
7. `PRE_ref,BoE`
8. `PRE+A_ref,BoE`
9. `Delta^PRE+A_ref,BoE`

## Local Environment

Use the local fair-chem wrapper:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
New_Analysis/local_analysis_python.sh <script.py> ...
```

Do not use bare `python` locally for these analyses.

## Current Local Blocker

The PRE+A high-alpha outputs exist on Trillium but are not downloaded locally yet. Download them first:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
mkdir -p New_Analysis/pre_plus_a/outputs_high_alpha_cv

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/pre_plus_a/outputs_high_alpha_cv/ \
  New_Analysis/pre_plus_a/outputs_high_alpha_cv/
```

After downloading, rerun:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/prepare_expanded_targets.py \
  --require-pre-plus-a
```

This will create:

- `New_Analysis/datasets/tmQM__BoB-ref-BoE.csv`
- `New_Analysis/datasets/tmQM__PRE-ref-BoE.csv`
- `New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv`
- `New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv`
- `New_Analysis/expanded_target_panel/target_registry.csv`

## Local Smoke Tests

These have already been verified locally for the seven targets available before the PRE+A download:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space bob \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 50 \
  --alphas=1.0 \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/ridge

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 50 \
  --alphas=1.0 \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/ridge

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre_plus_a \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 30 \
  --alphas=1.0 \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/ridge
```

Smoothness smoke tests:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space bob \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 300 \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/smoothness

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space pre \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 300 \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/smoothness

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space pre_plus_a \
  --targets dft,xtb,deltaxtb,deltabob,deltapre,bob,pre \
  --max-rows-per-split 200 \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs_smoke/smoothness
```

## Full Local Statistical Runs

After the PRE+A target is available locally, run:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space bob \
  --targets all \
  --alphas=-6:3:20 \
  --n-jobs 4 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre \
  --targets all \
  --alphas=-6:3:20 \
  --n-jobs 4 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre_plus_a \
  --targets all \
  --alphas=1000,3000,10000 \
  --n-jobs 3 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
```

The PRE full run is the longest local statistical step. If it is too slow locally, run the same commands on a Trillium debug node.

## Full Local Smoothness Runs

After all nine target CSVs exist:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space bob \
  --targets all \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs/smoothness

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space pre \
  --targets all \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs/smoothness

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/run_expanded_smoothness.py \
  --feature-space pre_plus_a \
  --targets all \
  --scalings raw,std \
  --out-dir New_Analysis/expanded_target_panel/outputs/smoothness
```

The primary smoothness metric remains:

```text
edge_avg_norm_by_population_var =
mean_edges((y_i - y_j)^2) / Var_population(y)
```

We also save:

```text
edge_rmse_over_iqr =
sqrt(mean_edges((y_i - y_j)^2)) / IQR(y)
```

`raw` means the feature vector is used as stored. `std` means each retained feature channel is divided by its standard deviation over the graph population before kNN construction.

## Trillium Work Needed

Once the nine target CSVs exist locally and are synced to Trillium:

- Train `GemNet` on the added targets: `BoB_ref,BoE`, `PRE_ref,BoE`, `PRE+A_ref,BoE`.
- Train `SchNet` on the same added targets.
- GemNet should use the existing 24-hour training protocol.
- SchNet should use the existing 2-hour training protocol.

Cluster sync requires MFA, so Codex should only provide commands. The user should run `rsync` manually.
