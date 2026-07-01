# GNN Training on MLP Baseline Residual Targets

This run bundle prepares and trains GemNet/SchNet on three nonlinear-baseline
residual targets:

- `Delta^MLP_BoB_ref,BoE`
- `Delta^MLP_PRE_ref,BoE`
- `Delta^MLP_PRE+A_ref,BoE`

The targets come from:

```text
New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv
New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv
New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv
```

## Sync

Run locally from the Mac:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/mlp_residual_gnn_runs/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/mlp_residual_gnn_runs/

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/mlp_baselines/outputs/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/mlp_baselines/outputs/
```

If the MLP outputs were generated directly on Trillium and are still present
there, the second `rsync` is not necessary.

## Prepare LMDBs and Configs

Run on Trillium:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" New_Analysis/mlp_residual_gnn_runs/prepare_mlp_residual_gnn_targets_trillium.py \
  --model gemnet \
  --targets all

"$PY" New_Analysis/mlp_residual_gnn_runs/prepare_mlp_residual_gnn_targets_trillium.py \
  --model schnet \
  --targets all

"$PY" /scratch/gameelka/gemnet_mlp_residuals_2026_06/preflight_check.py
"$PY" /scratch/gameelka/schnet_mlp_residuals_2026_06/preflight_check.py
```

## Submit Training Jobs

Submit from `trig-login01`.

GemNet, 24 h each:

```bash
cd /scratch/gameelka/deltaML_project

for cfg in /scratch/gameelka/gemnet_mlp_residuals_2026_06/configs/*.yml; do
  sbatch New_Analysis/mlp_residual_gnn_runs/run_gemnet_mlp_residual_train.sbatch "$cfg"
done
```

SchNet, 2 h each:

```bash
cd /scratch/gameelka/deltaML_project

for cfg in /scratch/gameelka/schnet_mlp_residuals_2026_06/configs/*.yml; do
  sbatch New_Analysis/mlp_residual_gnn_runs/run_schnet_mlp_residual_train.sbatch "$cfg"
done
```

## Notes

- Generated LMDBs are stored in:

```text
/scratch/gameelka/deltaML_project/lmdb_files/tmqm_mlp_residual_targets/
```

- Missing molecules in target CSVs are skipped and LMDB keys are renumbered
  densely, matching the fix used in the expanded target runs.
- This is in-domain target training only. No stacking or QM9 evaluation is
  included here.
