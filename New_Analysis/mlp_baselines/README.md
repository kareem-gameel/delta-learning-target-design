# MLP Baselines for Nonlinear Residual Targets

This module trains nonlinear MLP baselines on the same descriptors used by the
linear baselines:

- `MLP_BoB`: BoB features -> `DFT_ref,BoE`
- `MLP_PRE`: PRE features -> `DFT_ref,BoE`
- `MLP_PRE_plus_A`: PRE+A features -> `DFT_ref,BoE`

The immediate goal is to generate nonlinear residual targets:

- `Delta_MLP_BoB-ref-BoE`
- `Delta_MLP_PRE-ref-BoE`
- `Delta_MLP_PRE_plus_A-ref-BoE`

These are not replacements for the linear ridge baselines in the main panel.
They are a controlled capacity experiment: if a nonlinear model removes more
structure from a descriptor space, we can ask whether the resulting residual is
still smooth in that same feature space.

## Preprocessing

The training script is split-aware:

- train-only zero-variance feature filtering
- train-only sparse standardization using `with_mean=False, with_std=True`
- train-only target mean/std normalization
- no validation or test rows are used for filtering, scaling, target
  normalization, early stopping except for validation loss selection, or model
  coefficients

The sparse descriptor matrix is not centered, so molecular-size information
encoded in the descriptor counts is not removed by subtracting feature means.

## Regularization

The MLP uses several regularizers by default:

- bottleneck architecture: `512,128`
- dropout: `0.10`
- AdamW weight decay: `1e-4`
- SmoothL1/Huber loss
- gradient clipping
- early stopping on tmQM validation MAE

This is intentional. An unregularized MLP could overfit the train split and
make the residual target look artificially small without being stable.

## GPU Throughput

By default the script uses:

```text
--dense-cache auto --max-dense-cache-gb 16
```

After train-only feature filtering and sparse scaling, it materializes the full
retained design matrix as `float32` once. If CUDA is active and the dense matrix
is below the memory threshold, the matrix is cached on the GPU. Training batches
then use GPU row indexing instead of repeatedly slicing CSR rows and calling
`toarray()`.

Approximate full tmQM retained dense sizes are:

- BoB: `71038 x 867`, about `0.23 GiB`
- PRE: `71038 x 13230`, about `3.50 GiB`
- PRE+A: `71038 x 20796`, about `5.50 GiB`

These fit comfortably on an H100 80GB. If a future GPU has less memory, use
`--dense-cache none` to fall back to on-demand CSR-to-dense batches.

## Trillium Setup

Sync this module to Trillium yourself:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/mlp_baselines/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/mlp_baselines/
```

Run a GPU smoke test on a debug node:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" New_Analysis/mlp_baselines/train_mlp_baselines.py \
  --feature-spaces bob \
  --target-id dft \
  --pre-cache /scratch/gameelka/deltaML_project/pre_trillium_cpu_jobs/cache \
  --angular-cache New_Analysis/features/PRE_A_debug_full \
  --max-rows-per-split 200 \
  --max-epochs 3 \
  --patience 2 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --device cuda \
  --out-dir New_Analysis/mlp_baselines/outputs_smoke/bob
```

Submit the three full GPU jobs from `trig-login01`:

```bash
cd /scratch/gameelka/deltaML_project

sbatch --export=ALL,FEATURE_SPACE=bob,BATCH_SIZE=4096,EVAL_BATCH_SIZE=8192 \
  New_Analysis/mlp_baselines/run_mlp_baseline_trillium.sbatch

sbatch --export=ALL,FEATURE_SPACE=pre,BATCH_SIZE=1024,EVAL_BATCH_SIZE=2048 \
  New_Analysis/mlp_baselines/run_mlp_baseline_trillium.sbatch

sbatch --export=ALL,FEATURE_SPACE=pre_plus_a,BATCH_SIZE=768,EVAL_BATCH_SIZE=1536 \
  New_Analysis/mlp_baselines/run_mlp_baseline_trillium.sbatch
```

Download outputs when finished:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/mlp_baselines/outputs/ \
  New_Analysis/mlp_baselines/outputs/
```

## Outputs

Each feature-space run writes:

- `mlp_target_metrics.csv`
- `mlp_input_validation.json`
- `mlp_*_predictions_dft.csv`
- `tmQM__MLP_*_ref_BoE__pred_and_residual.csv`
- `tmQM__Delta_MLP_*-ref-BoE.csv`
- `mlp_*_training_history.csv`
- `mlp_*_preprocess.npz`
- `mlp_*_model.pt`

Use the validation MAE and residual target files before deciding whether these
nonlinear residual targets are scientifically useful enough for GemNet/SchNet
training.
