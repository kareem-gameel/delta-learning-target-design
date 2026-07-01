# PRE+A Ridge Baseline

This workflow tests whether adding compact angular environment features to
pair-only PRE produces a better split-aware baseline for `DFT_ref,BoE`.

Definitions:

```text
PRE+A_ref,BoE = Ridge([PRE, A] -> DFT_ref,BoE)
Delta^PRE+A_ref,BoE = DFT_ref,BoE - PRE+A_ref,BoE
```

This is an in-domain tmQM analysis only. It does not train GemNet/SchNet and
does not run QM9 transfer.

## Angular Descriptor

The angular block is a central-element angular environment histogram:

```text
A[Z_center, radial_shell_pair, angle_bin] += 1
```

Defaults:

```text
cutoff = 6.0 Å
radial shell edges = [0.0, 1.3, 1.8, 2.3, 3.0, 4.2, 6.0]
n_angle_bins = 12, uniform in cos(theta)
```

With 44 central element types this gives:

```text
A dims = 44 * 21 * 12 = 11,088
PRE dims = 34,650
PRE+A dims = 45,738
```

This is not a full angular force field. It captures central-element angular
environment and coordination geometry, but not fully typed angles such as
`C-N-O` vs `H-N-H`.

## Inputs

```text
truth: New_Analysis/datasets/tmQM__DFT_ref-BoE.csv
target column: DFT_ref-BoE
splits: trillium_project/split_ids/tmqm_{train,val,test}_ids.csv
PRE cache: New_Analysis/features/PRE/trillium_cache/tmqm_pre_features.npz
geometry: Original_Data/xyz_files/tmqm_curated.extxyz
```

## Train-Only Preprocessing Rule

For modeling, preprocessing must not use validation or test rows.

The ridge script therefore:

```text
1. builds X = [PRE | A]
2. selects train rows from the GemNet split
3. drops zero-variance channels using train rows only
4. fits StandardScaler(with_mean=False) on train rows only
5. fits Ridge/CV on train rows only
6. applies the same mask and scaler to val/test
```

## Trillium Commands

Activate an environment with `numpy`, `pandas`, `scipy`, `sklearn`, and
`joblib`:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"
```

Build a small angular-feature smoke cache:

```bash
"$PY" New_Analysis/pre_plus_a/build_angular_features.py \
  --max-molecules 500 \
  --progress-every 100 \
  --n-jobs 4 \
  --parallel-prefer processes \
  --batch-size 64 \
  --out-dir New_Analysis/features/PRE_A_smoke
```

Smoke test the ridge workflow:

```bash
"$PY" New_Analysis/pre_plus_a/run_pre_plus_a_baseline.py \
  --angular-cache New_Analysis/features/PRE_A_smoke \
  --max-rows-per-split 200 \
  --alphas=1.0 \
  --n-jobs 2 \
  --out-dir New_Analysis/pre_plus_a/outputs_smoke
```

Build the full angular feature cache:

```bash
"$PY" New_Analysis/pre_plus_a/build_angular_features.py \
  --progress-every 1000 \
  --n-jobs 128 \
  --parallel-prefer processes \
  --batch-size 512 \
  --out-dir New_Analysis/features/PRE_A
```

The angular feature builder is CPU-bound. It parallelizes over molecules:
each worker computes sparse angular counts for one molecule, then the main
process writes rows in deterministic `mol_id` order. On the tested Trillium
debug/compute CPU node, `--n-jobs 128` was the fastest measured setting for a
1000-molecule smoke run. Using more workers was not beneficial because system
and process overhead increased.

If a serial feature-build job is already running, it will not speed up in place.
Sync the patched script, cancel the old job, and restart with the parallel
flags.

Measured Trillium debug-node feature build record from 2026-06-18:

```text
serial: 250 molecules ~133 s
n_jobs=8:   250 molecules 17.4 s
n_jobs=16:  250 molecules 10.0 s
n_jobs=32:  250 molecules 6.7 s
n_jobs=64:  1000 molecules 17.0 s
n_jobs=128: 1000 molecules 13.9 s
n_jobs=200: 1000 molecules 14.4 s
```

Full debug-node run:

```text
command: build_angular_features.py --n-jobs 128 --parallel-prefer processes --batch-size 512
out_dir: New_Analysis/features/PRE_A_debug_full
rows: 71,038 tmQM molecules
shape: (71038, 11088)
nnz: 53,691,020
mean_nnz_per_row: 755.807
n_elements: 44
wall time: 6 min 8 s
user CPU time: 326 min 8 s
```

Run the full split-aware PRE+A fit:

```bash
"$PY" New_Analysis/pre_plus_a/run_pre_plus_a_baseline.py \
  --alphas=-6:3:20 \
  --n-jobs 4 \
  --out-dir New_Analysis/pre_plus_a/outputs
```

Compare PRE vs PRE+A:

```bash
"$PY" New_Analysis/pre_plus_a/compare_pre_vs_pre_plus_a.py \
  --out-dir New_Analysis/pre_plus_a/outputs
```

## Outputs

```text
New_Analysis/pre_plus_a/outputs/pre_plus_a_predictions_dft_ref_boe.csv
New_Analysis/pre_plus_a/outputs/pre_plus_a_metrics_raw.csv
New_Analysis/pre_plus_a/outputs/pre_vs_pre_plus_a_common_mask_metrics.csv
New_Analysis/pre_plus_a/outputs/pre_plus_a_alpha_grid_results.json
New_Analysis/pre_plus_a/outputs/tmQM__PRE_plus_A_ref_BoE__pred_and_residual.csv
New_Analysis/pre_plus_a/outputs/tmQM__Delta_PRE_plus_A-ref-BoE.csv
New_Analysis/pre_plus_a/outputs/pre_plus_a_ridge_model.joblib
```

## Decision Gate

Proceed to GemNet/SchNet training on `Delta^PRE+A_ref,BoE` only if:

```text
PRE+A val MAE improves over PRE val MAE by >= 5%
OR
Delta^PRE+A has clearly better normalized scale/smoothness than Delta^PRE.
```
