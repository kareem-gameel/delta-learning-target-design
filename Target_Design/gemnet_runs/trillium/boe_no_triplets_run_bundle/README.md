# GemNet BoE Residual No-Triplets Run

This bundle runs the existing Trillium GemNet target:

```text
DFT_refcorr - BoE
```

with the same FairChem/GemNet config used for the prior BoE residual run, but
with the GemNet triplet interaction branch disabled through the FairChem
overlay used in the SW no-triplets ablation.

## Target

The target is the verified `boe` residual target:

```text
CSV column: E_resid_BoE_tmQM
Meaning: DFT_refcorr - BoE
```

Verified Trillium config:

```text
/scratch/gameelka/BoE_residuals_reproducibility_2026_04/configs/BoE_residuals_seed_101.yml
```

Verified LMDB paths:

```text
/scratch/gameelka/deltaML_project/lmdb_files/tmqm_boe_residuals/tmqm_boe_residuals_train.lmdb
/scratch/gameelka/deltaML_project/lmdb_files/tmqm_boe_residuals/tmqm_boe_residuals_val.lmdb
/scratch/gameelka/deltaML_project/lmdb_files/tmqm_boe_residuals/tmqm_boe_residuals_test.lmdb
```

Verified normalization from the original config:

```text
target_mean = -5.429754747386332e-06
target_std  = 0.09117014494310192
```

## No-Triplets Ablation

The run expects this overlay to exist on Trillium:

```text
/scratch/gameelka/fairchem_no_triplets_overlay
```

and enables the gate:

```bash
export FAIRCHEM_GEMNET_DISABLE_TRIPLETS=1
```

This does not prune or delete GemNet parameters. It imports the patched overlay
and zeroes the triplet branch output `x3` before it is added to the direct
edge/radial pathway.

## Upload

Run from the Mac yourself because Trillium requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/trillium/boe_no_triplets_run_bundle/ \
  gameelka@trillium-gpu.scinet.utoronto.ca:/scratch/gameelka/BoE_residuals_no_triplets_2026_06/
```

## Preflight On Trillium

```bash
cd /scratch/gameelka/BoE_residuals_no_triplets_2026_06

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
export PATH="$HOME/miniforge3/envs/fairchem-conda/bin:$PATH"
hash -r

python preflight_boe_no_triplets_trillium.py
```

The preflight should confirm:

- config exists
- train/val/test LMDBs exist
- overlay import path points to `/scratch/gameelka/fairchem_no_triplets_overlay`
- patch marker `SW_NO_TRIPLETS_ABLATION` is present

## Debug Test

Submit a short debug job first:

```bash
cd /scratch/gameelka/BoE_residuals_no_triplets_2026_06

sbatch --partition=debug --time=1:00:00 \
  run_boe_no_triplets_train.sbatch
```

Monitor:

```bash
squeue --me
tail -f /scratch/gameelka/BoE_residuals_no_triplets_2026_06/jobs/slurm-<JOBID>.out
```

The log should show:

```text
no_triplets_patch_present: True
disable_triplets: 1
[SW_NO_TRIPLETS_ABLATION] triplet branch output x3 is zeroed
```

## Full Compute Run

After the debug test is clean:

```bash
cd /scratch/gameelka/BoE_residuals_no_triplets_2026_06

sbatch --partition=compute --time=24:00:00 \
  run_boe_no_triplets_train.sbatch
```

Outputs:

```text
/scratch/gameelka/BoE_residuals_no_triplets_2026_06/jobs/slurm-<JOBID>.out
/scratch/gameelka/BoE_residuals_no_triplets_2026_06/runs/seed_101/BoE_residuals_no_triplets_seed_101.train.log
/scratch/gameelka/BoE_residuals_no_triplets_2026_06/runs/seed_101/checkpoints/BoE_residuals_no_triplets_seed_101/
```
