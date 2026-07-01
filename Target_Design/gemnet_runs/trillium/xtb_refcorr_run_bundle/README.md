# Trillium GemNet on xTB Reference-Corrected Energy

This bundle prepares and trains:

```text
GemNet(xTB_refcorr)
```

The target is **not** the DFT total energy and not the DFT-minus-xTB residual.
It is the xTB reference-corrected energy column:

```text
Original_Data/ref-corrected_data/tmqm_xtb_ref-corr_neutral.csv
target column: reference_corrected_energy
```

The neutral tmQM xTB file has `71,038` rows. This is the intended training
universe; we do not use the full ~86.5k charged+neutral tmQM file.

## Files

- `prepare_xtb_refcorr_trillium.py`
  - rewrites the existing `tmqm_totE_matched_{train,val,test}.lmdb` graph objects
    with xTB reference-corrected labels.
  - writes a GemNet config under `/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/configs`.
  - writes a preflight script under `/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/preflight_check.py`.
- `run_xtb_refcorr_train.sbatch`
  - reusable Slurm script.
  - defaults to `compute` for `24:00:00`.
  - can be submitted to `debug` for `1:00:00` using `sbatch --partition=debug --time=1:00:00`.

## Trillium paths

```text
project root:      /scratch/gameelka/deltaML_project
source LMDB root:  /scratch/gameelka/deltaML_project/lmdb_files/tmqm_totE
output LMDB dir:   /scratch/gameelka/deltaML_project/lmdb_files/tmqm_xtb_refcorr
run root:          /scratch/gameelka/xtb_refcorr_reproducibility_2026_05
environment:       fairchem-conda
```

## Upload from Mac

Run this yourself because Trillium requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/trillium/xtb_refcorr_run_bundle/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/

rsync -avR --progress -e "ssh -i ~/.ssh/myKey" \
  Original_Data/ref-corrected_data/tmqm_xtb_ref-corr_neutral.csv \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/
```

## Prepare LMDBs and config on Trillium

```bash
ssh -i ~/.ssh/myKey gameelka@trillium.scinet.utoronto.ca

cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH

python -c "import lmdb, pandas, torch, yaml; print('env ok')"

cd /scratch/gameelka/xtb_refcorr_reproducibility_2026_05
python prepare_xtb_refcorr_trillium.py
python /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/preflight_check.py
```

Expected split sizes should match the existing neutral tmQM split, roughly:

```text
train: 56830
val:    7104
test:   7104
total: 71038
```

## Live debug-node smoke test

First get a live debug GPU node:

```bash
salloc --account=rrg-ovoznyy --partition=debug --time=1:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-node=1
```

Inside the allocation:

```bash
cd /scratch/gameelka/xtb_refcorr_reproducibility_2026_05
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH

python preflight_check.py

timeout 15m bash run_xtb_refcorr_train.sbatch \
  /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/configs/xtb_refcorr_seed_101.yml
```

The `timeout` is intentional for the live smoke test. We only need to verify
that FairChem loads the config, opens the LMDBs, sees the GPU, and starts
training without a path/schema error.

## Submit the same script to debug for 1 hour

```bash
cd /scratch/gameelka/xtb_refcorr_reproducibility_2026_05

sbatch --partition=debug --time=1:00:00 \
  run_xtb_refcorr_train.sbatch \
  /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/configs/xtb_refcorr_seed_101.yml
```

Monitor:

```bash
squeue --me
tail -f /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/jobs/slurm-<JOBID>.out
```

## Submit the full compute job for 24 hours

After the live smoke test and debug Slurm test are clean:

```bash
cd /scratch/gameelka/xtb_refcorr_reproducibility_2026_05

sbatch --partition=compute --time=24:00:00 \
  run_xtb_refcorr_train.sbatch \
  /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/configs/xtb_refcorr_seed_101.yml
```

Monitor:

```bash
squeue --me
tail -f /scratch/gameelka/xtb_refcorr_reproducibility_2026_05/jobs/slurm-<JOBID>.out
```

