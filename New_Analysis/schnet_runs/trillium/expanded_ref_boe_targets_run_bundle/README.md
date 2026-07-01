# SchNet Expanded `ref,BoE` Target Runs

This bundle trains SchNet on the four expanded targets missing from the 9-target
panel:

```text
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

It uses the same train/val/test LMDB geometry objects as the GemNet runs, but
rewrites the scalar target and replaces the model block with SchNet.

## Sync From Mac

Run locally because Trillium requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/schnet_runs/trillium/expanded_ref_boe_targets_run_bundle/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/datasets/tmQM__BoB-ref-BoE.csv \
  New_Analysis/datasets/tmQM__PRE-ref-BoE.csv \
  New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv \
  New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/datasets/
```

## Prepare LMDBs and Configs

Run on Trillium:

```bash
cd /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" -c "import fairchem.core.models.schnet, lmdb, pandas, torch, yaml; print('env ok')"

"$PY" prepare_expanded_schnet_ref_boe_targets_trillium.py \
  --targets all \
  --force

"$PY" preflight_check.py
```

Expected configs:

```text
/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_bob_ref_boe_seed_101.yml
/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_pre_ref_boe_seed_101.yml
/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_pre_plus_a_ref_boe_seed_101.yml
/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_delta_pre_plus_a_ref_boe_seed_101.yml
```

The prep script skips molecules missing from fitted-baseline target CSVs and
records skipped counts in:

```text
/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/summaries/schnet_expanded_ref_boe_targets_manifest.csv
```

## Debug-Node Smoke Test

Inside your live debug GPU allocation:

```bash
cd /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

CFG=/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_delta_pre_plus_a_ref_boe_seed_101.yml
RUN_DIR=/scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/runs/smoke_delta_pre_plus_a
mkdir -p "$RUN_DIR"

cd /scratch/gameelka/deltaML_project/train
timeout 15m "$PY" -u /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/main_schnet.py \
  --mode train \
  --config-yml "$CFG" \
  --identifier schnet_delta_pre_plus_a_ref_boe_smoke \
  --seed 101 \
  --run-dir "$RUN_DIR" \
  --timestamp-id schnet_delta_pre_plus_a_ref_boe_smoke \
  2>&1 | tee "$RUN_DIR/smoke.train.log"
```

The timeout is fine if the log shows training starts without data/model errors.

## Submit Full 2-Hour Runs

Submit from the proper GPU login node:

```bash
cd /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06

for CFG in \
  /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_bob_ref_boe_seed_101.yml \
  /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_pre_ref_boe_seed_101.yml \
  /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_pre_plus_a_ref_boe_seed_101.yml \
  /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/configs/schnet_delta_pre_plus_a_ref_boe_seed_101.yml
do
  sbatch --partition=compute --time=2:00:00 \
    --export=ALL,PYTHON_BIN="$HOME/miniforge3/envs/fairchem-conda/bin/python" \
    run_expanded_ref_boe_schnet_train.sbatch "$CFG"
done
```

Monitor:

```bash
squeue --me
tail -f /scratch/gameelka/schnet_expanded_ref_boe_targets_2026_06/jobs/slurm-<JOBID>.out
```
