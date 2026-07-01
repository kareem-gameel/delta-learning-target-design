# Trillium SchNet Controlled Runs on `ref,BoE` Targets

This bundle trains SchNet with the same split and target protocol as the GemNet
runs, changing only the model block and SchNet-safe batch settings.

Default targets:

- `DFT_ref,BoE`
- `xTB_ref,BoE`
- `Delta^xTB_ref,BoE = DFT_ref,BoE - xTB_ref,BoE`

The generated code labels are:

- `dft_ref_boe`
- `xtb_ref_boe`
- `delta_xtb_ref_boe`

## What Is Held Fixed

The preparation script reuses the existing GemNet tmQM LMDB graph objects and
only rewrites the scalar energy labels. It preserves:

- train/val/test molecule split
- target normalization from the train split only
- optimizer/scheduler settings copied from the GemNet template
- seed and validation protocol

The model block is replaced with:

```yaml
model:
  name: schnet
  use_pbc: false
  regress_forces: false
  otf_graph: true
  hidden_channels: 128
  num_filters: 128
  num_interactions: 6
  num_gaussians: 50
  cutoff: 6.0
  readout: sum
```

The default SchNet batch size is `16`; this is the main deliberate difference
from GemNet because SchNet is much lighter. Set `BATCH_SIZE=4` if you want the
strictest possible optimizer-step comparison.

## Upload From Mac

Run this yourself because Trillium requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/schnet_runs/trillium/ref_boe_core_run_bundle/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/schnet_ref_boe_core_2026_06/
```

The target CSVs already live under `New_Analysis/datasets/`. If Trillium does
not have them, sync them too:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -avR --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/datasets/tmQM__DFT_ref-BoE.csv \
  New_Analysis/datasets/tmQM__xTB_ref-BoE.csv \
  New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/
```

## Prepare LMDBs and Configs on Trillium

```bash
cd /scratch/gameelka/schnet_ref_boe_core_2026_06

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" -c "import fairchem.core.models.schnet, lmdb, pandas, torch, yaml; print('env ok')"

"$PY" prepare_schnet_ref_boe_core_trillium.py \
  --targets dft_ref_boe,xtb_ref_boe,delta_xtb_ref_boe \
  --eval-every 5000 \
  --batch-size 16 \
  --eval-batch-size 16

"$PY" preflight_check.py
```

Expected split sizes:

```text
train: 56830
val:    7104
test:   7104
```

The prep step writes:

```text
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_dft_ref_boe_seed_101.yml
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_xtb_ref_boe_seed_101.yml
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_delta_xtb_ref_boe_seed_101.yml
```

## Live Debug-Node Smoke Test

Inside a live debug GPU allocation:

```bash
cd /scratch/gameelka/schnet_ref_boe_core_2026_06

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

timeout 15m "$PY" -u main_schnet.py \
  --mode train \
  --config-yml /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_delta_xtb_ref_boe_seed_101.yml \
  --identifier schnet_delta_xtb_ref_boe_seed_101_smoke \
  --seed 101 \
  --run-dir /scratch/gameelka/schnet_ref_boe_core_2026_06/runs/smoke \
  --timestamp-id schnet_delta_xtb_ref_boe_seed_101_smoke
```

If it validates too often, check the config value:

```bash
grep -n "eval_every" /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/*.yml
```

For full runs, keep `eval_every: 5000`; the smoke-test `eval_every: 50` was only
for fast failure detection and is too frequent for real training.

## Submit a 1 Hour Debug Slurm Test

Submit from a login node:

```bash
cd /scratch/gameelka/schnet_ref_boe_core_2026_06

sbatch --partition=debug --time=1:00:00 \
  --export=ALL,PYTHON_BIN="$HOME/miniforge3/envs/fairchem-conda/bin/python" \
  run_schnet_ref_boe_train.sbatch \
  /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_delta_xtb_ref_boe_seed_101.yml
```

Monitor:

```bash
squeue --me
tail -f /scratch/gameelka/schnet_ref_boe_core_2026_06/jobs/slurm-<JOBID>.out
```

## Submit Full Compute Jobs

Start with one target. If throughput looks good, submit the others.

```bash
cd /scratch/gameelka/schnet_ref_boe_core_2026_06

for CFG in \
  /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_dft_ref_boe_seed_101.yml \
  /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_xtb_ref_boe_seed_101.yml \
  /scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_delta_xtb_ref_boe_seed_101.yml
do
  sbatch --partition=compute --time=12:00:00 \
    --export=ALL,PYTHON_BIN="$HOME/miniforge3/envs/fairchem-conda/bin/python" \
    run_schnet_ref_boe_train.sbatch "$CFG"
done
```

Use `--time=24:00:00` if the first run indicates SchNet needs the same walltime
as GemNet, but it should usually be lighter.
