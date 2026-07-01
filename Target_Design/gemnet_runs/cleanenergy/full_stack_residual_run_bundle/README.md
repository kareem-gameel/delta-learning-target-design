# Clean Energy GemNet Run: `y - b_comp,short,long`

This bundle prepares the remaining statistical diagnostic GemNet run on the
Clean Energy GPU machine.

## Scientific Target

Run GemNet on:

```text
y - b_comp,short,long
```

Target column:

```text
E_resid_after_BoE_PRE_PRE6p_tmQM
```

This is the residual after the fully stacked statistical baseline:

```text
b_comp,short,long = b_comp + b_short|comp + b_long|comp,short
```

Important distinction:

```text
y - b_comp,short,long
```

is **not** the main target-design hypothesis run. It removes the short/local
component that GemNet is expected to learn. The main hypothesis run is:

```text
y - b_comp,long|comp,short
```

which is already prepared/launched separately. This bundle exists so the fully
stacked residual can be tested as a diagnostic after the current Clean Energy
run finishes.

## Required Input

Expected source on Nibi:

```text
/scratch/gameelka/deltaML_project/pre_longrange/PRE_6+/processed/b_comp_short_long/tmQM__b_comp_short_long__pred_and_residual.csv
```

Expected local bundle input:

```text
Target_Design/gemnet_runs/cleanenergy/full_stack_residual_run_bundle/inputs/tmQM__b_comp_short_long__pred_and_residual.csv
```

Expected Clean Energy destination:

```text
/home/kareem/deltaML_project/pre_longrange/PRE_6+/processed/b_comp_short_long/tmQM__b_comp_short_long__pred_and_residual.csv
```

## Clean Energy Defaults

```text
project root:      /home/kareem/deltaML_project
train root:        /home/kareem/deltaML_project/train
source LMDB root:  /home/kareem/deltaML_project/lmdb_files/tmqm_totE
target CSV:        /home/kareem/deltaML_project/pre_longrange/PRE_6+/processed/b_comp_short_long/tmQM__b_comp_short_long__pred_and_residual.csv
output LMDB dir:   /home/kareem/deltaML_project/lmdb_files/tmqm_b_comp_short_long_residuals
run root:          /home/kareem/b_comp_short_long_residuals_reproducibility_2026_05
conda env:         fairchem-v110
python:            /home/kareem/miniconda3/envs/fairchem-v110/bin/python
scale file:        /home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/train/scaling_factors/gemnet-dT.json
```

On Clean Energy, the existing total-energy LMDBs have previously been found at:

```text
/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_totE
```

So when preparing, set:

```bash
SOURCE_LMDB_ROOT=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_totE
OUTPUT_LMDB_DIR=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_b_comp_short_long_residuals
```

## Upload From Mac

Run yourself from the Mac because cluster access requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/cleanenergy/full_stack_residual_run_bundle/ \
  kareem@142.1.107.46:~/full_stack_residual_run_bundle/
```

Also upload the target CSV after downloading it from Nibi into `inputs/`:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/cleanenergy/full_stack_residual_run_bundle/inputs/tmQM__b_comp_short_long__pred_and_residual.csv \
  kareem@142.1.107.46:/home/kareem/deltaML_project/pre_longrange/PRE_6+/processed/b_comp_short_long/tmQM__b_comp_short_long__pred_and_residual.csv
```

## Prepare On Clean Energy

Do not start this while another GemNet training is running.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairchem-v110
unset PYTHONPATH

cd ~/full_stack_residual_run_bundle

SOURCE_LMDB_ROOT=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_totE \
OUTPUT_LMDB_DIR=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_b_comp_short_long_residuals \
python prepare_full_stack_residual_cleanenergy.py
```

## Preflight

```bash
python /home/kareem/b_comp_short_long_residuals_reproducibility_2026_05/preflight_check.py
```

Confirm:

```text
train: True
val: True
test: True
scale_file: True
target_std: positive number
max_epochs: 80
```

## Launch With `nohup`

Only after the current Clean Energy GemNet run is finished:

```bash
cd /home/kareem/b_comp_short_long_residuals_reproducibility_2026_05
./run_nohup.sh
```

Then follow the printed log path.

## 2026-05-12 Status Note

The Clean Energy run for `b_comp_short_long_residuals_seed_101` stopped before
80 epochs because the filesystem filled up while TensorBoard was writing event
data:

```text
OSError: [Errno 28] No space left on device
```

The process was no longer running when checked with the PID file. The latest
visible log lines were around epoch `3.81e+01` (`~38.1`) and step `5.41e+05`.
This should be treated as an interrupted run, not a cleanly finished run.

Before resuming or restarting:

1. Check disk usage under `/home/kareem`.
2. Identify and preserve checkpoints from the run directory.
3. Clean only disposable logs/caches after confirming checkpoint locations.
4. Decide whether epoch 38 is sufficient for the diagnostic, or resume after
   freeing disk space.
