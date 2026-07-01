# Clean Energy GemNet Run: `y - b_comp,long|comp,short`

This bundle prepares and launches the next GemNet target-design run on the
Clean Energy GPU machine.

## Scientific Target

Run GemNet on:

```text
y - b_comp,long|comp,short
```

Equivalent residual definition:

```text
r_train = r_BoE - b_long|comp,short
```

Target column:

```text
E_resid_BoE_minus_PRE_6plus_on_after_BoE_PREresid_tmQM
```

This is the intended hybrid target because it removes composition and
conditionally isolated long-range structure, while preserving the short/local
component for GemNet.

Do not confuse this with:

```text
y - b_comp,short,long
```

That fully stacked residual subtracts the short PRE component and is not the
next GemNet target.

## Required Input

The preparation script expects this file on Clean Energy:

```text
/home/kareem/deltaML_project/Target_Design/data/b_comp_short_long/hybrid_gemnet_target.csv
```

It must contain:

```text
mol_id
E_resid_BoE_minus_PRE_6plus_on_after_BoE_PREresid_tmQM
```

On Nibi this file was created at:

```text
/scratch/gameelka/deltaML_project/Target_Design/data/b_comp_short_long/hybrid_gemnet_target.csv
```

## Clean Energy Defaults

The preparation script defaults to:

```text
project root:      /home/kareem/deltaML_project
train root:        /home/kareem/deltaML_project/train
source LMDB root:  /home/kareem/deltaML_project/lmdb_files/tmqm_totE
output LMDB dir:   /home/kareem/deltaML_project/lmdb_files/tmqm_b_comp_long_conditioned_residuals
run root:          /home/kareem/b_comp_long_conditioned_residuals_reproducibility_2026_05
conda env:         fairchem-v110
python:            /home/kareem/miniconda3/envs/fairchem-v110/bin/python
scale file:        /home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/train/scaling_factors/gemnet-dT.json
```

Override any of these with environment variables before running the preparation
script:

```text
DELTAML_PROJECT
FAIRCHEM_TRAIN_ROOT
RUN_ROOT
TARGET_CSV
SOURCE_LMDB_ROOT
OUTPUT_LMDB_DIR
TEMPLATE_CONFIG
GEMNET_SCALE_FILE
PYTHON_BIN
```

## Upload From Mac

Run this yourself from the Mac because cluster access requires your MFA.

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/cleanenergy/hybrid_conditional_long_run_bundle/ \
  kareem@cleanenergy-gpu:~/hybrid_conditional_long_run_bundle/
```

Also upload the hybrid target CSV. If it is already downloaded locally into
`Target_Design/gemnet_runs/cleanenergy/hybrid_conditional_long_run_bundle/inputs/`,
use:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  Target_Design/gemnet_runs/cleanenergy/hybrid_conditional_long_run_bundle/inputs/hybrid_gemnet_target.csv \
  kareem@cleanenergy-gpu:/home/kareem/deltaML_project/Target_Design/data/b_comp_short_long/hybrid_gemnet_target.csv
```

If the CSV is still only on Nibi, first copy it from Nibi to the local `inputs/`
folder, then run the upload command above.

## Prepare On Clean Energy

```bash
ssh -i ~/.ssh/myKey kareem@cleanenergy-gpu

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairchem-v110
unset PYTHONPATH

cd ~/hybrid_conditional_long_run_bundle

python prepare_hybrid_conditional_long_cleanenergy.py
```

The script will:

1. Read the hybrid target CSV.
2. Copy the existing `tmqm_totE` train/val/test LMDB graph objects.
3. Replace `y` and `y_relaxed` with the hybrid target values.
4. Compute train target mean/std.
5. Write a Clean Energy FairChem config.
6. Write a preflight check and a `nohup` launcher.

## Preflight

```bash
python /home/kareem/b_comp_long_conditioned_residuals_reproducibility_2026_05/preflight_check.py
```

Confirm:

```text
train: True
val: True
test: True
scale_file: True
target_mean: finite number
target_std: positive number
```

## Launch With `nohup`

Clean Energy should run only one GemNet training at a time.

```bash
cd /home/kareem/b_comp_long_conditioned_residuals_reproducibility_2026_05
./run_nohup.sh
```

Then follow:

```bash
tail -f /home/kareem/b_comp_long_conditioned_residuals_reproducibility_2026_05/runs/b_comp_long_conditioned_residuals_seed_101/b_comp_long_conditioned_residuals_seed_101.nohup.log
```

The run is expected to take roughly 24 hours.
