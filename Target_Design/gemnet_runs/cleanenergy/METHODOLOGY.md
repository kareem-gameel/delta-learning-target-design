# Clean Energy GemNet Training Methodology

This document records the general protocol used to run GemNet/FairChem training
on the Clean Energy GPU machine for tmQM residual-target experiments.

Cluster access requires MFA, so all `ssh` and `rsync` commands should be run
manually by Kareem.

## Machine And Environment

Clean Energy login:

```bash
ssh -i ~/.ssh/myKey kareem@142.1.107.46
```

Use the FairChem environment installed on Clean Energy:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fairchem-v110
unset PYTHONPATH

which python
python -c "import lmdb, pandas, torch, yaml; print('env ok')"
```

Expected Python:

```text
/home/kareem/miniconda3/envs/fairchem-v110/bin/python
```

Important runtime rule:

```text
Run only one GemNet training at a time on Clean Energy.
```

A full GemNet run usually takes roughly one day, depending on target, disk state,
and whether the run finishes all 80 epochs.

## Project Layout On Clean Energy

The local project root on Clean Energy is:

```text
/home/kareem/deltaML_project
```

The FairChem training entrypoint is:

```text
/home/kareem/deltaML_project/train/main.py
```

Several existing source LMDBs live under the older FAIR-Chem project tree:

```text
/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files
```

The total-energy source LMDB root used for target rewriting has been:

```text
/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_totE
```

with split files:

```text
tmqm_totE_matched_train.lmdb
tmqm_totE_matched_val.lmdb
tmqm_totE_matched_test.lmdb
```

The GemNet scaling factors file used by the generated configs is:

```text
/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/train/scaling_factors/gemnet-dT.json
```

## General Target Preparation Logic

For each new residual target, we do not rebuild molecular graphs from scratch.
Instead, we:

1. Start from the existing matched tmQM total-energy LMDBs.
2. Read a target CSV keyed by `mol_id`.
3. Copy each LMDB graph object.
4. Replace the energy target fields with the new residual value.
5. Preserve the geometry, atom types, split membership, `mol_id`, and `sid`
   metadata from the source LMDB.
6. Write new train/val/test LMDBs for the target.
7. Generate a FairChem YAML config from a known working GemNet template.
8. Run a preflight check before launching training.

The target CSV must contain:

```text
mol_id
<target residual column>
```

For example:

```text
E_resid_BoE_minus_PRE_6plus_on_after_BoE_PREresid_tmQM
E_resid_after_BoE_PRE_PRE6p_tmQM
```

The preparation scripts update both energy-like fields used by FairChem, so the
LMDB target agrees with the intended residual column.

## Configuration Preparation

Each Clean Energy run bundle contains a `prepare_*_cleanenergy.py` script. The
script is responsible for:

1. Validating the target CSV and required target column.
2. Locating source train/val/test LMDBs.
3. Rewriting the LMDB target values.
4. Computing the train target mean and standard deviation.
5. Loading a template GemNet config from an existing successful run.
6. Replacing dataset paths with the newly generated LMDBs.
7. Setting the run identifier, run root, checkpoint/log directories, target
   normalizer values, and scaling-factor path.
8. Writing:
   - `configs/<run_name>.yml`
   - `preflight_check.py`
   - `run_nohup.sh`
   - `summaries/*manifest.csv`

Preferred template configs are existing Clean Energy PRE6plus configs, for
example:

```text
/home/kareem/PRE6plus_BoE_residuals_reproducibility_2026_04/configs/PRE6plus_BoE_residuals_seed_101.yml
/home/kareem/PRE6plus_residuals_reproducibility_2026_04/configs/PRE6plus_residuals_seed_101.yml
```

The generated run config should preserve the established GemNet architecture and
training settings unless there is an explicit reason to change them.

Typical expected config properties:

```text
seed: 101
max_epochs: 80
logger: tensorboard
dataset train/val/test src: newly generated LMDB paths
scale_file: gemnet-dT.json
```

## Environment Variables For Preparation Scripts

The preparation scripts support environment-variable overrides. Common variables
are:

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

Use these when the default paths do not match the Clean Energy file layout.

Example:

```bash
SOURCE_LMDB_ROOT=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_totE \
OUTPUT_LMDB_DIR=/home/kareem/FAIR-Chem/deltaE_Project/OCP_GemNet_ML_Project/lmdb_files/tmqm_b_comp_short_long_residuals \
python prepare_full_stack_residual_cleanenergy.py
```

## Preflight Check

Before launching training, always run the generated preflight script:

```bash
python /home/kareem/<run_root>/preflight_check.py
```

Confirm:

```text
train: True
val: True
test: True
scale_file: True
target_mean: finite number
target_std: positive number
max_epochs: 80
```

Do not start training if:

- any LMDB path is missing,
- the scale file is missing,
- target standard deviation is zero or `nan`,
- the config points to the wrong LMDB root,
- another GemNet process is already running.

## Launching Training With nohup

The generated launcher is:

```text
/home/kareem/<run_root>/run_nohup.sh
```

It activates `fairchem-v110`, resets distributed-training environment variables,
sets NCCL compatibility flags for the single-GPU machine, changes into the
FairChem train root, and launches:

```bash
python -u main.py \
  --mode train \
  --config-yml <config.yml> \
  --identifier <run_name> \
  --seed 101 \
  --run-dir <run_dir> \
  --timestamp-id <run_name>
```

Standard launch:

```bash
cd /home/kareem/<run_root>
./run_nohup.sh
```

The launcher prints:

```text
pid
nohup log path
train log path
config path
```

Follow the log:

```bash
tail -f /home/kareem/<run_root>/runs/<run_name>/<run_name>.nohup.log
```

Check process status:

```bash
ps -p "$(cat /home/kareem/<run_root>/runs/<run_name>/<run_name>.pid)" \
  -o pid,stat,etime,%cpu,%mem,cmd
```

## Monitoring And Disk Safety

Clean Energy can fill its root filesystem during long training runs, especially
from TensorBoard event files, package caches, checkpoints, and logs.

Before launching or resuming, check:

```bash
df -h /home/kareem /tmp
du -sh /home/kareem/* 2>/dev/null | sort -h | tail -n 30
```

Useful checkpoint inspection:

```bash
find /home/kareem/<run_root>/runs \
  -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

TensorBoard event files can be large:

```bash
find /home/kareem -type f -name 'events.out.tfevents*' \
  -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 50
```

Do not delete checkpoints unless we have explicitly decided they are no longer
needed. Disposable cleanup targets are usually package caches and very large
event files after preserving the run summary and checkpoint state.

## Resume Procedure

If a run stops before finishing, first check whether the checkpoint exists:

```text
<run_root>/runs/<run_name>/checkpoints/<run_name>/checkpoint.pt
<run_root>/runs/<run_name>/checkpoints/<run_name>/best_checkpoint.pt
```

To resume, create a copy of the original config and add:

```yaml
checkpoint: /home/kareem/<run_root>/runs/<run_name>/checkpoints/<run_name>/checkpoint.pt
```

Then copy `run_nohup.sh` to a resume launcher and point `CFG_PATH` at the resume
config.

Before resuming, inspect validation behavior if possible. If validation has
already plateaued and the run has a good `best_checkpoint.pt`, it may be better
to treat the run as sufficiently complete for diagnostic purposes rather than
spend another full training cycle.

## QM9 Inference From A Trained Clean Energy Checkpoint

After a tmQM training run finishes, QM9 transfer requires a separate prediction
step. The model is not retrained on QM9.

General procedure:

1. Copy the trained run config to an eval config.
2. Replace the dataset `src` with the QM9 LMDB.
3. Set `checkpoint` to the trained `best_checkpoint.pt`.
4. Run FairChem in `predict` mode.
5. Copy the resulting `is2re_predictions.npz` into the project prediction
   folder and update the evaluation manifest.

Prediction command pattern:

```bash
cd /home/kareem/deltaML_project/train

python -u main.py \
  --mode predict \
  --config-yml /home/kareem/<run_root>/eval/configs/<run_name>_qm9_predict.yml \
  --checkpoint /home/kareem/<run_root>/runs/<run_name>/checkpoints/<run_name>/best_checkpoint.pt \
  --identifier <run_name>_qm9 \
  --run-dir /home/kareem/<run_root>/eval/qm9_predict \
  --timestamp-id <run_name>_qm9 \
  2>&1 | tee /home/kareem/<run_root>/eval/qm9_predict/predict.log
```

Then locate the generated prediction file:

```bash
find /home/kareem/<run_root>/eval/qm9_predict \
  -type f \( -name '*.npz' -o -name '*.csv' -o -name '*.pkl' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

## Run-Specific Bundles

Concrete examples are stored here:

```text
Target_Design/gemnet_runs/cleanenergy/hybrid_conditional_long_run_bundle/
Target_Design/gemnet_runs/cleanenergy/full_stack_residual_run_bundle/
```

The hybrid conditional-long bundle prepares:

```text
y - b_comp,long|comp,short
```

The full-stack diagnostic bundle prepares:

```text
y - b_comp,short,long
```

Use the run-specific README in each bundle for target-column names and upload
locations.

## Current Known Clean Energy Runs

Clean Energy logs and bundles currently tracked locally include:

```text
PRE6plus_residuals_seed_101
PRE6plus_BoE_residuals_seed_101
b_comp_long_conditioned_residuals_seed_101
b_comp_short_long_residuals_seed_101
```

The `b_comp_short_long_residuals_seed_101` diagnostic run stopped around epoch
38 because the filesystem filled up while TensorBoard was writing event data.
Treat it as interrupted unless later resumed or explicitly accepted as a
diagnostic checkpoint.

