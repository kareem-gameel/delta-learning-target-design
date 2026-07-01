# Trillium GemNet Training Logs

This folder is the local landing area for raw GemNet training logs from Trillium.

## Layout

- `raw_logs/`: raw `.train.log`, `.log`, and `slurm-*.out` files copied from Trillium.
- `parsed/`: derived learning-curve tables or metrics generated locally from raw logs.
- `../cleanenergy/`: logs/checkpoints copied from the Clean Energy private GPU cluster.

The training-curve notebook lives at:

- `Target_Design/notebooks/gemnet_training_curves.ipynb`


## Locally Recovered Legacy Seed-0 Logs

A later local search found five older Trillium GemNet logs outside this repository, under:

- `/Users/kareem/PycharmProjects/GNN_Hamiltonian/logs/`

They were copied, not moved, into:

- `Target_Design/gemnet_runs/trillium/raw_logs/seed0_legacy/`

Recovered files:

- `gemnet_totE_297079.log`
- `gemnet_xtb_resid_297080.log`
- `gemnet_boe_295928.log`
- `gemnet_bob_295927.log`
- `gemnet_bob_boe_295926.log`

These are raw FairChem/GemNet training logs from February 2026. They contain `Loaded GemNetT`, seed/config information, and training `energy_mae` lines. They do not appear to contain `Total time taken` or epoch-80 completion markers, so treat them as legacy single-run logs rather than the later April 2026 full seed-sweep logs.

## Sync From Trillium

Run these commands yourself from your Mac because Trillium access requires your SSH key and MFA.

## Find Missing Logs On Trillium

If a log is not already local, first search Trillium from an interactive Trillium shell. This is safer than broad `rsync` because the project has several old locations.

```bash
find /scratch/gameelka -type f \( \
  -name 'gemnet*.log' -o \
  -name '*.train.log' -o \
  -name 'slurm-*.out' \
\) 2>/dev/null | sort > /scratch/gameelka/gemnet_log_candidates.txt

grep -Ei \
  'gemnet_(pre|totE|boe|bob|xtb)|PRE_residuals|PRE_BoE|PRE6plus|BoE_residuals|totE_reproducibility|xTB_residuals|BoB' \
  /scratch/gameelka/gemnet_log_candidates.txt
```

For content-based search, use:

```bash
for f in $(cat /scratch/gameelka/gemnet_log_candidates.txt)
do
  if grep -qE 'Loaded GemNetT|energy_mae:|ocp_trainer.py|checkpoint_dir:' "$f" 2>/dev/null
  then
    echo "$f"
  fi
done > /scratch/gameelka/gemnet_log_confirmed_by_content.txt

cat /scratch/gameelka/gemnet_log_confirmed_by_content.txt
```

The missing/extra logs to look for specifically include:

- `gemnet_pre_energy_*` or equivalent: GemNet trained on PRE residual after PRE fitted to total energy.
- `gemnet_pre_boe_*` or equivalent: GemNet trained on residual after PRE fitted to BoE residual.
- `totE_reproducibility_2026_04` seed-sweep logs.
- `PRE_residuals_reproducibility_2026_04` seed-sweep logs.
- `PRE_BoE_residuals_reproducibility_2026_04` seed-sweep logs.

### Older `deltaML_project/train/logs` logs

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
mkdir -p Target_Design/gemnet_runs/trillium/raw_logs/deltaML_project_train_logs

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  --include='*/' \
  --include='*.log' \
  --exclude='*' \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/train/logs/ \
  Target_Design/gemnet_runs/trillium/raw_logs/deltaML_project_train_logs/
```

### April 2026 reproducibility runs

This copies raw training logs and Slurm outputs from each explicit run root, without downloading checkpoints, LMDBs, exports, or TensorBoard files.

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

for root in \
  totE_reproducibility_2026_04 \
  BoE_residuals_reproducibility_2026_04 \
  xTB_residuals_reproducibility_2026_04 \
  BoB_residuals_reproducibility_2026_04 \
  PRE_residuals_reproducibility_2026_04 \
  BoB_BoE_residuals_reproducibility_2026_04 \
  BoE_xTB_residuals_reproducibility_2026_04 \
  PRE_BoE_residuals_reproducibility_2026_04 \
  PRE6plus_residuals_reproducibility_2026_04 \
  PRE6plus_BoE_residuals_reproducibility_2026_04
do
  mkdir -p "Target_Design/gemnet_runs/trillium/raw_logs/$root"
  rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
    --include='*/' \
    --include='*.train.log' \
    --include='slurm-*.out' \
    --exclude='*' \
    "gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/$root/" \
    "Target_Design/gemnet_runs/trillium/raw_logs/$root/"
done
```

## Model Families Covered

The sync list includes all Trillium GemNet targets we used or queued during the project:

- `totE`: GemNet directly on reference-corrected total energy.
- `BoE_residuals`: GemNet on residual after BoE baseline.
- `xTB_residuals`: GemNet on residual after xTB baseline.
- `BoB_residuals`: GemNet on residual after BoB baseline.
- `PRE_residuals`: GemNet on residual after PRE fitted to total energy.
- `BoB_BoE_residuals`: GemNet on residual after BoB fitted to BoE residual.
- `BoE_xTB_residuals`: GemNet on residual after BoE plus xTB-style correction.
- `PRE_BoE_residuals`: GemNet on residual after PRE fitted to BoE residual.
- `PRE6plus_residuals`: GemNet on residual after PRE6+ fitted to total energy.
- `PRE6plus_BoE_residuals`: GemNet on residual after PRE6+ fitted to BoE residual.

## Notes

- As of this reorganization, raw Trillium logs were not found under `trillium_project`; only notebooks and derived analysis artifacts were present locally.
- The old training-curve notebook was moved from `trillium_project/gemnet_training_curves.ipynb` to `Target_Design/notebooks/gemnet_training_curves.ipynb`.
- Do not store checkpoints here unless specifically needed; this area is for lightweight logs and parsed curves.

## Residual-Target Summary Table

The PRE-proxy table uses full-dataset residual fit MAE. GemNet does not have the
same quantity in the current artifacts. The corresponding GemNet table should
therefore use parsed validation MAE from training logs:

```text
Target_Design/gemnet_runs/plots/gemnet_best_val_mae_epoch5plus.csv
```

Helper:

```text
Target_Design/gemnet_runs/plots/summarize_gemnet_residual_target_table.py
```

Output:

```text
Target_Design/gemnet_runs/plots/gemnet_residual_target_mae_table.csv
```

Current table, sorted from highest to lowest completed GemNet best validation
MAE:

```text
target residual                    status       GemNet best val MAE
y - b_long                         done         0.0574
y - b_comp,long                    done         0.0234
y - b_short                        done         0.0149
full target y                      done         0.0136
y - b_comp,short                   done         0.0123
y - b_comp,long|comp,short         done         0.0107
y - b_comp                         done         0.0101
y - b_comp,short,long              not_run
```

Interpretation:

```text
GemNet strongly confirms the target-design concern: long-range residualization
directly fitted to y is very poor, and naive b_comp,long residualization is also
much worse than the BoE residual. Conditioning the long-range PRE6+ component
on composition plus the short PRE component recovers most of the lost
learnability: y - b_comp,long|comp,short reaches 0.0107 best validation MAE,
close to the BoE residual at 0.0101 and much better than naive b_comp,long at
0.0234. The BoE residual remains the best completed single-seed statistical
target so far.
```

Update on 2026-05-11: the Clean Energy
`b_comp_long_conditioned_residuals_seed_101.nohup.log` file was downloaded and
parsed. This is the GemNet run on `y - b_comp,long|comp,short`, the hybrid
target `r_BoE - b_long|comp,short`.

Update on 2026-05-09: the Clean Energy
`PRE6plus_residuals_seed_101.train.log` file was downloaded and parsed. This is
the GemNet run on `y - b_long`, where `b_long` is PRE6+ fitted to `y`.
