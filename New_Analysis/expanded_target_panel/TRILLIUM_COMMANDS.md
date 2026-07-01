# Trillium Commands for Expanded Target Panel

These commands assume the local repo is:

```text
/Users/kareem/PycharmProjects/deltaML_Paper
```

and the Trillium repo is:

```text
/scratch/gameelka/deltaML_project
```

## 1. Sync Local Scripts and Target CSVs to Trillium

Run locally on the Mac:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/expanded_target_panel/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/expanded_target_panel/

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/datasets/tmQM__BoB-ref-BoE.csv \
  New_Analysis/datasets/tmQM__PRE-ref-BoE.csv \
  New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv \
  New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/datasets/
```

## 2. Verify on Trillium

Run on a Trillium debug node or login node:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" - <<'PY'
import pandas as pd
p = "New_Analysis/expanded_target_panel/target_registry.csv"
df = pd.read_csv(p)
print(df[["target_id", "target_label", "exists"]].to_string(index=False))
assert len(df) == 9
PY
```

If `exists` is stale because paths were checked on the Mac, rerun:

```bash
"$PY" New_Analysis/expanded_target_panel/prepare_expanded_targets.py --require-pre-plus-a
```

## 3. PRE Ridge for Missing Expanded Targets

Run on a Trillium debug node if walltime is enough; otherwise submit through the appropriate compute flow.

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

time "$PY" New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre \
  --targets bob,pre,pre_plus_a,deltapre_plus_a \
  --alphas=100,335.9818286283781,1000,3000 \
  --n-jobs 4 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
```

Expected output:

```text
New_Analysis/expanded_target_panel/outputs/ridge/pre_target_metrics.csv
New_Analysis/expanded_target_panel/outputs/ridge/pre_alpha_grid_results.json
New_Analysis/expanded_target_panel/outputs/ridge/pre_predictions_*.csv
```

## 4. PRE+A Ridge for All Nine Targets

Use the narrower high-alpha grid because the DFT_ref,BoE PRE+A check selected `alpha=3000`, and this is mainly to extend the same model class across targets.

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

time "$PY" New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre_plus_a \
  --targets all \
  --alphas=1000,3000,10000 \
  --n-jobs 3 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
```

Expected output:

```text
New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_target_metrics.csv
New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_alpha_grid_results.json
New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_predictions_*.csv
```

## 5. Download Completed Ridge Outputs

Run locally on the Mac:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
mkdir -p New_Analysis/expanded_target_panel/outputs/ridge

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/expanded_target_panel/outputs/ridge/ \
  New_Analysis/expanded_target_panel/outputs/ridge/
```

## 6. Later GNN Training

After statistical target metrics are in place, prepare/train these additional GNN targets:

```text
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
```

Use the existing training protocols:

```text
GemNet: 24 h
SchNet: 2 h
```

Do not add QM9 or stacking logic to this step. This panel is for in-domain target learnability first.
