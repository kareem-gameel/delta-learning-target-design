#!/bin/bash
set -euo pipefail

cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" - <<'PY'
import pandas as pd
df = pd.read_csv("New_Analysis/expanded_target_panel/target_registry.csv")
print(df[["target_id", "target_label", "exists"]].to_string(index=False))
assert len(df) == 9
PY

time "$PY" New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py \
  --feature-space pre_plus_a \
  --targets all \
  --pre-cache /scratch/gameelka/deltaML_project/pre_trillium_cpu_jobs/cache \
  --angular-cache New_Analysis/features/PRE_A_debug_full \
  --alphas=1000,3000,10000 \
  --n-jobs 3 \
  --out-dir New_Analysis/expanded_target_panel/outputs/ridge
