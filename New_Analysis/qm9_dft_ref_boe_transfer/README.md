# QM9 Transfer To DFT_ref,BoE

This folder evaluates QM9 predictions against the target:

```text
DFT_ref,BoE = DFT_ref - BoE_fit(DFT_ref)
```

Models compared:

```text
GemNet(DFT_ref,BoE)
GemNet(xTB_ref,BoE)
xTB_ref,BoE + GemNet(Delta^xTB_ref,BoE)
```

For `GemNet(Delta^xTB_ref,BoE)`, the evaluator adds the QM9
`xTB_ref,BoE` baseline before comparing to `DFT_ref,BoE`.

## Upload Bundle

Run from the Mac because Trillium sync requires MFA:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -avR --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/qm9_dft_ref_boe_transfer/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/
```

If validation reports missing old QM9 prediction NPZs, upload these too:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -avR --progress -e "ssh -i ~/.ssh/myKey" \
  trillium_project/qm9_predictions/qm9_pred_boe_job302891.npz \
  trillium_project/qm9_predictions/qm9_pred_boe_xtb_job309204.npz \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/
```

## Validate Inputs On Trillium

```bash
cd /scratch/gameelka/deltaML_project

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH

PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"
"$PY" -c "import lmdb, numpy, pandas, scipy, torch, yaml; print('env ok')"
"$PY" New_Analysis/qm9_dft_ref_boe_transfer/run_qm9_dft_ref_boe_transfer.py --validate-only
```

## Prepare Delta^xTB_ref,BoE QM9 Prediction Config

Use the best checkpoint from the current Trillium run:

```bash
cd /scratch/gameelka/deltaML_project

PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" New_Analysis/qm9_dft_ref_boe_transfer/make_trillium_qm9_predict_config.py \
  --run-root /scratch/gameelka/delta_xtb_ref_boe_reproducibility_2026_05 \
  --run-name delta_xtb_ref_boe_seed_101 \
  --qm9-lmdb /scratch/gameelka/deltaML_project/lmdb_files/qm9_psi4_totE.lmdb
```

Expected config:

```text
/scratch/gameelka/delta_xtb_ref_boe_reproducibility_2026_05/eval/configs/delta_xtb_ref_boe_seed_101_qm9_predict.yml
```

## Run QM9 Inference On A Live Debug Node

After entering a debug node:

```bash
cd /scratch/gameelka/deltaML_project

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH

PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"
RUN_ROOT=/scratch/gameelka/delta_xtb_ref_boe_reproducibility_2026_05
RUN_NAME=delta_xtb_ref_boe_seed_101
CFG="$RUN_ROOT/eval/configs/${RUN_NAME}_qm9_predict.yml"
CKPT="$RUN_ROOT/runs/seed_101/checkpoints/$RUN_NAME/best_checkpoint.pt"
OUT_DIR="$RUN_ROOT/eval/qm9_predict"

mkdir -p "$OUT_DIR"
cd /scratch/gameelka/deltaML_project/train

"$PY" -u main.py \
  --mode predict \
  --config-yml "$CFG" \
  --checkpoint "$CKPT" \
  --identifier "${RUN_NAME}_qm9" \
  --run-dir "$OUT_DIR" \
  --timestamp-id "${RUN_NAME}_qm9" \
  2>&1 | tee "$OUT_DIR/predict.log"
```

Copy the prediction NPZ into the analysis location:

```bash
cd /scratch/gameelka/deltaML_project

mkdir -p New_Analysis/qm9_dft_ref_boe_transfer/predictions
cp /scratch/gameelka/delta_xtb_ref_boe_reproducibility_2026_05/eval/qm9_predict/results/delta_xtb_ref_boe_seed_101_qm9/is2re_predictions.npz \
  New_Analysis/qm9_dft_ref_boe_transfer/predictions/qm9_pred_delta_xtb_ref_boe_seed_101.npz
```

## Evaluate

```bash
cd /scratch/gameelka/deltaML_project

PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"
"$PY" New_Analysis/qm9_dft_ref_boe_transfer/run_qm9_dft_ref_boe_transfer.py
```

Outputs:

```text
New_Analysis/qm9_dft_ref_boe_transfer/outputs/manifest_validation.csv
New_Analysis/qm9_dft_ref_boe_transfer/outputs/qm9_DFT_ref_BoE_gemnet_predictions.csv
New_Analysis/qm9_dft_ref_boe_transfer/outputs/qm9_DFT_ref_BoE_gemnet_metrics.csv
```
