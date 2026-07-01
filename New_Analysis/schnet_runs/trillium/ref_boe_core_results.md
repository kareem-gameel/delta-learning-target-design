# SchNet Ref,BoE Core Results

Date: 2026-06-18

These results use SchNet trained with the same tmQM split protocol as the
GemNet runs, but with the model block replaced by FairChem `schnet`.

## Training Runs

Trillium run root:

```text
/scratch/gameelka/schnet_ref_boe_core_2026_06
```

Configs:

```text
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_DFT_ref_BoE_seed_101.yml
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_xTB_ref_BoE_seed_101.yml
/scratch/gameelka/schnet_ref_boe_core_2026_06/configs/schnet_Delta_xTB_ref_BoE_seed_101.yml
```

Slurm jobs:

```text
605876 -> SchNet(DFT_ref,BoE)
605877 -> SchNet(xTB_ref,BoE)
605878 -> SchNet(Delta^xTB_ref,BoE)
```

Approximate runtime:

```text
SchNet(DFT_ref,BoE):          01:13:59
SchNet(xTB_ref,BoE):          01:13:57
SchNet(Delta^xTB_ref,BoE):    ~01:12:07
```

## tmQM Validation MAE

Parsed from the SchNet training logs:

```text
model                         n_val_evals  best_epoch  best_step  best_val_mae  last_epoch  last_step  last_val_mae
SchNet(DFT_ref,BoE)           56           45.0450     160000     0.0153        78.8288     280000     0.0163
SchNet(xTB_ref,BoE)           56           54.8986     195000     0.0138        78.8288     280000     0.0140
SchNet(Delta^xTB_ref,BoE)     56           46.4527     165000     0.0093        78.8288     280000     0.0101
```

Interpretation:

```text
SchNet preserves the GemNet target ranking:

Delta^xTB_ref,BoE  <  xTB_ref,BoE  <  DFT_ref,BoE
```

SchNet is worse than GemNet in absolute MAE, but the target-design effect is
not GemNet-specific within this local GNN comparison.

## QM9 Transfer Against DFT_ref,BoE

Predictions:

```text
/scratch/gameelka/schnet_ref_boe_core_2026_06/eval/qm9_predict/schnet_DFT_ref_BoE_seed_101/results/schnet_DFT_ref_BoE_seed_101_qm9/is2re_predictions.npz
/scratch/gameelka/schnet_ref_boe_core_2026_06/eval/qm9_predict/schnet_xTB_ref_BoE_seed_101/results/schnet_xTB_ref_BoE_seed_101_qm9/is2re_predictions.npz
/scratch/gameelka/schnet_ref_boe_core_2026_06/eval/qm9_predict/schnet_Delta_xTB_ref_BoE_seed_101/results/schnet_Delta_xTB_ref_BoE_seed_101_qm9/is2re_predictions.npz
```

Corrected scoring must align predictions to QM9 by `mol_id` from the LMDB, not
by CSV row order. The valid aligned output was written on Trillium to:

```text
/scratch/gameelka/schnet_ref_boe_core_2026_06/eval/qm9_metrics_mol_id_aligned
```

Mol-id aligned QM9 metrics:

```text
model                                      n       raw_mae   raw_rmse  raw_r2    raw_tau  shift       shift_mae  shift_rmse  shift_r2   shift_tau  bias
xTB_ref,BoE + SchNet(Delta^xTB_ref,BoE)   130807  0.026205  0.028016  0.704232  0.896732 -0.026202   0.007030   0.009918    0.962936   0.896732   0.026202
SchNet(DFT_ref,BoE)                       130807  0.252614  0.253325 -23.181829 0.778988  0.251940   0.013792   0.026453    0.736309   0.778988  -0.251940
xTB_ref,BoE baseline                      130807  0.026683  0.033322  0.581589  0.681503 -0.024370   0.018055   0.022726    0.805388   0.681503   0.024370
SchNet(xTB_ref,BoE)                       130807  0.252072  0.253785 -23.269710 0.596772  0.251315   0.022492   0.035320    0.529921   0.596772  -0.251315
```

Main interpretation:

```text
xTB_ref,BoE + SchNet(Delta^xTB_ref,BoE)
beats direct SchNet(DFT_ref,BoE) and the standalone xTB_ref,BoE baseline on QM9.
```

Thus, for SchNet as well as GemNet, the xTB decomposition is not only baseline
heavy lifting. The learned residual transfers meaningful structure.

## GemNet xTB QM9 Availability Note

We have QM9 predictions for:

```text
GemNet(xTB_ref,BoE)
```

through the existing `New_Analysis/qm9_dft_ref_boe_transfer` analysis and the
prediction file:

```text
trillium_project/qm9_predictions/qm9_pred_boe_xtb_job309204.npz
```

Those metrics against `DFT_ref,BoE` are:

```text
GemNet(xTB_ref,BoE), mean-shifted:
MAE = 0.028650 Ha
RMSE = 0.042709 Ha
R2 = 0.312672
Kendall tau = 0.481901
```

What is not established here is a separate QM9 prediction for:

```text
GemNet(xTB_ref)
```

without BoE removal, or for raw unreferenced `xTB`.

## Insight: xTB Target Versus xTB Correction

Across both GemNet and SchNet, `xTB_ref,BoE` behaves differently as a direct
learning target than as a baseline used to define a correction target.

In-domain tmQM validation:

```text
GemNet(xTB_ref,BoE)  = 0.0086 Ha
GemNet(DFT_ref,BoE)  = 0.0101 Ha

SchNet(xTB_ref,BoE)  = 0.0138 Ha
SchNet(DFT_ref,BoE)  = 0.0153 Ha
```

So `xTB_ref,BoE` is easier to learn in-domain than `DFT_ref,BoE` for both
GemNet and SchNet.

OOD QM9 transfer against `DFT_ref,BoE`, using mean-shifted MAE:

```text
GemNet(xTB_ref,BoE)  = 0.028650 Ha
GemNet(DFT_ref,BoE)  = 0.015567 Ha

SchNet(xTB_ref,BoE)  = 0.022492 Ha
SchNet(DFT_ref,BoE)  = 0.013792 Ha
```

So a GNN trained to imitate `xTB_ref,BoE` transfers worse than a GNN trained
directly on `DFT_ref,BoE`.

The transferable object is instead the xTB-to-DFT correction field:

```text
Delta^xTB_ref,BoE = DFT_ref,BoE - xTB_ref,BoE
```

When this learned correction is added back to the actual QM9 `xTB_ref,BoE`
baseline, it gives the best OOD performance:

```text
xTB_ref,BoE + GemNet(Delta^xTB_ref,BoE):  shifted MAE = 0.006154 Ha, tau = 0.899092
xTB_ref,BoE + SchNet(Delta^xTB_ref,BoE):  shifted MAE = 0.007030 Ha, tau = 0.896732
```

This separates three claims:

```text
1. xTB_ref,BoE is an easier in-domain target than DFT_ref,BoE.
2. A learned xTB_ref,BoE surrogate is not a good OOD DFT_ref,BoE predictor.
3. The xTB-to-DFT residual correction is highly transferable when added to the
   real xTB_ref,BoE baseline.
```
