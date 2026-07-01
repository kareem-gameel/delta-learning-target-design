# xTB Ref-Corrected GemNet Run Notes

## 2026-05-20

Planned new diagnostic:

```text
GemNet(xTB_refcorr)
```

Target source:

```text
Original_Data/ref-corrected_data/tmqm_xtb_ref-corr_neutral.csv
target column: reference_corrected_energy
```

Confirmed local row counts:

```text
tmqm_xtb_ref-corr.csv          86,502 rows
tmqm_xtb_ref-corr_neutral.csv  71,038 rows
tmqm_dft_ref-corr.csv          86,507 rows
tmqm_dft_ref-corr_neutral.csv  71,042 rows
```

Rationale:

This is distinct from the existing `GemNet(y - xTB)` residual model. It tests
whether GemNet can learn the analytical xTB reference-corrected target itself
and provides a diagnostic for how much the xTB-like target induces global or
composition-like behavior in the trained GNN.

Execution plan:

1. Upload this bundle and `tmqm_xtb_ref-corr_neutral.csv` to Trillium.
2. Run `prepare_xtb_refcorr_trillium.py` on Trillium to rewrite the neutral
   tmQM train/val/test LMDBs with xTB reference-corrected labels.
3. Run `preflight_check.py`.
4. Smoke test on a live debug GPU node.
5. Submit the exact same Slurm script to the debug partition for 1 hour.
6. If clean, submit the exact same Slurm script to the compute partition for
   24 hours.

## 2026-05-21

Full Trillium compute run completed cleanly.

```text
job_id: 527898
job_name: xtb-refcorr
partition: compute
node: trig0037
submit_time: 2026-05-20 13:35:36
end_time: 2026-05-21 08:28:22
runtime: 18:52:45
exit_code: 0:0
max_epochs: 80
total_time_reported_by_fairchem: 67947.968 s
```

Artifacts on Trillium:

```text
/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/jobs/slurm-527898.out
/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/runs/seed_101/xtb_refcorr_seed_101.train.log
/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/runs/seed_101/checkpoints/xtb_refcorr_seed_101/best_checkpoint.pt
/scratch/gameelka/xtb_refcorr_reproducibility_2026_05/runs/seed_101/checkpoints/xtb_refcorr_seed_101/checkpoint.pt
```

Last logged training minibatches near epoch 80 had `energy_mae` values mostly
around `0.008-0.013 Ha`, but these are training minibatch values, not the held
out validation/test metrics. The next step is to parse validation metrics from
the log and run explicit prediction/evaluation on the tmQM test split.

Parsed validation metrics using the same convention as
`Target_Design/scripts/plot_gemnet_learning_curves.py`:

```text
GemNet(xTB_refcorr) best val MAE: 0.0121 Ha
best epoch: 74.9578
log line: 210362
```

Comparison to existing in-domain GemNet validation MAE table:

```text
xTB residual      GemNet(y - xTB)          0.0083
BoE+xTB residual  GemNet(y - b_comp,xTB)   0.0086
BoE residual      GemNet(y - b_comp)       0.0101
xTB refcorr       GemNet(xTB)              0.0121
totE              GemNet(y)                0.0136
```

Interpretation: GemNet can learn the xTB reference-corrected target better than
the DFT target itself, but not as well as the best residual targets. In this
comparison, learning `xTB` itself is easier than learning `totE`, while learning
the correction `y - xTB` is easier still. That supports the view that the xTB
residual target is unusually well aligned with the GemNet inductive bias, not
merely smaller in scale.
