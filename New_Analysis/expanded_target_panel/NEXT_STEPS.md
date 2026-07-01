# Expanded Target Panel: Next Steps

Last updated: 2026-06-19.

## Current State

The expanded target panel now has nine `ref,BoE` targets:

| target_id | target |
|---|---|
| `dft` | `DFT_ref,BoE` |
| `xtb` | `xTB_ref,BoE` |
| `deltaxtb` | `Delta^xTB_ref,BoE` |
| `deltabob` | `Delta^BoB_ref,BoE` |
| `deltapre` | `Delta^PRE_ref,BoE` |
| `bob` | `BoB_ref,BoE` |
| `pre` | `PRE_ref,BoE` |
| `pre_plus_a` | `PRE+A_ref,BoE` |
| `deltapre_plus_a` | `Delta^PRE+A_ref,BoE` |

Current model-target completion:

| model | complete | missing |
|---|---:|---:|
| BoB ridge | 9 | 0 |
| PRE ridge | 9 | 0 |
| PRE+A ridge | 9 | 0 |
| GemNet | 5 | 4 |
| SchNet | 5 | 4 |

The only missing rows are GNN trainings on the four added targets:

| model | missing target |
|---|---|
| GemNet | `BoB_ref,BoE` |
| GemNet | `PRE_ref,BoE` |
| GemNet | `PRE+A_ref,BoE` |
| GemNet | `Delta^PRE+A_ref,BoE` |
| SchNet | `BoB_ref,BoE` |
| SchNet | `PRE_ref,BoE` |
| SchNet | `PRE+A_ref,BoE` |
| SchNet | `Delta^PRE+A_ref,BoE` |

## Files Already Generated

Status:

```text
New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv
New_Analysis/expanded_target_panel/outputs/status/model_target_completion_grid.csv
New_Analysis/expanded_target_panel/outputs/status/model_target_completion_summary.md
```

Ridge metrics:

```text
New_Analysis/expanded_target_panel/outputs/ridge/bob_target_metrics.csv
New_Analysis/expanded_target_panel/outputs/ridge/pre_target_metrics.csv
New_Analysis/expanded_target_panel/outputs/ridge/pre_plus_a_target_metrics.csv
```

Smoothness:

```text
New_Analysis/expanded_target_panel/outputs/smoothness/bob_space_dirichlet_val.csv
New_Analysis/expanded_target_panel/outputs/smoothness/pre_space_dirichlet_val.csv
New_Analysis/expanded_target_panel/outputs/smoothness/pre_plus_a_space_dirichlet_val.csv
```

Current available plots:

```text
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_model_target_val_mae.png
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_model_target_val_nmae_iqr.png
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_smoothness_vs_nmae_available_std.png
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_smoothness_correlations_available.csv
```

## Immediate Local Step

Regenerate current status and plots any time after new metrics are added:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/build_panel_status.py

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/plot_expanded_panel.py
```

## Next Trillium Step: SchNet

SchNet is the faster GNN check, so it should be trained before GemNet on the four added targets.

Expected walltime:

```text
~2 hours per target
```

Targets:

```text
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

Recommended purpose:

1. Check whether the expanded-target trends are learner-specific or shared by another local geometric GNN.
2. Decide whether all four expanded targets are worth the more expensive GemNet training.
3. Use SchNet as a quick filter before committing 24 h GemNet jobs.

## Next Trillium Step: GemNet

GemNet is the main model, but it is expensive.

Expected walltime:

```text
~24 hours per target
```

Targets:

```text
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

Recommended priority:

1. `Delta^PRE+A_ref,BoE`
2. `PRE+A_ref,BoE`
3. `PRE_ref,BoE`
4. `BoB_ref,BoE`

Rationale:

- `Delta^PRE+A_ref,BoE` is the most direct test of whether the angular-augmented baseline gives a better residual target.
- `PRE+A_ref,BoE` tests whether the target produced by the new baseline is itself aligned with the GNN.
- `PRE_ref,BoE` and `BoB_ref,BoE` are useful controls, but they are less central.

## Interpretation Check After SchNet/GemNet Finish

After adding GNN rows, rerun:

```bash
New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/build_panel_status.py

New_Analysis/local_analysis_python.sh \
  New_Analysis/expanded_target_panel/plot_expanded_panel.py
```

Then inspect:

```text
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_model_target_val_nmae_iqr.png
New_Analysis/expanded_target_panel/outputs/final_plots/expanded_smoothness_correlations_available.csv
```

Key questions:

1. Does `Delta^PRE+A_ref,BoE` improve over `Delta^PRE_ref,BoE` for SchNet/GemNet?
2. Does PRE+A-space smoothness correlate better with GemNet normalized MAE than PRE-space smoothness?
3. Does SchNet follow the same smoothness/learnability ordering as GemNet?
4. Are baseline-target rows (`BoB_ref,BoE`, `PRE_ref,BoE`, `PRE+A_ref,BoE`) useful controls, or do they distract from the residual-target story?

## Decision Gates

Train GemNet on all four expanded targets only if SchNet suggests the added targets are informative.

Minimum useful outcome:

```text
SchNet confirms that Delta^PRE+A_ref,BoE behaves differently from Delta^PRE_ref,BoE.
```

Strong outcome:

```text
PRE+A-space smoothness predicts GemNet/SchNet normalized MAE better than PRE-space smoothness,
especially for the residual targets.
```

Negative but useful outcome:

```text
PRE+A improves the ridge baseline but does not produce a better GNN residual target.
```

That would support the paper’s larger claim: a better baseline fit is not automatically a better delta-learning target.
