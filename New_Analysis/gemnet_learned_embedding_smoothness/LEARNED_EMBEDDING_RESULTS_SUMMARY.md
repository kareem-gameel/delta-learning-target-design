# Learned GemNet Embedding Smoothness Results

Date: 2026-06-29

## Purpose

This analysis tests a post-training diagnostic: after training GemNet on a target, extract the model's learned atom embeddings on the tmQM held-out split, pool them into molecular embeddings, construct a kNN graph in that learned molecular representation, and measure whether the trained target is smooth in its own learned embedding space.

This is different from the PRE/PRE+A smoothness diagnostics. PRE/PRE+A smoothness is a pre-training representation-based hypothesis test. Learned-embedding smoothness is a post-training diagnostic of the representation GemNet actually formed.

## Source Files

Main local outputs:

```text
New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/manifest_exports/learned_embedding_roughness_mean_std_all_targets_corrected_shared.csv
New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/manifest_exports/learned_embedding_self_target_roughness_vs_gemnet_val_mae_corrected_shared.csv
New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/manifest_exports/gemnet_embedding_val_metrics_manifest_corrected.csv
New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/manifest_exports/shared_embedding_mol_ids_corrected.csv
```

Heavy parquet atom/molecule embedding files were intentionally kept on Trillium and not retained locally.

## Workflow

Atom-level GemNet embeddings were exported from trained GemNet checkpoints using the clean FairChem export wrapper. The exported atom embedding is a 512-dimensional hidden state per atom.

Molecular embeddings were formed by `mean_std` pooling:

```text
h_mol = concat(mean_atoms(h_atom), std_atoms(h_atom))
```

This gives a 1024-dimensional molecular vector per molecule.

Euclidean kNN graphs were constructed in learned molecular embedding space with:

```text
k = 10
feature scaling = raw and z-scored/std
```

The final corrected analysis used the shared intersection of molecules across all exported embedding spaces:

```text
n_molecules = 6126
embedding spaces = 13
targets = 13
self-target rows = 26 = 13 models x 2 feature scalings
```

The xTB labeling was corrected:

```text
gemnet_xtb_ref_boe        = GemNet(xTB_ref,BoE)
gemnet_delta_xtb_ref_boe  = GemNet(Delta^xTB_ref,BoE)
```

## Metrics

For graph edges `E`, target values `y`, and `N` molecules:

```text
D_edge = (1 / |E|) sum_(i,j in E) (y_i - y_j)^2
edge_rmse = sqrt(D_edge)
edge_avg_norm_by_population_var = D_edge / Var(y)
edge_rmse_over_iqr = sqrt(D_edge) / IQR(y)
```

The table was merged against the best validation MAE for each trained GemNet target. The column is currently named `test_mae` in the CSV for compatibility with the merge script, but the metric source is `best_val_mae`.

## Correlations With GemNet MAE

Across the 13 self-target rows:

| feature scaling | roughness metric | Pearson vs MAE | Spearman vs MAE |
|---|---:|---:|---:|
| std | `edge_avg_norm_by_population_var` | 0.168 | 0.082 |
| std | `edge_rmse` | -0.063 | -0.099 |
| std | `edge_rmse_over_iqr` | 0.688 | 0.659 |
| raw | `edge_avg_norm_by_population_var` | -0.069 | -0.258 |
| raw | `edge_rmse` | -0.062 | -0.082 |
| raw | `edge_rmse_over_iqr` | 0.693 | 0.648 |

## Main Result

Learned GemNet embedding smoothness does not produce a strong monotonic relationship between variance-normalized Dirichlet roughness and GemNet MAE.

The variance-normalized metric:

```text
edge_avg_norm_by_population_var = D_edge / Var(y)
```

has weak correlation with GemNet MAE in learned embedding space:

```text
std scaling: Spearman = 0.082
raw scaling: Spearman = -0.258
```

The absolute neighbor jump:

```text
edge_rmse = sqrt(D_edge)
```

also does not correlate with MAE, because it is dominated by target scale and can make small residual targets look trivially smooth.

The robust relative neighbor jump:

```text
edge_rmse_over_iqr = sqrt(D_edge) / IQR(y)
```

is the most informative learned-embedding metric:

```text
std scaling: Pearson = 0.688, Spearman = 0.659
raw scaling: Pearson = 0.693, Spearman = 0.648
```

This supports the broader conclusion that the useful diagnostic is not smoothness alone or scale alone, but the size of local target jumps relative to robust target scale.

## Self-Target Highlights

With standardized learned embedding coordinates:

| model target | GemNet MAE | normalized Dirichlet | edge RMSE | edge RMSE / IQR |
|---|---:|---:|---:|---:|
| `Delta^xTB_ref,BoE` | 0.0064 | 1.693 | 0.0668 | 1.293 |
| `xTB_ref,BoE` | 0.0086 | 1.618 | 0.1113 | 1.290 |
| `DFT_ref,BoE` | 0.0103 | 1.602 | 0.1110 | 1.320 |
| `Delta^BoB_ref,BoE` | 0.0108 | 1.574 | 0.0586 | 1.455 |
| `BoB_ref,BoE` | 0.0113 | 0.805 | 0.6564 | 0.710 |
| `Delta^PRE_ref,BoE` | 0.0123 | 1.815 | 0.0368 | 1.718 |
| `PRE_ref,BoE` | 0.0126 | 0.619 | 0.5761 | 0.624 |
| `DFT_ref` | 0.0139 | 0.756 | 0.6366 | 0.689 |
| `PRE+A_ref,BoE` | 0.0142 | 1.723 | 0.1439 | 1.820 |
| `Delta^PRE+A_ref,BoE` | 0.0165 | 1.579 | 0.1026 | 4.416 |
| `Delta^MLP_PRE+A_ref,BoE` | 0.0168 | 1.645 | 0.0824 | 4.675 |
| `Delta^MLP_BoB_ref,BoE` | 0.0174 | 1.402 | 0.0494 | 1.930 |
| `Delta^MLP_PRE_ref,BoE` | 0.0181 | 2.363 | 0.0980 | 4.823 |

## Interpretation

The learned embedding analysis reinforces the target-design message rather than replacing it with a simple smoothness rule.

`Delta^xTB_ref,BoE` remains the best GemNet target by MAE, but it is not the lowest target by variance-normalized learned-space roughness. Conversely, `PRE_ref,BoE`, `BoB_ref,BoE`, and `DFT_ref` are relatively smooth by variance-normalized roughness in their learned spaces, yet they are not the best targets. This means learned-space normalized Dirichlet energy alone is not a sufficient predictor of GemNet performance.

The MLP residuals are especially informative. Their absolute edge RMSE values can be small, because the residuals are target-shrunk, but their GemNet MAEs remain poor. This supports the conclusion that target shrinkage by itself is not enough. A residual must leave a target that is locally structured in a way GemNet can learn.

The most stable learned-embedding diagnostic is `edge_rmse_over_iqr`. It captures local neighbor jumps relative to robust target scale and gives the strongest relationship with GemNet MAE. This is consistent with the paper's emerging thesis:

```text
Delta learning is target design, not merely target shrinkage.
Good residual targets should be small enough and locally structured enough in the learner's representation.
```

## Recommended Use In The Paper

Use this as a supporting diagnostic, not the central predictor.

The paper should emphasize:

1. Learned GemNet embeddings do not make target smoothness universal or trivially predictive.
2. Variance-normalized Dirichlet roughness is useful but incomplete.
3. Robust relative local jumps, `edge_rmse_over_iqr`, are more predictive of GemNet MAE.
4. MLP residuals demonstrate that lower target scale does not guarantee better downstream GemNet learning.
5. The strongest conclusion remains target design: residual quality depends on the interaction between scale, local structure, and model inductive bias.

