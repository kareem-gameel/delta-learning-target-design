# GemNet Smoothness Correlation Grid

This is an exploratory sensitivity analysis over available local smoothness outputs.

Fixed choices in this scan:
- split: tmQM validation
- kNN graph: k=10
- kNN distance: Euclidean
- feature scaling: raw or standardized feature columns, using the precomputed smoothness files
- target sets: manuscript MLP-residual five-target set, expanded linear nine-target set, and delta-only subsets

The main ranking uses `positive_min_r_rho = min(Pearson r, Spearman rho)`, because the hypothesis is that rougher targets should have larger normalized GemNet error.

## Best Positive Agreements

| target_set    | feature_space   | feature_scaling   | x_metric                         | y_metric          |   pearson_r |   spearman_rho |   positive_min_r_rho |
|:--------------|:----------------|:------------------|:---------------------------------|:------------------|------------:|---------------:|---------------------:|
| delta_mlp4    | PRE             | std               | sum_normalized_dirichlet         | mae_over_variance |       0.958 |              1 |                0.958 |
| delta_mlp4    | PRE             | std               | variance_normalized_dirichlet    | mae_over_variance |       0.958 |              1 |                0.958 |
| delta_mlp4    | PRE             | std               | random_pair_normalized_dirichlet | mae_over_variance |       0.958 |              1 |                0.958 |
| delta_mlp4    | PRE+A           | std               | sum_normalized_dirichlet         | mae_over_std      |       0.957 |              1 |                0.957 |
| delta_mlp4    | PRE+A           | std               | variance_normalized_dirichlet    | mae_over_std      |       0.957 |              1 |                0.957 |
| delta_mlp4    | PRE+A           | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.957 |              1 |                0.957 |
| delta_mlp4    | PRE+A           | std               | variance_normalized_dirichlet    | mae_over_variance |       0.953 |              1 |                0.953 |
| delta_mlp4    | PRE+A           | std               | random_pair_normalized_dirichlet | mae_over_variance |       0.953 |              1 |                0.953 |
| delta_mlp4    | PRE+A           | std               | sum_normalized_dirichlet         | mae_over_variance |       0.953 |              1 |                0.953 |
| delta_linear4 | PRE             | raw               | variance_normalized_dirichlet    | mae               |       0.938 |              1 |                0.938 |
| delta_linear4 | PRE             | raw               | random_pair_normalized_dirichlet | mae               |       0.938 |              1 |                0.938 |
| delta_linear4 | PRE             | raw               | sum_normalized_dirichlet         | mae               |       0.938 |              1 |                0.938 |
| delta_mlp4    | PRE             | std               | sum_normalized_dirichlet         | mae_over_std      |       0.936 |              1 |                0.936 |
| delta_mlp4    | PRE             | std               | variance_normalized_dirichlet    | mae_over_std      |       0.936 |              1 |                0.936 |
| delta_mlp4    | PRE             | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.936 |              1 |                0.936 |

## Best Positive Agreements for Manuscript MLP Five-Target Set

| target_set      | feature_space   | feature_scaling   | x_metric                         | y_metric          |   pearson_r |   spearman_rho |   positive_min_r_rho |
|:----------------|:----------------|:------------------|:---------------------------------|:------------------|------------:|---------------:|---------------------:|
| manuscript_mlp5 | PRE             | std               | edge_rmse_over_iqr               | mae_over_iqr      |       0.932 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | std               | edge_rmse_over_iqr               | mae_over_iqr      |       0.929 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | sum_normalized_dirichlet         | mae_over_variance |       0.923 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | variance_normalized_dirichlet    | mae_over_variance |       0.923 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | random_pair_normalized_dirichlet | mae_over_variance |       0.923 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | raw               | variance_normalized_dirichlet    | mae_over_iqr      |       0.919 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | raw               | random_pair_normalized_dirichlet | mae_over_iqr      |       0.919 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | raw               | sum_normalized_dirichlet         | mae_over_iqr      |       0.919 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | sum_normalized_dirichlet         | mae_over_std      |       0.908 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | variance_normalized_dirichlet    | mae_over_std      |       0.908 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.908 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE             | raw               | edge_rmse_over_iqr               | mae_over_iqr      |       0.907 |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | std               | variance_normalized_dirichlet    | mae_over_std      |       0.9   |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.9   |            0.9 |                  0.9 |
| manuscript_mlp5 | PRE+A           | std               | sum_normalized_dirichlet         | mae_over_std      |       0.9   |            0.9 |                  0.9 |

## Best Absolute Agreements

| target_set    | feature_space   | feature_scaling   | x_metric                         | y_metric          |   pearson_r |   spearman_rho |   abs_min_r_rho |
|:--------------|:----------------|:------------------|:---------------------------------|:------------------|------------:|---------------:|----------------:|
| delta_mlp4    | PRE             | std               | sum_normalized_dirichlet         | mae_over_variance |       0.958 |              1 |           0.958 |
| delta_mlp4    | PRE             | std               | variance_normalized_dirichlet    | mae_over_variance |       0.958 |              1 |           0.958 |
| delta_mlp4    | PRE             | std               | random_pair_normalized_dirichlet | mae_over_variance |       0.958 |              1 |           0.958 |
| delta_mlp4    | PRE+A           | std               | sum_normalized_dirichlet         | mae_over_std      |       0.957 |              1 |           0.957 |
| delta_mlp4    | PRE+A           | std               | variance_normalized_dirichlet    | mae_over_std      |       0.957 |              1 |           0.957 |
| delta_mlp4    | PRE+A           | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.957 |              1 |           0.957 |
| delta_mlp4    | PRE+A           | std               | variance_normalized_dirichlet    | mae_over_variance |       0.953 |              1 |           0.953 |
| delta_mlp4    | PRE+A           | std               | random_pair_normalized_dirichlet | mae_over_variance |       0.953 |              1 |           0.953 |
| delta_mlp4    | PRE+A           | std               | sum_normalized_dirichlet         | mae_over_variance |       0.953 |              1 |           0.953 |
| delta_linear4 | PRE             | raw               | variance_normalized_dirichlet    | mae               |       0.938 |              1 |           0.938 |
| delta_linear4 | PRE             | raw               | random_pair_normalized_dirichlet | mae               |       0.938 |              1 |           0.938 |
| delta_linear4 | PRE             | raw               | sum_normalized_dirichlet         | mae               |       0.938 |              1 |           0.938 |
| delta_mlp4    | PRE             | std               | sum_normalized_dirichlet         | mae_over_std      |       0.936 |              1 |           0.936 |
| delta_mlp4    | PRE             | std               | variance_normalized_dirichlet    | mae_over_std      |       0.936 |              1 |           0.936 |
| delta_mlp4    | PRE             | std               | random_pair_normalized_dirichlet | mae_over_std      |       0.936 |              1 |           0.936 |

## Interpretation Guardrail

A perfect or near-perfect score in this grid should be treated as descriptive, not confirmatory. With 4-5 targets, many normalization choices can produce high rank agreement by construction. The defensible result is the stability of a relationship across related choices, not a single best row.
