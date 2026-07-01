# PRE+A Smoothness Summary

Date: 2026-06-18

## Scope

Feature-space Dirichlet smoothness was computed for the five `ref,BoE` targets:

```text
DFT_ref,BoE
xTB_ref,BoE
Delta^xTB_ref,BoE
Delta^BoB_ref,BoE
Delta^PRE_ref,BoE
```

The smoothness feature space was:

```text
PRE+A = [PRE typed pair-radial features | central-element angular histograms]
```

Validation split only:

```text
n_val = 7103 molecules
k = 10 nearest neighbors
distance = Euclidean
scalings = raw and feature-wise standardized
```

Main smoothness metric:

```text
edge_avg_norm_by_population_var
= mean_edges[(y_i - y_j)^2] / Var(y)
```

The comparison table was saved to:

```text
New_Analysis/pre_plus_a/outputs_smoothness/pre_vs_pre_plus_a_space_model_correlation_comparison.csv
```

## PRE+A Smoothness Ordering

Using standardized PRE+A features and `edge_avg_norm_by_population_var`, the
target roughness order was:

```text
Delta^xTB  <  xTB  <  DFT  <  Delta^BoB  <  Delta^PRE
```

This is consistent with the broader observation that `Delta^xTB_ref,BoE` is the
most locally smooth target in chemically meaningful local feature spaces.

## PRE Versus PRE+A Correlation With Model MAE/IQR

Pearson `r` between smoothness and model validation `MAE/IQR`:

```text
metric: edge_avg_norm_by_population_var

model   scaling   PRE      PRE+A
BoB     raw       0.888    0.832
BoB     std       0.915    0.904
GemNet  raw       0.988    0.991
GemNet  std       0.977    0.971
PRE     raw       0.996    0.986
PRE     std       0.993    0.987
SchNet  raw       0.995    0.986
SchNet  std       0.990    0.983
```

Using `edge_rmse_over_iqr`:

```text
model   scaling   PRE      PRE+A
BoB     raw       0.965    0.959
BoB     std       0.960    0.954
GemNet  raw       0.892    0.898
GemNet  std       0.924    0.917
PRE     raw       0.938    0.942
PRE     std       0.962    0.956
SchNet  raw       0.931    0.935
SchNet  std       0.955    0.948
```

## Interpretation

PRE+A is not dramatically better than PRE as a smoothness diagnostic. The two
spaces give very similar correlations with normalized validation error across
GemNet, SchNet, BoB, and PRE.

The important result is stronger than the specific feature choice:

```text
Local Dirichlet smoothness in chemically meaningful local representations
correlates strongly with normalized model difficulty.
```

PRE+A confirms this rather than replacing PRE. It adds angular information and
improves the fitted PRE baseline, but the smoothness diagnostic was already
capturing most of the relevant target ordering in pair-radial PRE space.

