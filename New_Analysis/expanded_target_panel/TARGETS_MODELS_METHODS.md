# Target, Baseline, and GNN Methods Reference

This document records the target definitions and model protocols used in the current manuscript analysis. The main learned molecular models are **GemNet** and **SchNet** trained on target-specific tmQM LMDBs. Linear and MLP baselines are included here only because they define fitted baseline targets and residual targets used for GemNet/SchNet training.

All energies and errors are reported in Hartree unless otherwise stated.

## Scope

Included:

- tmQM target generation for reference-corrected and BoE-corrected targets.
- Linear baseline target generation with BoB, PRE, and PRE+A descriptors.
- MLP baseline target generation with BoB, PRE, and PRE+A descriptors.
- GemNet and SchNet training on the target panel.
- Split-aware fitting, preprocessing, and validation rules.

Excluded:

- Morgan fingerprints, GFN-FF, learned-embedding diagnostics, smoothness-only experiments, QM9 stacking experiments, and exploratory OOD analyses.

## Dataset and Splits

The in-domain dataset is tmQM. All supervised target models use the same GemNet split IDs:

```text
trillium_project/split_ids/tmqm_train_ids.csv
trillium_project/split_ids/tmqm_val_ids.csv
trillium_project/split_ids/tmqm_test_ids.csv
```

For the fully aligned DFT/PRE/PRE+A targets, the split sizes are:

```text
train: 56830
val:    7104
test:   7104
```

Some targets based on BoB features have a few fewer molecules because a small number of molecules are unavailable in the BoB target/features table. LMDB generation scripts skip missing target rows and renumber LMDB keys densely so FairChem reads keys `0..length-1`.

## Reference and BoE Corrections

The raw method energy for molecule `i` is denoted

```text
E_raw^m(i)
```

where `m` is DFT or xTB. The first correction removes method-specific atomic reference energies:

```text
E_ref^m(i) = E_raw^m(i) - sum_Z n_Z(i) epsilon_Z^m
```

Here `n_Z(i)` is the number of atoms of element `Z` in molecule `i`, and `epsilon_Z^m` is the atomic reference energy for element `Z` computed for the same electronic-structure method `m`. DFT and xTB therefore use separate atomic reference tables.

The BoE correction is then applied to the reference-corrected target:

```text
E_ref,BoE^m(i) = E_ref^m(i) - BoE_fit_m(i)
```

where `BoE_fit_m` is the fitted bond-order energy model for the same method-specific reference target. In the saved target CSVs, these appear as:

| Quantity | Column |
|---|---|
| DFT raw energy | `DFT_raw` |
| DFT reference-corrected energy | `DFT_ref` |
| DFT BoE fitted value | `BoE_fit_DFT_ref` |
| DFT reference/BoE-corrected target | `DFT_ref-BoE` |
| xTB raw energy | `xTB_raw` |
| xTB reference-corrected energy | `xTB_ref` |
| xTB BoE fitted value | `BoE_fit_xTB_ref` |
| xTB reference/BoE-corrected target | `xTB_ref-BoE` |

The xTB residual target is defined after both DFT and xTB receive their own reference and BoE corrections:

```text
Delta^xTB_ref,BoE = DFT_ref,BoE - xTB_ref,BoE
```

The arithmetic identity is checked in:

```text
New_Analysis/prove_boe_difference_equivalence.py
```

## Canonical Target Panel

The canonical expanded target registry is:

```text
New_Analysis/expanded_target_panel/target_registry.csv
```

| Target ID | Manuscript label | CSV | Column | Role |
|---|---|---|---|---|
| `dft` | `DFT_ref,BoE` | `New_Analysis/datasets/tmQM__DFT_ref-BoE.csv` | `DFT_ref-BoE` | reference target |
| `xtb` | `xTB_ref,BoE` | `New_Analysis/datasets/tmQM__xTB_ref-BoE.csv` | `xTB_ref-BoE` | physics baseline target |
| `deltaxtb` | `Delta^xTB_ref,BoE` | `New_Analysis/datasets/tmQM__Delta_xTB-ref-BoE.csv` | `Delta_xTB-ref-BoE` | residual target |
| `deltabob` | `Delta^BoB_ref,BoE` | `New_Analysis/datasets/tmQM__Delta_BoB-ref-BoE.csv` | `E_resid_after_BoE_BoB_tmQM` | residual target |
| `deltapre` | `Delta^PRE_ref,BoE` | `New_Analysis/datasets/tmQM__Delta_PRE-ref-BoE.csv` | `E_resid_after_BoE_PRE_tmQM` | residual target |
| `bob` | `BoB_ref,BoE` | `New_Analysis/datasets/tmQM__BoB-ref-BoE.csv` | `BoB_ref-BoE` | fitted baseline target |
| `pre` | `PRE_ref,BoE` | `New_Analysis/datasets/tmQM__PRE-ref-BoE.csv` | `PRE_ref-BoE` | fitted baseline target |
| `pre_plus_a` | `PRE+A_ref,BoE` | `New_Analysis/datasets/tmQM__PRE_plus_A-ref-BoE.csv` | `PRE_plus_A_ref-BoE` | fitted baseline target |
| `deltapre_plus_a` | `Delta^PRE+A_ref,BoE` | `New_Analysis/datasets/tmQM__Delta_PRE_plus_A-ref-BoE.csv` | `Delta_PRE_plus_A-ref-BoE` | residual target |

The fitted baseline residual targets are:

```text
Delta^BoB_ref,BoE   = DFT_ref,BoE - BoB_ref,BoE
Delta^PRE_ref,BoE   = DFT_ref,BoE - PRE_ref,BoE
Delta^PRE+A_ref,BoE = DFT_ref,BoE - PRE+A_ref,BoE
```

## Descriptor Baselines Used for Target Generation

### BoB

BoB is a precomputed Bag-of-Bonds descriptor used as a global Coulomb-matrix-like molecular representation. The local feature matrix has:

```text
BoB dimensions: 4950
```

Primary source files:

```text
trillium_project/baseline_validation/bob/tmqm_bob_feature_matrix.npz
trillium_project/baseline_validation/bob/tmqm_bob_feature_mol_ids.csv
```

### PRE

PRE is the pair-radial expansion descriptor. It represents molecules with typed pair-radial features and is pair-only: it uses atom-pair radial information but does not explicitly encode angles.

```text
PRE dimensions: 34650
```

Primary source cache:

```text
pre_trillium_cpu_jobs/cache
```

or the local default cache location:

```text
New_Analysis/features/PRE/trillium_cache
```

### PRE+A

PRE+A augments PRE with a compact angular environment descriptor. The angular part is not a full typed angular force field. It is a central-element angular histogram:

```text
A[Z_center, radial_shell_pair, angle_bin] += 1
```

For each central atom, neighbor pairs within a 6.0 Angstrom cutoff define an angle. Neighbor distances are binned into radial shells, and the angle is binned uniformly in `cos(theta)`.

Angular descriptor defaults:

```text
cutoff = 6.0 Angstrom
radial shell edges = [0.0, 1.3, 1.8, 2.3, 3.0, 4.2, 6.0]
n radial shells = 6
n radial shell pairs = 21
n angle bins = 12
angle bins = uniform in cos(theta) from -1 to 1
central element vocabulary size = 44
angular dimensions = 44 * 21 * 12 = 11088
```

Therefore:

```text
PRE dimensions   = 34650
A dimensions     = 11088
PRE+A dimensions = 45738
```

The full angular cache was built from:

```text
Original_Data/xyz_files/tmqm_curated.extxyz
```

and saved as:

```text
New_Analysis/features/PRE_A_debug_full/tmqm_angular_features.npz
New_Analysis/features/PRE_A_debug_full/tmqm_angular_mol_ids.csv
New_Analysis/features/PRE_A_debug_full/tmqm_angular_feature_names.json
New_Analysis/features/PRE_A_debug_full/tmqm_angular_meta.json
```

In the full run:

```text
n molecules = 71038
n angular features = 11088
mean atoms per molecule = 66.31
mean nonzero angular channels per molecule = 755.81
```

PRE+A captures angular distributions around central elements, coordination geometry, and local packing or strain signatures. It does not capture fully typed angles such as C-N-O versus H-N-H, neighbor-pair element-specific angular interactions, or a complete angular force-field expansion.

## Linear Ridge Baseline Fitting

Linear BoB, PRE, and PRE+A baselines are fit with ridge regression. The main script is:

```text
New_Analysis/expanded_target_panel/run_expanded_ridge_targets.py
```

For a descriptor matrix `X` and target `y`, the fitted baseline is:

```text
baseline_ref,BoE = Ridge(X -> DFT_ref,BoE)
```

or, in the expanded panel, `Ridge(X -> target)` for each requested target. Residual targets are always constructed as:

```text
Delta^baseline_ref,BoE = DFT_ref,BoE - baseline_ref,BoE
```

Preprocessing is split-aware:

- Drop zero-variance channels using the training split only.
- Apply the same retained-channel mask to validation and test.
- Estimate feature variances on the training split only.
- Scale features by train-set standard deviation only.
- Do not center sparse feature matrices.
- Select ridge alpha using train-only 5-fold CV.
- No validation or test rows enter filtering, scaling, alpha selection, or coefficient fitting.

Default ridge settings:

```text
CV folds = 5
CV shuffle = True
CV random_state = 42
zero_var_eps = 1e-12
solver = sklearn Ridge(solver="auto")
metric for alpha selection = CV MAE
```

The broad alpha grid used in the original all-target scans was:

```text
alphas = logspace(-6, 3, 20)
```

After high-alpha checks showed the relevant optima were near larger regularization values for PRE/PRE+A, the expanded completion runs used narrower target-specific grids:

```text
PRE expanded targets:    100, 335.9818286283781, 1000, 3000
PRE+A expanded targets:  1000, 3000, 10000
```

The selected alpha is not expected to be identical across targets. It can change because target scale, effective noise level, descriptor alignment, and retained feature rank all change across targets. The scan only needs enough resolution to locate the performance plateau; exact alpha precision is not scientifically meaningful when adjacent alphas have nearly identical CV MAE.

Important historical caveat: some older GemNet residual LMDB targets were generated from full-dataset linear baseline artifacts. The current expanded target panel is split-aware and should be used for strict no-leakage baseline evaluation.

## MLP Baseline Fitting

The MLP baselines were introduced to test whether nonlinear BoB, PRE, or PRE+A baselines produce smaller and less smooth residuals than linear ridge baselines. These are still baseline target generators, not the main manuscript GNNs.

Main script:

```text
New_Analysis/mlp_baselines/train_mlp_baselines.py
```

The fitted MLP targets are:

```text
MLP_BoB_ref,BoE   = MLP(BoB -> DFT_ref,BoE)
MLP_PRE_ref,BoE   = MLP(PRE -> DFT_ref,BoE)
MLP_PRE+A_ref,BoE = MLP(PRE+A -> DFT_ref,BoE)
```

The residual targets are:

```text
Delta^MLP_BoB_ref,BoE   = DFT_ref,BoE - MLP_BoB_ref,BoE
Delta^MLP_PRE_ref,BoE   = DFT_ref,BoE - MLP_PRE_ref,BoE
Delta^MLP_PRE+A_ref,BoE = DFT_ref,BoE - MLP_PRE+A_ref,BoE
```

Default MLP architecture and optimization:

```text
hidden dimensions = 512, 128
activation = SiLU
dropout = 0.10
output = scalar linear head
optimizer = AdamW
learning rate = 1e-3
weight decay = 1e-4
loss = SmoothL1Loss, beta = 0.10
batch size = 1024
eval batch size = 2048
max epochs = 800
early stopping patience = 80
min delta = 1e-5
gradient clipping = 1.0
seed = 101
AMP = enabled unless disabled
```

Preprocessing is also split-aware:

- Drop zero-variance channels using train only.
- Scale sparse features using train-only variances.
- Normalize the target using train-set mean and standard deviation.
- Cache dense feature tensors on GPU when memory allows, because sparse-to-dense conversion is otherwise the bottleneck.

The DFT_ref,BoE validation results for the saved MLP baselines were:

| Model | Feature space | Val MAE | Val MAE/IQR | Best epoch | Total features | Retained features |
|---|---:|---:|---:|---:|---:|---:|
| `MLP_BoB` | BoB | 0.021207 | 0.250219 | 461 | 4950 | 867 |
| `MLP_PRE` | PRE | 0.018927 | 0.223337 | 97 | 34650 | 13230 |
| `MLP_PRE_plus_A` | PRE+A | 0.017052 | 0.201206 | 146 | 45738 | 20796 |

Saved MLP residual target paths:

```text
New_Analysis/mlp_baselines/outputs/bob/tmQM__Delta_MLP_BoB-ref-BoE.csv
New_Analysis/mlp_baselines/outputs/pre/tmQM__Delta_MLP_PRE-ref-BoE.csv
New_Analysis/mlp_baselines/outputs/pre_plus_a/tmQM__Delta_MLP_PRE_plus_A-ref-BoE.csv
```

## LMDB Target Generation for GemNet and SchNet

GemNet and SchNet train on FairChem single-point LMDB datasets. For each target, the preparation scripts:

1. Read the canonical target CSV by `mol_id`.
2. Read the source tmQM geometry LMDB split.
3. Replace each molecule's energy target with the requested scalar target.
4. Write a new train/val/test LMDB for that target.
5. Compute train-set target mean and standard deviation.
6. Store train-set target normalization in the FairChem config.
7. Write a preflight script and manifest.

Core source LMDBs:

```text
lmdb_files/tmqm_totE/tmqm_totE_matched_train.lmdb
lmdb_files/tmqm_totE/tmqm_totE_matched_val.lmdb
lmdb_files/tmqm_totE/tmqm_totE_matched_test.lmdb
```

Expanded target preparation scripts:

```text
New_Analysis/gemnet_runs/trillium/expanded_ref_boe_targets_run_bundle/prepare_expanded_gemnet_ref_boe_targets_trillium.py
New_Analysis/schnet_runs/trillium/expanded_ref_boe_targets_run_bundle/prepare_expanded_schnet_ref_boe_targets_trillium.py
New_Analysis/mlp_residual_gnn_runs/prepare_mlp_residual_gnn_targets_trillium.py
```

## GemNet Model

GemNet is the higher-capacity local geometric GNN used as the main model. It is trained separately for each scalar target.

Representative GemNet-T config:

```text
model.name = gemnet_t
num_spherical = 7
num_radial = 128
num_blocks = 3
emb_size_atom = 512
emb_size_edge = 512
emb_size_trip = 64
emb_size_rbf = 16
emb_size_cbf = 16
emb_size_bil_trip = 64
num_before_skip = 1
num_after_skip = 2
num_concat = 1
num_atom = 3
cutoff = 6.0 Angstrom
max_neighbors = 50
rbf = gaussian
cbf = spherical_harmonics
envelope = polynomial, exponent 5
activation = silu
extensive = True
regress_forces = False
direct_forces = False
use_pbc = False
otf_graph = True
```

GemNet is local by construction because the graph is built from neighbors within the cutoff. Its radial basis encodes pair distances, and its spherical-harmonic/circular basis terms encode angular triplet geometry. Many-body and longer-range effects can be approximated through stacked interaction/message-passing blocks, but the explicit geometric features are still cutoff-local.

Representative optimizer settings from the GemNet template:

```text
trainer = energy
optimizer = AdamW
lr_initial = 5e-4
amsgrad = True
scheduler = ReduceLROnPlateau
scheduler mode = min
factor = 0.8
patience = 3
max_epochs = 80
eval_every = 5000
num_workers = 2
ema_decay = 0.999
clip_grad_norm = 10
loss_energy = mse
energy_coefficient = 1
```

The generated target-specific configs are the source of truth for each run. They set:

```text
cmd.seed = 101
dataset[train].normalize_labels = True
dataset[train].target_mean = train target mean
dataset[train].target_std = train target std
model.scale_file = train/scaling_factors/gemnet-dT.json
```

GemNet validation MAE is taken from the best validation checkpoint in the training log.

## SchNet Model

SchNet is the controlled lower-capacity local geometric GNN used as the second main model. It is also trained separately for each scalar target.

The SchNet config block generated for expanded targets is:

```text
model.name = schnet
use_pbc = False
regress_forces = False
otf_graph = True
hidden_channels = 128
num_filters = 128
num_interactions = 6
num_gaussians = 50
cutoff = 6.0 Angstrom
readout = sum
```

Shared SchNet training settings:

```text
seed = 101
max_epochs = 80
eval_every = 5000
batch_size = 16
eval_batch_size = 16
num_workers = 2
dataset[train].normalize_labels = True
dataset[train].target_mean = train target mean
dataset[train].target_std = train target std
```

SchNet is also a local interaction model: it constructs an on-the-fly neighbor graph, expands pair distances with Gaussian radial functions, applies multiple continuous-filter interaction blocks, and uses sum readout to produce an extensive molecular energy.

## Completed GNN Target Coverage

The main target panel for GemNet and SchNet includes:

```text
DFT_ref,BoE
xTB_ref,BoE
Delta^xTB_ref,BoE
Delta^BoB_ref,BoE
Delta^PRE_ref,BoE
BoB_ref,BoE
PRE_ref,BoE
PRE+A_ref,BoE
Delta^PRE+A_ref,BoE
```

Additional MLP-residual targets trained by GemNet and SchNet:

```text
Delta^MLP_BoB_ref,BoE
Delta^MLP_PRE_ref,BoE
Delta^MLP_PRE+A_ref,BoE
```

Key result tables:

```text
New_Analysis/expanded_target_panel/outputs/status/completed_model_target_val_metrics.csv
New_Analysis/expanded_target_panel/outputs/gnn/expanded_gnn_val_mae_summary.csv
New_Analysis/mlp_residual_gnn_runs/outputs/mlp_residual_gnn_val_mae_summary_all.csv
```

## Validation Rules and Reproducibility Checks

The following checks are part of the workflow:

- All target CSVs are keyed by `mol_id`.
- Train/val/test rows are assigned using the fixed GemNet split IDs.
- Baseline preprocessing is fit on train only.
- Validation/test rows never enter baseline filtering, scaling, alpha CV, or coefficient fitting.
- GNN target LMDBs are generated by matching source LMDB molecules to target CSV rows by molecule ID.
- Target-specific LMDBs are checked by preflight scripts before submission.
- GNN validation MAE is parsed from train logs, using the best validation checkpoint.
- Target scale normalization for reported normalized MAE uses the validation target IQR.

## Main Takeaway for Manuscript Language

The baseline descriptors define chemically motivated target transformations, but the primary predictive models are GemNet and SchNet. Each target asks whether removing a particular reference or baseline component leaves a residual better aligned with local geometric neural-network inductive bias. The key comparisons should therefore be framed as:

```text
How well do GemNet and SchNet learn each transformed target on the same tmQM split?
```

not as a competition among all possible baseline regressors.
