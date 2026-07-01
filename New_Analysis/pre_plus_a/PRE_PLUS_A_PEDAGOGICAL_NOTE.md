# PRE+A Pedagogical Note

## Motivation

Our original `PRE` baseline is a typed pair-radial linear model. It describes a molecule by counting pairwise interatomic environments:

```text
which element pair?
at what distance?
```

Then ridge regression learns a linear map from those pair-radial features to the target energy.

For the ref,BoE target-design work, the target is:

```text
DFT_ref,BoE
```

and the original PRE baseline is:

```text
PRE_ref,BoE = Ridge(PRE -> DFT_ref,BoE)
Delta^PRE_ref,BoE = DFT_ref,BoE - PRE_ref,BoE
```

The question behind `PRE+A` is:

> Does adding compact angular environment information improve a pair-radial PRE baseline, or produce a better residual target for GNN learning?

This is useful because GemNet-like models are not purely pair-radial models. They use geometric information that includes angular structure. If PRE is missing angular information, then `Delta^PRE_ref,BoE` may contain structured angular signal. `PRE+A` tests that idea without building a full angular force field.

## Definition

`PRE+A` is a ridge baseline using a concatenated feature vector:

```text
PRE+A features = [PRE typed pair-radial features | A angular environment features]
```

The fitted baseline is:

```text
PRE+A_ref,BoE = Ridge([PRE, A] -> DFT_ref,BoE)
```

and its residual target is:

```text
Delta^PRE+A_ref,BoE = DFT_ref,BoE - PRE+A_ref,BoE
```

So `PRE+A` differs from `PRE` only by the added angular block `A`.

## PRE Block

The original `PRE` descriptor is a typed pair-radial expansion.

Conceptually, it has channels like:

```text
(element pair type, radial basis channel)
```

For each molecule, we loop over atom pairs and accumulate contributions into the appropriate channel. The target value is not used when building the feature vector. The feature vector is purely geometric/compositional.

In the current tmQM cache:

```text
PRE dimensions = 34,650
```

This is large because the descriptor is typed by element pairs and radial channels.

## Angular A Block

The angular block is intentionally compact. It is not a full typed triplet model.

For each central atom, we look at unordered neighbor pairs around that center:

```text
i - j - k
```

where `j` is the central atom and `i`, `k` are neighbors within the cutoff.

For each angle, we accumulate:

```text
A[Z_center, radial_shell_pair, angle_bin] += 1
```

This means the angular descriptor records:

1. The element of the central atom.
2. The radial shells of the two neighbor distances.
3. The angle bin for `cos(theta_ijk)`.

The default angular setup is:

```text
cutoff = 6.0 Angstrom
radial shell edges = [0.0, 1.3, 1.8, 2.3, 3.0, 4.2, 6.0]
n radial shells = 6
n unordered radial shell pairs = 21
n angle bins = 12
central element vocabulary size = 44
```

Therefore:

```text
A dimensions = 44 * 21 * 12 = 11,088
```

The full `PRE+A` vector has:

```text
PRE+A dimensions = 34,650 + 11,088 = 45,738
```

## What PRE+A Captures

`PRE+A` captures:

- pair-radial element-pair structure from PRE,
- angular distributions around each central element,
- coordination geometry,
- local packing and strain signatures,
- broad angular environment differences such as linear, bent, trigonal, tetrahedral, or crowded neighborhoods.

It does not capture:

- fully typed angles such as `C-N-O` versus `H-N-H`,
- explicit neighbor-pair element typing in the angular block,
- continuous radial-angular basis coupling like `R(r_ij) R(r_jk) A(theta_ijk)`,
- a full angular force field,
- learned nonlinear interactions between angular channels.

This is deliberate. `PRE+A` is meant to be a clean, feasible, linear diagnostic baseline, not a replacement for GemNet.

## Why Central-Element Typing Only?

A fully typed triplet descriptor would use channels like:

```text
Z_i, Z_j, Z_k, radial channel for r_ij, radial channel for r_jk, angle channel
```

That can explode in dimensionality because tmQM contains many elements. We chose central-element typing as a compact compromise:

```text
Z_center only
```

This still captures coordination geometry around different element types, while keeping the angular block manageable.

## Fitting Protocol

The fitting protocol follows the same split-aware rule used for the other clean baselines.

Target:

```text
DFT_ref,BoE
```

Splits:

```text
GemNet train / val / test split IDs
```

Model:

```text
Ridge([PRE, A] -> DFT_ref,BoE)
```

Preprocessing:

1. Build `[PRE, A]` for all aligned molecules.
2. Use only train rows to decide which channels have nonzero variance.
3. Drop zero-variance channels based on train only.
4. Scale retained channels using train-only standard deviations.
5. Select alpha using train-only cross-validation.
6. Fit final ridge model on train only.
7. Evaluate on train, val, and test.

Validation/test rows do not affect feature filtering, scaling, alpha selection, or coefficients.

## Comparison to PRE

The scientific comparison is:

```text
PRE_ref,BoE      vs PRE+A_ref,BoE
Delta^PRE_ref,BoE vs Delta^PRE+A_ref,BoE
```

If PRE+A improves the baseline MAE, then some of the error left by PRE was likely angular/local-geometry signal that the pair-radial descriptor did not represent well.

In our split-aware `DFT_ref,BoE` check, PRE+A improved over PRE on the shared filtered validation mask:

```text
PRE val MAE      ~ 0.0200 Ha
PRE+A val MAE    ~ 0.0176 Ha
```

So PRE+A improves the fitted baseline by roughly:

```text
about 12%
```

That is large enough to justify keeping PRE+A as a target-design diagnostic.

However, this does not automatically mean `Delta^PRE+A_ref,BoE` will be the best GNN target. The residual may be smaller, but it may also remove structure that the GNN could have learned well. That is why we still compare residual smoothness and GNN validation MAE.

## Toy Example

Suppose we have a tiny molecule with one central carbon atom and three hydrogens:

```text
    H1
     \
      C
     / \
   H2   H3
```

Assume all three C-H distances fall in radial shell 1.

There are three unordered neighbor pairs around the central carbon:

```text
H1-C-H2
H1-C-H3
H2-C-H3
```

For each pair, we compute the angle:

```text
theta(H1-C-H2)
theta(H1-C-H3)
theta(H2-C-H3)
```

Then we bin `cos(theta)` into one of 12 angle bins.

If all three angles fall into angle bin 4, then the angular block receives:

```text
A[C, shell_pair=(1,1), angle_bin=4] += 3
```

All other angular channels remain zero for this simple example.

Now imagine a second molecule with the same pair distances but different geometry. Maybe its H-C-H angles fall into angle bin 8 instead. Then its PRE pair-radial features could look similar, but its angular block would differ:

```text
A[C, shell_pair=(1,1), angle_bin=8] += 3
```

This is exactly what PRE cannot distinguish well: same kinds of pair distances, different angular arrangement.

## Another Toy Example With Two Radial Shells

Suppose a central metal `M` has two close nitrogen neighbors and one farther chlorine neighbor:

```text
N_close - M - N_close
Cl_far  - M - N_close
Cl_far  - M - N_close
```

If the two M-N distances fall in shell 1 and the M-Cl distance falls in shell 3, then the angular descriptor accumulates channels such as:

```text
A[M, shell_pair=(1,1), angle_bin=a] += 1
A[M, shell_pair=(1,3), angle_bin=b] += 2
```

The descriptor does not know that the neighbors are specifically N and Cl, because the angular block only types the central atom. But it does know that the central metal has one close-close angle and two close-far angles.

This is why PRE+A captures coordination geometry but is not a fully typed angular force field.

## Interpretation

PRE+A tests a specific mechanistic hypothesis:

> Pair-radial PRE may leave angular environment signal in the residual. Adding compact angular histograms should reduce that residual if the missing signal is geometrically angular.

If PRE+A improves PRE, then angular environment information matters for the baseline fit.

If `Delta^PRE+A_ref,BoE` becomes a better GNN target, then removing this angular component helps the GNN.

If `Delta^PRE+A_ref,BoE` becomes a worse GNN target, then the angular component may have been useful structure that the GNN was already good at learning.

Therefore, PRE+A is not just another baseline. It is a controlled probe of whether angular geometric structure is helpful or harmful to remove before GNN training.

