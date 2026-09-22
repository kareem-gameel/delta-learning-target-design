# Method

## Two-dimensional target assessment

For a parent target `y` and baseline prediction `b`, delta learning trains the
downstream model on the residual `r = y - b`. The transformation changes both:

- target scale, measured here by `IQR(r)`; and
- target variation over a representation relevant to the downstream model.

The second quantity is measured by graph Dirichlet roughness.

## Weighted graph roughness

Let `x_i` be a fixed descriptor for molecule `i`. Descriptor channels are
divided by their population standard deviation without mean centering. A
cosine `k`-nearest-neighbour graph is then constructed. The directed graph is
converted to an undirected union graph, averaging the two directed weights
when an edge appears in both directions.

For directed neighbour `j` of `i`, the primary edge weight is

```text
w_ij = exp[-d(i,j)^2 / (sigma_i sigma_j)]
```

where `sigma_i` is the cosine distance from molecule `i` to its `k`th nearest
neighbour. The normalized roughness is

```text
D_IQR(y) = sum_ij w_ij (y_i - y_j)^2
           --------------------------------- .
           sum_ij w_ij IQR(y)^2
```

`D_IQR` is invariant to a constant target offset and to nonzero linear target
rescaling. A lower value means that nearby molecules have more similar labels
relative to the target's own scale.

## Architecture-informed feature spaces

### PRE

Pair-Radial Expansion (PRE) accumulates unordered, element-resolved pair
distances in Gaussian radial basis channels up to 6 angstrom. The paper
configuration uses 44 elements and 35 radial channels, giving 34,650 features.
PRE approximates the local pair-distance information used by SchNet.

### PRE+A

PRE+A appends an angular block:

```text
A[central element, radial shell pair, cos(theta) bin] += 1
```

The paper configuration uses six radial shells and 12 angular bins, giving
11,088 angular features and 45,738 total PRE+A features. PRE+A approximates the
explicit pair and triplet geometry available to GemNet.

These descriptors are fixed proxies chosen before downstream training. They do
not reproduce every learned internal feature of either architecture.

## Interpretation

`D_IQR` estimates normalized target difficulty in a specified feature space; it
does not determine absolute MAE by itself. Absolute performance also depends on
target scale. A transformation that lowers both IQR and `D_IQR` is favoured by
both criteria. Lower IQR accompanied by higher `D_IQR` is a tradeoff whose final
performance must be assessed empirically.
