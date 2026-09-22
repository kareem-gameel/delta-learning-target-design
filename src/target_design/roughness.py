"""Architecture-informed graph Dirichlet roughness.

The implementation follows the primary paper convention: descriptor channels
are divided by their population standard deviation without mean centering, a
cosine k-nearest-neighbour graph is built, directed neighbours are combined
into an undirected union graph, and edges receive locally scaled Gaussian
weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class GraphRoughnessResult:
    """Summary of target variation on a fixed molecular graph."""

    d_iqr: float
    target_iqr: float
    weighted_mean_squared_difference: float
    n_samples: int
    n_edges: int
    weight_sum: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def iqr(values: np.ndarray) -> float:
    """Return the interquartile range using NumPy's default quantiles."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Target values must be non-empty and finite.")
    return float(np.subtract(*np.percentile(array, [75.0, 25.0])))


def _population_std(features) -> np.ndarray:
    if issparse(features):
        matrix = features.tocsr()
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix.")
    return np.std(array, axis=0, ddof=0)


def scale_features_by_population_std(features, eps: float = 1e-12):
    """Drop constant channels and divide retained channels by population SD.

    Mean centering is intentionally omitted. PRE and PRE+A are sparse count or
    accumulated-response descriptors, and centering would destroy sparsity and
    remove part of their molecular-size signal.
    """

    std = _population_std(features)
    keep = np.isfinite(std) & (std > eps)
    if not keep.any():
        raise ValueError("No non-constant feature channels remain after scaling.")
    if issparse(features):
        scaled = features.tocsr()[:, keep] @ diags(1.0 / std[keep], format="csr")
        return scaled.tocsr(), keep
    array = np.asarray(features, dtype=float)
    return array[:, keep] / std[keep], keep


def aggregate_undirected_edges(
    n_samples: int,
    neighbour_indices: np.ndarray,
    directed_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a union graph and average duplicate directed edge weights."""

    neighbours = np.asarray(neighbour_indices, dtype=np.int64)
    weights = np.asarray(directed_weights, dtype=float)
    if neighbours.shape != weights.shape:
        raise ValueError("Neighbour indices and directed weights must have equal shape.")
    source = np.repeat(np.arange(n_samples, dtype=np.int64), neighbours.shape[1])
    destination = neighbours.reshape(-1)
    values = weights.reshape(-1)
    low = np.minimum(source, destination)
    high = np.maximum(source, destination)
    keep = low != high
    low, high, values = low[keep], high[keep], values[keep]

    keys = low * np.int64(n_samples) + high
    order = np.argsort(keys, kind="mergesort")
    keys, values = keys[order], values[order]
    unique_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    averaged = np.add.reduceat(values, starts) / counts
    edges = np.column_stack((unique_keys // n_samples, unique_keys % n_samples)).astype(np.int64)
    return edges, averaged


def build_weighted_knn_graph(
    features,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Build the locally weighted undirected k-nearest-neighbour union graph."""

    n_samples = int(features.shape[0])
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    if k < 1 or k >= n_samples:
        raise ValueError(f"k must satisfy 1 <= k < n_samples; got k={k}, n={n_samples}.")

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(features)
    distances, indices = model.kneighbors(features)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    epsilon = np.finfo(float).eps
    local_scale = np.maximum(distances[:, -1], epsilon)
    source = np.repeat(np.arange(n_samples), k)
    destination = indices.reshape(-1)
    denominator = np.maximum(local_scale[source] * local_scale[destination], epsilon)
    directed_weights = np.exp(-(distances.reshape(-1) ** 2) / denominator).reshape(distances.shape)
    edges, weights = aggregate_undirected_edges(n_samples, indices, directed_weights)
    metadata: dict[str, float | int | str] = {
        "n_samples": n_samples,
        "n_edges": int(len(edges)),
        "k": int(k),
        "metric": metric,
        "weighting": "locally_scaled_gaussian",
        "weight_min": float(np.min(weights)),
        "weight_median": float(np.median(weights)),
        "weight_max": float(np.max(weights)),
        "weight_sum": float(np.sum(weights)),
    }
    return edges, weights, metadata


def compute_graph_roughness(
    target: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
) -> GraphRoughnessResult:
    """Compute IQR-normalized weighted graph Dirichlet roughness."""

    values = np.asarray(target, dtype=float).reshape(-1)
    edge_array = np.asarray(edges, dtype=np.int64)
    edge_weights = np.asarray(weights, dtype=float).reshape(-1)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("Edges must have shape (n_edges, 2).")
    if len(edge_array) != len(edge_weights):
        raise ValueError("Each edge must have one weight.")
    if len(edge_array) == 0 or np.sum(edge_weights) <= 0:
        raise ValueError("The graph must contain positive total edge weight.")
    if edge_array.min() < 0 or edge_array.max() >= len(values):
        raise ValueError("Edge indices fall outside the target array.")
    target_iqr = iqr(values)
    if target_iqr <= 0:
        raise ValueError("D_IQR is undefined for a target with zero IQR.")
    squared_difference = (values[edge_array[:, 0]] - values[edge_array[:, 1]]) ** 2
    weighted_msd = float(np.average(squared_difference, weights=edge_weights))
    return GraphRoughnessResult(
        d_iqr=weighted_msd / target_iqr**2,
        target_iqr=target_iqr,
        weighted_mean_squared_difference=weighted_msd,
        n_samples=len(values),
        n_edges=len(edge_array),
        weight_sum=float(np.sum(edge_weights)),
    )
