"""Target-design diagnostics for delta learning."""

from .roughness import (
    GraphRoughnessResult,
    build_weighted_knn_graph,
    compute_graph_roughness,
    iqr,
    scale_features_by_population_std,
)
from .screening import compare_to_parent

__all__ = [
    "GraphRoughnessResult",
    "build_weighted_knn_graph",
    "compare_to_parent",
    "compute_graph_roughness",
    "iqr",
    "scale_features_by_population_std",
]

__version__ = "0.1.0"
