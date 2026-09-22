"""Fixed molecular descriptors used as architecture-informed proxies."""

from .angular import angular_feature_names, build_angular_matrix
from .pre import PAPER_ELEMENTS, build_pre_matrix, radial_grid

__all__ = [
    "PAPER_ELEMENTS",
    "angular_feature_names",
    "build_angular_matrix",
    "build_pre_matrix",
    "radial_grid",
]
