"""Angular block used to augment PRE into PRE+A."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from .pre import PAPER_ELEMENTS, _rows_to_csr


DEFAULT_SHELL_EDGES = (0.0, 1.3, 1.8, 2.3, 3.0, 4.2, 6.0)


def shell_pair_indices(n_shells: int) -> dict[tuple[int, int], int]:
    return {
        pair: index
        for index, pair in enumerate((first, second) for first in range(n_shells) for second in range(first, n_shells))
    }


def angular_feature_names(
    elements=PAPER_ELEMENTS,
    shell_edges=DEFAULT_SHELL_EDGES,
    n_angle_bins: int = 12,
) -> list[str]:
    angle_edges = np.linspace(-1.0, 1.0, n_angle_bins + 1)
    names = []
    for element in elements:
        for first in range(len(shell_edges) - 1):
            for second in range(first, len(shell_edges) - 1):
                shell = f"r{shell_edges[first]:g}-{shell_edges[first + 1]:g}__r{shell_edges[second]:g}-{shell_edges[second + 1]:g}"
                for angle in range(n_angle_bins):
                    names.append(
                        f"A_center={element}__{shell}__costheta{angle_edges[angle]:.3f}:{angle_edges[angle + 1]:.3f}"
                    )
    return names


def molecule_angular_counts(
    symbols: tuple[str, ...],
    positions: np.ndarray,
    *,
    elements=PAPER_ELEMENTS,
    shell_edges=DEFAULT_SHELL_EDGES,
    n_angle_bins: int = 12,
) -> dict[int, float]:
    element_to_index = {element: index for index, element in enumerate(elements)}
    unknown = sorted(set(symbols) - set(element_to_index))
    if unknown:
        raise ValueError(f"Elements outside the configured vocabulary: {unknown}")
    shell_edges_array = np.asarray(shell_edges, dtype=float)
    angle_edges = np.linspace(-1.0, 1.0, n_angle_bins + 1)
    n_shells = len(shell_edges) - 1
    pair_to_index = shell_pair_indices(n_shells)
    n_shell_pairs = len(pair_to_index)
    output: dict[int, float] = defaultdict(float)
    if len(symbols) < 3:
        return {}

    displacement = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=-1)
    cutoff = float(shell_edges[-1])
    for center in range(len(symbols)):
        neighbours = np.where((distance[center] > 1e-8) & (distance[center] <= cutoff))[0]
        shells = np.searchsorted(shell_edges_array, distance[center, neighbours], side="right") - 1
        keep = (shells >= 0) & (shells < n_shells)
        neighbours, shells = neighbours[keep], shells[keep]
        base = element_to_index[symbols[center]] * n_shell_pairs * n_angle_bins
        for left in range(len(neighbours)):
            first = int(neighbours[left])
            first_vector = positions[first] - positions[center]
            for right in range(left + 1, len(neighbours)):
                second = int(neighbours[right])
                second_vector = positions[second] - positions[center]
                denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
                if denominator <= 0:
                    continue
                cosine = float(np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0))
                angle_bin = int(np.searchsorted(angle_edges, cosine, side="right") - 1)
                angle_bin = max(0, min(n_angle_bins - 1, angle_bin))
                shell_pair = tuple(sorted((int(shells[left]), int(shells[right]))))
                feature = base + pair_to_index[shell_pair] * n_angle_bins + angle_bin
                output[feature] += 1.0
    return dict(output)


def build_angular_matrix(
    records,
    *,
    elements=PAPER_ELEMENTS,
    shell_edges=DEFAULT_SHELL_EDGES,
    n_angle_bins: int = 12,
) -> tuple[csr_matrix, list[str]]:
    molecule_ids: list[str] = []
    rows: list[dict[int, float]] = []
    for molecule_id, symbols, positions in records:
        molecule_ids.append(str(molecule_id))
        rows.append(
            molecule_angular_counts(
                symbols,
                positions,
                elements=elements,
                shell_edges=shell_edges,
                n_angle_bins=n_angle_bins,
            )
        )
    n_shells = len(shell_edges) - 1
    n_features = len(elements) * (n_shells * (n_shells + 1) // 2) * n_angle_bins
    return _rows_to_csr(rows, n_features), molecule_ids
