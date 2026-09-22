"""Pair-Radial Expansion (PRE) descriptor construction."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.sparse import csr_matrix


PAPER_ELEMENTS = (
    "Ag", "As", "Au", "B", "Br", "C", "Cd", "Cl", "Co", "Cr", "Cu", "F", "Fe", "H", "Hf", "Hg",
    "I", "Ir", "La", "Mn", "Mo", "N", "Nb", "Ni", "O", "Os", "P", "Pd", "Pt", "Re", "Rh", "Ru",
    "S", "Sc", "Se", "Si", "Ta", "Tc", "Ti", "V", "W", "Y", "Zn", "Zr",
)


def radial_grid(cutoff: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper PRE radial centers and Gaussian widths."""

    if cutoff < 0.8:
        raise ValueError("PRE requires a cutoff of at least 0.8 angstrom.")
    parts = [np.arange(0.8, min(3.0, cutoff) + 1e-12, 0.1)]
    if cutoff >= 3.25:
        parts.append(np.arange(3.25, cutoff + 1e-12, 0.25))
    centers = np.concatenate(parts)
    widths = np.where(centers <= 3.0, 0.12, 0.30)
    return centers, widths


def pre_feature_names(elements=PAPER_ELEMENTS, cutoff: float = 6.0) -> list[str]:
    centers, _ = radial_grid(cutoff)
    names = []
    for first, second in combinations_with_replacement(elements, 2):
        names.extend(f"PRE_{first}-{second}_mu={center:.2f}" for center in centers)
    return names


def molecule_pre_counts(
    symbols: tuple[str, ...],
    positions: np.ndarray,
    *,
    elements=PAPER_ELEMENTS,
    cutoff: float = 6.0,
    eps: float = 1e-10,
) -> dict[int, float]:
    """Accumulate element-resolved radial-basis responses for one molecule."""

    centers, widths = radial_grid(cutoff)
    pairs = list(combinations_with_replacement(elements, 2))
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    allowed = set(elements)
    output: dict[int, float] = {}
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            if symbols[first] not in allowed or symbols[second] not in allowed:
                raise ValueError(f"Element outside the configured vocabulary: {symbols[first]}, {symbols[second]}")
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            if distance > cutoff:
                continue
            cutoff_weight = 0.5 * (np.cos(np.pi * distance / cutoff) + 1.0)
            response = np.exp(-((distance - centers) ** 2) / (2.0 * widths**2)) * cutoff_weight
            pair = tuple(sorted((symbols[first], symbols[second])))
            base = pair_to_index[pair] * len(centers)
            for channel, value in enumerate(response):
                if value > eps:
                    output[base + channel] = output.get(base + channel, 0.0) + float(value)
    return output


def _rows_to_csr(rows: list[dict[int, float]], n_features: int) -> csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        for column, value in sorted(row.items()):
            indices.append(column)
            data.append(value)
        indptr.append(len(indices))
    return csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(rows), n_features),
        dtype=np.float32,
    )


def build_pre_matrix(records, *, elements=PAPER_ELEMENTS, cutoff: float = 6.0, eps: float = 1e-10):
    """Build a sparse PRE matrix from extxyz-style records."""

    centers, _ = radial_grid(cutoff)
    n_pairs = len(tuple(combinations_with_replacement(elements, 2)))
    molecule_ids: list[str] = []
    rows: list[dict[int, float]] = []
    for molecule_id, symbols, positions in records:
        molecule_ids.append(str(molecule_id))
        rows.append(molecule_pre_counts(symbols, positions, elements=elements, cutoff=cutoff, eps=eps))
    return _rows_to_csr(rows, n_pairs * len(centers)), molecule_ids
