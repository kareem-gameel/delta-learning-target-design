"""Minimal streaming extended-XYZ reader used by the descriptor builders."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np


ID_PATTERN = re.compile(r'(?:mol_id|CSD_code)=(?:"([^"]+)"|([^\s]+))')


def iter_extxyz(path: Path) -> Iterator[tuple[str, tuple[str, ...], np.ndarray]]:
    """Yield ``(molecule_id, symbols, positions)`` records."""

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Expected atom count, found {line!r}.") from exc
            header = handle.readline()
            match = ID_PATTERN.search(header)
            if match is None:
                raise ValueError("Each extxyz header must contain mol_id= or CSD_code=.")
            molecule_id = match.group(1) or match.group(2)
            symbols: list[str] = []
            positions: list[tuple[float, float, float]] = []
            for _ in range(n_atoms):
                tokens = handle.readline().split()
                if len(tokens) < 4:
                    raise ValueError(f"Malformed atom row in molecule {molecule_id!r}.")
                symbols.append(tokens[0])
                positions.append(tuple(float(value) for value in tokens[1:4]))
            yield molecule_id, tuple(symbols), np.asarray(positions, dtype=np.float64)
