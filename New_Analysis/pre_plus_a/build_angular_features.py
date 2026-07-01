#!/usr/bin/env python3
"""Build compact central-element angular environment features for tmQM.

The descriptor is intentionally smaller than a fully typed triplet tensor:

    A[Z_center, radial_shell_pair, angle_bin] += 1

For every central atom j, unordered neighbor pair (i, k), and cutoff-limited
angle i-j-k, the feature stores the central element type, the two radial
shells, and the angle bin. Neighbor element types are not included in v1.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTXYZ = ROOT / "Original_Data/xyz_files/tmqm_curated.extxyz"
DEFAULT_OUT_DIR = ROOT / "New_Analysis/features/PRE_A"
DEFAULT_SHELL_EDGES = "0.0,1.3,1.8,2.3,3.0,4.2,6.0"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_float_list(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(out) < 2:
        raise ValueError("Need at least two radial shell edges.")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"Radial shell edges must be strictly increasing: {out}")
    return out


def iter_extxyz(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            nline = handle.readline()
            if not nline:
                break
            nline = nline.strip()
            if not nline:
                continue
            try:
                n_atoms = int(nline)
            except ValueError:
                continue
            header = handle.readline()
            if not header:
                break
            mol_id = None
            for token in header.split():
                if token.startswith("mol_id=") or token.startswith("CSD_code="):
                    mol_id = token.split("=", 1)[1].strip('"')
                    break
            symbols: list[str] = []
            positions: list[tuple[float, float, float]] = []
            for _ in range(n_atoms):
                toks = handle.readline().split()
                if len(toks) < 4:
                    continue
                try:
                    symbols.append(toks[0])
                    positions.append((float(toks[1]), float(toks[2]), float(toks[3])))
                except ValueError:
                    continue
            if mol_id is not None and len(symbols) == n_atoms and len(positions) == n_atoms:
                yield mol_id, tuple(symbols), np.asarray(positions, dtype=np.float64)


def collect_element_vocab(extxyz: Path, max_molecules: int | None = None) -> list[str]:
    seen: set[str] = set()
    for idx, (_mol_id, symbols, _positions) in enumerate(iter_extxyz(extxyz), start=1):
        seen.update(symbols)
        if max_molecules is not None and idx >= max_molecules:
            break
    if not seen:
        raise ValueError(f"No elements found in {extxyz}")
    return sorted(seen)


def shell_pair_indices(n_shells: int) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    idx = 0
    for a in range(n_shells):
        for b in range(a, n_shells):
            pairs[(a, b)] = idx
            idx += 1
    return pairs


def feature_names(
    element_vocab: list[str],
    shell_edges: list[float],
    angle_edges: np.ndarray,
) -> list[str]:
    n_shells = len(shell_edges) - 1
    names: list[str] = []
    for element in element_vocab:
        for s1 in range(n_shells):
            for s2 in range(s1, n_shells):
                shell = f"r{shell_edges[s1]:g}-{shell_edges[s1 + 1]:g}__r{shell_edges[s2]:g}-{shell_edges[s2 + 1]:g}"
                for angle_idx in range(len(angle_edges) - 1):
                    names.append(
                        f"A_center={element}__{shell}__costheta{angle_edges[angle_idx]:.3f}:{angle_edges[angle_idx + 1]:.3f}"
                    )
    return names


def molecule_angular_counts(
    symbols: tuple[str, ...],
    positions: np.ndarray,
    element_to_idx: dict[str, int],
    shell_edges: np.ndarray,
    angle_edges: np.ndarray,
    pair_to_idx: dict[tuple[int, int], int],
) -> dict[int, float]:
    n_atoms = int(len(symbols))
    if n_atoms < 3:
        return {}

    n_shells = len(shell_edges) - 1
    n_shell_pairs = n_shells * (n_shells + 1) // 2
    n_angle_bins = len(angle_edges) - 1
    features: dict[int, float] = defaultdict(float)

    delta = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    cutoff = float(shell_edges[-1])

    for center in range(n_atoms):
        element_idx = element_to_idx.get(symbols[center])
        if element_idx is None:
            continue
        neigh = np.where((dist[center] > 1e-8) & (dist[center] <= cutoff))[0]
        if len(neigh) < 2:
            continue
        shell_idx = np.searchsorted(shell_edges, dist[center, neigh], side="right") - 1
        keep = (shell_idx >= 0) & (shell_idx < n_shells)
        neigh = neigh[keep]
        shell_idx = shell_idx[keep]
        if len(neigh) < 2:
            continue
        base = element_idx * n_shell_pairs * n_angle_bins
        for a_pos in range(len(neigh)):
            i = int(neigh[a_pos])
            v_i = positions[i] - positions[center]
            r_i = float(np.linalg.norm(v_i))
            if r_i <= 0:
                continue
            for b_pos in range(a_pos + 1, len(neigh)):
                k = int(neigh[b_pos])
                v_k = positions[k] - positions[center]
                r_k = float(np.linalg.norm(v_k))
                denom = r_i * r_k
                if denom <= 0:
                    continue
                cos_theta = float(np.clip(np.dot(v_i, v_k) / denom, -1.0, 1.0))
                angle_idx = int(np.searchsorted(angle_edges, cos_theta, side="right") - 1)
                angle_idx = max(0, min(n_angle_bins - 1, angle_idx))
                s1 = int(shell_idx[a_pos])
                s2 = int(shell_idx[b_pos])
                if s2 < s1:
                    s1, s2 = s2, s1
                shell_pair_idx = pair_to_idx[(s1, s2)]
                feat_idx = base + shell_pair_idx * n_angle_bins + angle_idx
                features[feat_idx] += 1.0
    return features


def process_molecule(
    record: tuple[str, tuple[str, ...], np.ndarray],
    element_to_idx: dict[str, int],
    shell_edges: np.ndarray,
    angle_edges: np.ndarray,
    pair_to_idx: dict[tuple[int, int], int],
) -> tuple[str, int, Counter[str], dict[int, float]]:
    mol_id, symbols, positions = record
    counts = molecule_angular_counts(
        symbols=symbols,
        positions=positions,
        element_to_idx=element_to_idx,
        shell_edges=shell_edges,
        angle_edges=angle_edges,
        pair_to_idx=pair_to_idx,
    )
    return mol_id, len(symbols), Counter(symbols), counts


def process_batch(
    batch: list[tuple[str, tuple[str, ...], np.ndarray]],
    element_to_idx: dict[str, int],
    shell_edges: np.ndarray,
    angle_edges: np.ndarray,
    pair_to_idx: dict[tuple[int, int], int],
    n_jobs: int,
    parallel_prefer: str,
) -> list[tuple[str, int, Counter[str], dict[int, float]]]:
    if n_jobs <= 1:
        return [process_molecule(record, element_to_idx, shell_edges, angle_edges, pair_to_idx) for record in batch]
    if joblib is None:
        raise ImportError("--n-jobs > 1 requires joblib")
    return joblib.Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
        joblib.delayed(process_molecule)(record, element_to_idx, shell_edges, angle_edges, pair_to_idx)
        for record in batch
    )


def build_matrix(
    extxyz: Path,
    element_vocab: list[str],
    shell_edges: list[float],
    n_angle_bins: int,
    max_molecules: int | None,
    progress_every: int,
    n_jobs: int,
    parallel_prefer: str,
    batch_size: int,
) -> tuple[csr_matrix, list[str], dict[str, object]]:
    element_to_idx = {element: idx for idx, element in enumerate(element_vocab)}
    shell_edges_arr = np.asarray(shell_edges, dtype=float)
    angle_edges = np.linspace(-1.0, 1.0, n_angle_bins + 1, dtype=float)
    pair_to_idx = shell_pair_indices(len(shell_edges) - 1)
    n_features = len(element_vocab) * len(pair_to_idx) * n_angle_bins

    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    mol_ids: list[str] = []
    atom_counts: list[int] = []
    nnz_by_row: list[int] = []
    element_counts: Counter[str] = Counter()

    if batch_size < 1:
        raise ValueError("--batch-size must be positive")

    processed = 0
    batch: list[tuple[str, tuple[str, ...], np.ndarray]] = []

    def flush_batch() -> None:
        nonlocal processed, batch
        if not batch:
            return
        results = process_batch(
            batch=batch,
            element_to_idx=element_to_idx,
            shell_edges=shell_edges_arr,
            angle_edges=angle_edges,
            pair_to_idx=pair_to_idx,
            n_jobs=n_jobs,
            parallel_prefer=parallel_prefer,
        )
        for mol_id, n_atoms, row_element_counts, counts in results:
            mol_ids.append(mol_id)
            atom_counts.append(n_atoms)
            element_counts.update(row_element_counts)
            if counts:
                for col, value in sorted(counts.items()):
                    indices.append(int(col))
                    data.append(float(value))
            indptr.append(len(indices))
            nnz_by_row.append(len(counts))
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                log(f"Angular features: processed {processed} molecules")
        batch = []

    for record in iter_extxyz(extxyz):
        if max_molecules is not None and processed + len(batch) >= max_molecules:
            break
        batch.append(record)
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()

    x = csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(mol_ids), n_features),
        dtype=np.float32,
    )
    meta = {
        "n_molecules": len(mol_ids),
        "n_features": n_features,
        "n_elements": len(element_vocab),
        "element_vocab": element_vocab,
        "radial_shell_edges": shell_edges,
        "n_radial_shells": len(shell_edges) - 1,
        "n_radial_shell_pairs": len(pair_to_idx),
        "angle_edges_cos_theta": angle_edges.tolist(),
        "n_angle_bins": n_angle_bins,
        "cutoff_angstrom": shell_edges[-1],
        "mean_n_atoms": float(np.mean(atom_counts)) if atom_counts else math.nan,
        "mean_nnz_per_row": float(np.mean(nnz_by_row)) if nnz_by_row else math.nan,
        "element_counts": dict(sorted(element_counts.items())),
        "descriptor": "central-element angular environment histogram",
        "n_jobs": n_jobs,
        "parallel_prefer": parallel_prefer,
        "batch_size": batch_size,
    }
    return x, mol_ids, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extxyz", type=Path, default=DEFAULT_EXTXYZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--radial-shell-edges", default=DEFAULT_SHELL_EDGES)
    parser.add_argument("--n-angle-bins", type=int, default=12)
    parser.add_argument("--element-vocab", default="", help="Optional comma-separated central element vocabulary.")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--parallel-prefer", default="processes", choices=["processes", "threads"])
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_file(args.extxyz, "tmQM extxyz")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shell_edges = parse_float_list(args.radial_shell_edges)
    if args.n_angle_bins < 1:
        raise ValueError("--n-angle-bins must be positive")

    if args.element_vocab.strip():
        element_vocab = sorted({x.strip() for x in args.element_vocab.split(",") if x.strip()})
    else:
        log("Collecting central element vocabulary from extxyz")
        element_vocab = collect_element_vocab(args.extxyz, args.max_molecules)
    log(f"Central element vocabulary size: {len(element_vocab)}")

    x, mol_ids, meta = build_matrix(
        extxyz=args.extxyz,
        element_vocab=element_vocab,
        shell_edges=shell_edges,
        n_angle_bins=args.n_angle_bins,
        max_molecules=args.max_molecules,
        progress_every=args.progress_every,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        batch_size=args.batch_size,
    )
    names = feature_names(element_vocab, shell_edges, np.asarray(meta["angle_edges_cos_theta"], dtype=float))
    if len(names) != x.shape[1]:
        raise ValueError(f"Feature name mismatch: names={len(names)} matrix_cols={x.shape[1]}")

    matrix_path = args.out_dir / "tmqm_angular_features.npz"
    ids_path = args.out_dir / "tmqm_angular_mol_ids.csv"
    names_path = args.out_dir / "tmqm_angular_feature_names.json"
    meta_path = args.out_dir / "tmqm_angular_meta.json"

    save_npz(matrix_path, x)
    pd.DataFrame({"mol_id": mol_ids}).to_csv(ids_path, index=False)
    names_path.write_text(json.dumps(names, indent=2) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    log(f"WROTE: {matrix_path}")
    log(f"WROTE: {ids_path}")
    log(f"WROTE: {names_path}")
    log(f"WROTE: {meta_path}")
    log(f"shape={x.shape} nnz={x.nnz} mean_nnz_per_row={meta['mean_nnz_per_row']:.2f}")


if __name__ == "__main__":
    main()
