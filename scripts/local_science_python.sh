#!/usr/bin/env bash
set -euo pipefail

# Clean local scientific Python runner for this repo.
#
# Why this exists:
# - The miniconda base Python is fragile on this Mac: NumPy/OpenBLAS can miss
#   libgfortran, and adding DYLD_LIBRARY_PATH can break SciPy/Accelerate.
# - The local fair-chem conda env has the scientific stack needed for this repo.
# - PYTHONNOUSERSITE prevents accidental mixing of ~/.local packages with
#   conda packages, which previously caused sklearn/pyarrow import failures.
# - MPLCONFIGDIR keeps matplotlib from trying to write to ~/.matplotlib.

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MPLCONFIGDIR="${TMPDIR:-/tmp}/deltaml_matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec /Users/kareem/miniconda3/envs/fair-chem/bin/python "$@"
