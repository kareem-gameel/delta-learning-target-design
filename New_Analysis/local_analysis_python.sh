#!/usr/bin/env bash
# Stable local Python launcher for New_Analysis scripts on Kareem's Mac.
#
# Why this exists:
# - The bundled Codex Python is intentionally lightweight and lacks scipy,
#   sklearn, joblib, and matplotlib.
# - The global miniconda/base Python is fragile on this Mac: NumPy/OpenBLAS can
#   miss libgfortran, while forcing DYLD_LIBRARY_PATH can break SciPy/Accelerate.
# - The local fair-chem conda env has the needed packages and imports cleanly.
# - User-site package leakage can still break imports, so it is disabled.
#
# Usage:
#   New_Analysis/local_analysis_python.sh path/to/script.py [args...]
#   New_Analysis/local_analysis_python.sh - <<'PY'
#   import numpy, pandas, scipy, sklearn, joblib, matplotlib
#   print("ok")
#   PY

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/kareem/miniconda3/envs/fair-chem/bin/python}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"

mkdir -p "$MPLCONFIGDIR"

exec "$PYTHON_BIN" "$@"
