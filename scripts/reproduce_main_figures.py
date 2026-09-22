#!/usr/bin/env python3
"""Regenerate all corrected main-manuscript figures from frozen test data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"


def run(script: str) -> None:
    subprocess.run([sys.executable, str(MANUSCRIPT / "scripts" / script)], check=True)


def main() -> None:
    run("generate_corrected_fig01.py")
    run("generate_figure2.py")
    run("generate_figure3.py")
    run("generate_figure4.py")
    outputs = [
        "fig01_reference_correction.png",
        "fig02_residual_design.png",
        "fig04_roughness_nmae.png",
        "fig03_qm9_transfer.png",
    ]
    for name in outputs:
        path = MANUSCRIPT / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
