"""Minimal example for the public manuscript workflow.

This example assumes that the user has already placed local CENC catalog files
under data/private. The catalog data are not distributed with this repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    data_dir = Path("data/private")
    processed_dir = Path("data/processed_china_10ch_d0_70_corrected")

    if not data_dir.exists():
        raise SystemExit(
            "Missing data/private. Place local CENC catalog files there before running this example."
        )

    run(
        [
            sys.executable,
            "scripts/build_china_depth_filtered_tensors.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(processed_dir),
            "--bbox",
            "95",
            "110",
            "22",
            "35",
            "--start",
            "2000-01-01",
            "--end-exclusive",
            "2026-01-01",
            "--bins-end",
            "2025-12-01",
            "--min-mag",
            "4.0",
            "--depth-max",
            "70",
        ]
    )

    print(f"Wrote tensors to {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
