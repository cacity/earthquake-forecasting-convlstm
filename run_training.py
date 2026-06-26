#!/usr/bin/env python
"""Convenience runner for the public CENC ten-channel workflow.

The script does not download or ship earthquake catalog data. Place local CENC
catalog files under an ignored directory such as data/private before running it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/private"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_china_10ch_d0_70_corrected"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/convlstm_china_10ch_d0_70_corrected"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples_dir = args.processed_dir / "samples_L12_H1"
    splits_dir = args.processed_dir / "splits_L12_H1"

    run(
        [
            sys.executable,
            "scripts/build_china_depth_filtered_tensors.py",
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(args.processed_dir),
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

    run(
        [
            sys.executable,
            "-m",
            "eqgrid.make_samples",
            "--x",
            str(args.processed_dir / "X.npy"),
            "--y",
            str(args.processed_dir / "Y.npy"),
            "--bins",
            str(args.processed_dir / "bins.npy"),
            "--out-dir",
            str(samples_dir),
            "--lookback",
            "12",
            "--horizon",
            "1",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "eqgrid.make_splits",
            "--samples-dir",
            str(samples_dir),
            "--out-dir",
            str(splits_dir),
            "--train-end",
            "2018-12-01",
            "--val-end",
            "2022-12-01",
            "--require-non-empty",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "eqgrid.train",
            "--splits-dir",
            str(splits_dir),
            "--out-dir",
            str(args.output_dir),
            "--model",
            "convlstm",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "16",
            "--lr",
            "0.001",
            "--weight-decay",
            "0.01",
            "--grad-clip",
            "1.0",
            "--patience",
            "5",
            "--hidden-channels",
            "64",
            "--num-layers",
            "1",
            "--pos-weight-cap",
            "100",
            "--threshold-scan",
            "0.01:0.95:0.01",
            "--seed",
            str(args.seed),
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
