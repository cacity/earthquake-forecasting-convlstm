from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=Path, required=True)
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path)
    args = p.parse_args(argv)

    X = np.load(args.x, mmap_mode="r")
    Y = np.load(args.y, mmap_mode="r")
    bins = np.load(args.bins, mmap_mode="r")

    print(f"X shape: {tuple(X.shape)} dtype={X.dtype}")
    print(f"Y shape: {tuple(Y.shape)} dtype={Y.dtype}")
    print(f"bins: {bins.shape} start={bins[0]} end={bins[-1]}")
    if X.shape[0] != Y.shape[0] or X.shape[0] != bins.shape[0]:
        raise SystemExit("T mismatch across X/Y/bins")

    if args.splits_dir is not None:
        try:
            from .split_sanity import print_split_summary
        except ImportError:  # allow running as a plain script
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from eqgrid.split_sanity import print_split_summary

        try:
            print_split_summary(args.splits_dir)
        except ValueError as e:
            raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
