"""Build tensors from Chinese earthquake catalog.

This script loads the Chinese earthquake catalog (EQT + Excel files),
builds the grid, and generates X.npy, Y.npy, and bins.npy.

Usage:
    python -m eqgrid.build_tensors_chinese \
        --out-dir data/processed_china \
        --bbox 95 110 22 35 \
        --freq monthly \
        --bins-start 1965-01-01 \
        --bins-end today
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .chinese_catalog import load_chinese_catalog
from .cli_utils import BBox, parse_bbox
from .tensors import (
    TensorBuildConfig,
    add_grid_index,
    add_time_bin,
    build_bins,
    build_grid_meta,
    build_tensors,
    save_grid_meta,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build tensors from Chinese earthquake catalog")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--bbox", nargs=4, type=float, default=[95, 110, 22, 35],
                   help="Bounding box: min_lon max_lon min_lat max_lat (default: 95 110 22 35)")
    p.add_argument("--freq", type=str, default="monthly", choices=["monthly", "weekly"])
    p.add_argument("--bins-start", type=str, default="1965-01-01")
    p.add_argument("--bins-end", type=str, default="today")
    p.add_argument("--count-clip", type=float, default=10.0)
    p.add_argument("--mag-scale", type=float, default=10.0)
    p.add_argument("--min-mag", type=float, default=None,
                   help="Minimum magnitude filter (optional)")
    args = p.parse_args(argv)

    # Load Chinese earthquake catalog
    print("Loading Chinese earthquake catalog...")
    events = load_chinese_catalog()
    print(f"Loaded {len(events)} events")

    # Filter by magnitude if specified
    if args.min_mag is not None:
        events = events[events["mag"] >= args.min_mag].copy()
        print(f"After magnitude >= {args.min_mag}: {len(events)} events")

    # Build grid and bins
    bbox = BBox(*args.bbox)
    grid = build_grid_meta(bbox, resolution_deg=1.0)
    bins = build_bins(freq=args.freq, start=args.bins_start, end=args.bins_end)
    config = TensorBuildConfig(
        freq=args.freq, count_clip=args.count_clip, mag_scale=args.mag_scale
    )

    print(f"\nGrid: {grid.n_lon} x {grid.n_lat} cells")
    print(f"Lon: {grid.min_lon} to {grid.max_lon}")
    print(f"Lat: {grid.min_lat} to {grid.max_lat}")
    print(f"Bins: {len(bins)} ({bins[0]} to {bins[-1]})")

    # Build tensors
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_indexed = add_time_bin(add_grid_index(events, grid), args.freq)
    if not events_indexed.empty:
        events_indexed.to_parquet(args.out_dir / "events_indexed.parquet", index=False)
        print(f"\nEvents within bbox: {len(events_indexed)}")

    print("\nBuilding tensors...")
    X, Y = build_tensors(events=events_indexed, grid=grid, bins=bins, config=config)

    # Save outputs
    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Y.npy", Y)
    np.save(args.out_dir / "bins.npy", bins)
    save_grid_meta(args.out_dir / "grid_meta.json", grid, config)

    print(f"\nSaved to {args.out_dir}:")
    print(f"  X.npy: {X.shape}")
    print(f"  Y.npy: {Y.shape}")
    print(f"  bins.npy: {len(bins)}")
    print(f"  grid_meta.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
