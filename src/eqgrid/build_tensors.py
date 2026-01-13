from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cli_utils import parse_bbox
from .tensors import (
    TensorBuildConfig,
    add_grid_index,
    add_time_bin,
    build_bins,
    build_grid_meta,
    build_tensors,
    save_grid_meta,
)


def _load_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    raise ValueError("events must be .parquet or .csv")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bbox", nargs=4, required=True)
    p.add_argument("--freq", type=str, default="monthly", choices=["monthly", "weekly"])
    p.add_argument("--bins-start", type=str, default="2000-01-01")
    p.add_argument("--bins-end", type=str, default="today")
    p.add_argument("--count-clip", type=float, default=10.0)
    p.add_argument("--mag-scale", type=float, default=10.0)
    args = p.parse_args(argv)

    bbox = parse_bbox(args.bbox)
    grid = build_grid_meta(bbox, resolution_deg=1.0)
    bins = build_bins(freq=args.freq, start=args.bins_start, end=args.bins_end)
    config = TensorBuildConfig(
        freq=args.freq, count_clip=args.count_clip, mag_scale=args.mag_scale
    )

    events = _load_events(args.events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_indexed = add_time_bin(add_grid_index(events, grid), args.freq)
    if not events_indexed.empty:
        events_indexed.to_parquet(args.out_dir / "events_indexed.parquet", index=False)
    X, Y = build_tensors(events=events_indexed, grid=grid, bins=bins, config=config)
    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Y.npy", Y)
    np.save(args.out_dir / "bins.npy", bins)
    save_grid_meta(args.out_dir / "grid_meta.json", grid, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
