"""Build 10-channel China tensors with corrected depth filtering.

This script is intended for the CENC catalog workflow used by the manuscript.
It applies two catalog rules before tensor construction:

1. EQT fixed-width records before 2009 store depth in deci-kilometers, so the
   raw depth code is divided by 10.
2. From 2009 onward, the table/parquet catalog is used instead of the EQT
   overlap to avoid duplicate source mixing.

Depth filtering keeps depth=0 records because in this catalog they usually
represent unspecified depth rather than quarry blasts or mining events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eqgrid.cli_utils import BBox
from eqgrid.tensors import (
    TensorBuildConfig,
    add_grid_index,
    add_time_bin,
    build_bins,
    build_grid_meta,
    save_grid_meta,
)
from eqgrid.tensors_enhanced import build_tensors_enhanced


def parse_eqt(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="gbk", errors="replace") as f:
        for line in f:
            if len(line) < 37:
                continue
            try:
                dt = line[1:15]
                time = pd.Timestamp(
                    int(dt[:4]),
                    int(dt[4:6]),
                    int(dt[6:8]),
                    int(dt[8:10]),
                    int(dt[10:12]),
                    int(dt[12:14]),
                )
                depth_code = int(line[32:36])
                depth_km = np.nan if depth_code == 9999 else depth_code / 10.0
                rows.append(
                    {
                        "time": time,
                        "lat": float(line[16:21]),
                        "lon": float(line[22:28]),
                        "mag": float(line[28:32]),
                        "depth_km": depth_km,
                        "source": "EQ09_fixed_width_pre2009",
                        "depth_code_raw": depth_code,
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def parse_parquet_parts(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("eq_20090101_*.parquet")):
        df = pd.read_parquet(path)
        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(df["origin_time"]),
                    "lat": pd.to_numeric(df["lat"], errors="coerce"),
                    "lon": pd.to_numeric(df["lon"], errors="coerce"),
                    "mag": pd.to_numeric(df["mag"], errors="coerce"),
                    "depth_km": pd.to_numeric(df["depth"], errors="coerce"),
                    "source": path.name,
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No eq_20090101_*.parquet files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    keep: list[int] = []
    for idx, row in df.iterrows():
        duplicate = False
        j = len(keep) - 1
        while j >= 0:
            prev = df.loc[keep[j]]
            dt = abs((row["time"] - prev["time"]).total_seconds())
            if dt > 10:
                break
            if (
                abs(row["lat"] - prev["lat"]) <= 0.01
                and abs(row["lon"] - prev["lon"]) <= 0.01
                and abs(row["mag"] - prev["mag"]) <= 0.1
            ):
                duplicate = True
                break
            j -= 1
        if not duplicate:
            keep.append(idx)
    return df.loc[keep].copy().reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, default=[95, 110, 22, 35])
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end-exclusive", default="2026-01-01")
    parser.add_argument("--bins-end", default="2025-12-01")
    parser.add_argument("--min-mag", type=float, default=4.0)
    parser.add_argument("--depth-max", type=float, default=70.0)
    args = parser.parse_args(argv)

    bbox = BBox(*args.bbox)
    start = pd.Timestamp(args.start)
    end_exclusive = pd.Timestamp(args.end_exclusive)

    eqt_candidates = sorted(args.data_dir.glob("*.EQT"))
    if not eqt_candidates:
        raise FileNotFoundError(f"No EQT file found in {args.data_dir}")
    eqt = parse_eqt(eqt_candidates[0])
    eqt = eqt[eqt["time"] < pd.Timestamp("2009-01-01")]

    post = parse_parquet_parts(args.data_dir)
    post = post[post["time"] >= pd.Timestamp("2009-01-01")]
    events = pd.concat([eqt, post], ignore_index=True)
    events = events.dropna(subset=["time", "lat", "lon", "mag"])
    events = events[
        (events["time"] >= start)
        & (events["time"] < end_exclusive)
        & (events["lon"] >= bbox.min_lon)
        & (events["lon"] < bbox.max_lon)
        & (events["lat"] >= bbox.min_lat)
        & (events["lat"] < bbox.max_lat)
        & (events["mag"] >= args.min_mag)
    ].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out_dir / "events_study_raw_before_dedup.parquet", index=False)

    deduped = deduplicate(events)
    deduped.to_parquet(
        args.out_dir / "events_study_dedup_before_depth_filter.parquet", index=False
    )

    filtered = deduped[
        deduped["depth_km"].isna()
        | ((deduped["depth_km"] >= 0) & (deduped["depth_km"] <= args.depth_max))
    ].copy()
    filtered = filtered[["time", "lat", "lon", "mag", "depth_km", "source"]]
    filtered.to_parquet(args.out_dir / "events_filtered_before_index.parquet", index=False)

    grid = build_grid_meta(bbox, resolution_deg=1.0)
    bins = build_bins(freq="monthly", start=args.start, end=args.bins_end)
    config = TensorBuildConfig(freq="monthly", count_clip=10.0, mag_scale=10.0)
    indexed = add_time_bin(add_grid_index(filtered, grid), "monthly")
    indexed.to_parquet(args.out_dir / "events_indexed.parquet", index=False)

    X, Y = build_tensors_enhanced(indexed, grid, bins, config)
    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Y.npy", Y)
    np.save(args.out_dir / "bins.npy", bins)
    save_grid_meta(args.out_dir / "grid_meta.json", grid, config)

    valid_depth = filtered[filtered["depth_km"].notna()]
    summary = {
        "source_policy": (
            "EQT fixed-width catalog before 2009 with depth_code/10; "
            "XLS/parquet table catalog from 2009 onward"
        ),
        "filter": (
            f"M>={args.min_mag}, include missing depth and "
            f"0<=depth_km<={args.depth_max}; exclude known depth>{args.depth_max}"
        ),
        "raw_study_events_before_dedup": int(len(events)),
        "dedup_study_events_before_depth_filter": int(len(deduped)),
        "events_after_depth_filter": int(len(filtered)),
        "events_removed_known_depth_gt70": int((deduped["depth_km"] > args.depth_max).sum()),
        "depth_after_filter": {
            "min": float(valid_depth["depth_km"].min()) if len(valid_depth) else None,
            "max": float(valid_depth["depth_km"].max()) if len(valid_depth) else None,
            "median": float(valid_depth["depth_km"].median()) if len(valid_depth) else None,
            "zero_count": int((valid_depth["depth_km"] == 0).sum()),
            "missing_count": int(filtered["depth_km"].isna().sum()),
        },
        "mag_min": float(filtered["mag"].min()),
        "mag_max": float(filtered["mag"].max()),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "positive_cell_months_all": int(Y.sum()),
    }
    (args.out_dir / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
