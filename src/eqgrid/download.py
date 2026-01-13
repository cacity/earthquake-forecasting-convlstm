from __future__ import annotations

import argparse
from pathlib import Path

from .cli_utils import parse_bbox, parse_ymd
from .usgs import download_usgs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default="today")
    p.add_argument("--min-mag", type=float, default=3.5)
    p.add_argument("--bbox", nargs=4, required=True)
    p.add_argument("--segment", type=str, default="yearly", choices=["yearly"])
    p.add_argument("--events-out", type=Path, default=Path("data/interim/events.parquet"))
    args = p.parse_args(argv)

    bbox = parse_bbox(args.bbox)
    start = parse_ymd(args.start)
    end = parse_ymd(args.end)

    df = download_usgs(
        out_dir=args.out_dir,
        start=start,
        end=end,
        bbox=bbox,
        min_mag=args.min_mag,
        segment=args.segment,
    )
    args.events_out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_parquet(args.events_out, index=False)
        return 0

    df.to_parquet(args.events_out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
