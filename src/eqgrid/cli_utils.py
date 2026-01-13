from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def parse_ymd(s: str) -> datetime:
    if s.lower() == "today":
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class BBox:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


def parse_bbox(args: list[str]) -> BBox:
    if len(args) != 4:
        raise ValueError("bbox requires 4 numbers: min_lon max_lon min_lat max_lat")
    min_lon, max_lon, min_lat, max_lat = map(float, args)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("invalid bbox ordering")
    return BBox(min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat)
