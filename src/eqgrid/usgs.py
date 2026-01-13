from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tqdm import tqdm

from .cli_utils import BBox


USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MAX_LIMIT = 20000


@dataclass(frozen=True)
class Segment:
    start: datetime
    end: datetime
    label: str


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def yearly_segments(start: datetime, end: datetime) -> list[Segment]:
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    segments: list[Segment] = []
    year = start.year
    while year <= end.year:
        seg_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        seg_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if year == start.year:
            seg_start = start
        if year == end.year:
            seg_end = end
        if seg_start < seg_end:
            segments.append(Segment(start=seg_start, end=seg_end, label=f"{year}"))
        year += 1
    return segments


def _request_geojson(
    session: requests.Session, params: dict[str, Any], timeout_s: int = 60
) -> dict[str, Any]:
    resp = session.get(USGS_ENDPOINT, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _fetch_segment_all_pages(
    session: requests.Session,
    bbox: BBox,
    start: datetime,
    end: datetime,
    min_mag: float,
    out_dir: Path,
    label: str,
    sleep_s: float = 0.2,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_params: dict[str, Any] = {
        "format": "geojson",
        "starttime": _to_iso(start),
        "endtime": _to_iso(end),
        "minmagnitude": min_mag,
        "minlatitude": bbox.min_lat,
        "maxlatitude": bbox.max_lat,
        "minlongitude": bbox.min_lon,
        "maxlongitude": bbox.max_lon,
        "orderby": "time-asc",
        "limit": MAX_LIMIT,
        "offset": 1,
    }
    head = _request_geojson(session, {**base_params, "limit": 1, "offset": 1})
    total = int(head.get("metadata", {}).get("count", 0))
    if total == 0:
        return []

    n_pages = math.ceil(total / MAX_LIMIT)
    paths: list[Path] = []
    for page in range(n_pages):
        offset = page * MAX_LIMIT + 1
        payload = _request_geojson(session, {**base_params, "offset": offset})
        out_path = out_dir / f"{label}_offset{offset:06d}.geojson"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(out_path)
        if sleep_s:
            time.sleep(sleep_s)
    return paths


def _parse_geojson_files(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            fid = feat.get("id")
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None, None]
            lon, lat = coords[0], coords[1]
            depth = coords[2] if len(coords) > 2 else None
            mag = props.get("mag")
            t_ms = props.get("time")
            if t_ms is None:
                continue
            t = pd.to_datetime(int(t_ms), unit="ms", utc=True)
            rows.append(
                {
                    "time": t,
                    "lon": lon,
                    "lat": lat,
                    "mag": mag,
                    "depth": depth,
                    "id": fid,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["time", "lon", "lat", "mag"]).copy()
    df["lon"] = df["lon"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["mag"] = df["mag"].astype(float)
    if "depth" in df.columns:
        df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["id"] = df["id"].astype("string")
    return df


def _approx_id(df: pd.DataFrame) -> pd.Series:
    t = df["time"].dt.strftime("%Y%m%d%H%M%S")
    lon = df["lon"].round(5).astype(str)
    lat = df["lat"].round(5).astype(str)
    mag = df["mag"].round(2).astype(str)
    return "approx_" + t + "_" + lon + "_" + lat + "_" + mag


def download_usgs(
    out_dir: Path,
    start: datetime,
    end: datetime,
    bbox: BBox,
    min_mag: float,
    segment: str = "yearly",
) -> pd.DataFrame:
    if segment != "yearly":
        raise ValueError("only segment=yearly is implemented")

    segs = yearly_segments(start, end)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = []
    with requests.Session() as session:
        for seg in tqdm(segs, desc="segments"):
            seg_dir = out_dir / seg.label
            paths = _fetch_segment_all_pages(
                session=session,
                bbox=bbox,
                start=seg.start,
                end=seg.end,
                min_mag=min_mag,
                out_dir=seg_dir,
                label=seg.label,
            )
            all_paths.extend(paths)

    df = _parse_geojson_files(all_paths)
    if df.empty:
        return df

    missing = df["id"].isna() | (df["id"].str.len() == 0)
    if missing.any():
        df.loc[missing, "id"] = _approx_id(df.loc[missing])
    df = df.drop_duplicates(subset=["id"]).sort_values("time").reset_index(drop=True)
    return df
