from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cli_utils import BBox


@dataclass(frozen=True)
class GridMeta:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    resolution_deg: float
    n_lon: int
    n_lat: int
    M: int


def build_grid_meta(bbox: BBox, resolution_deg: float = 1.0) -> GridMeta:
    n_lon = int(math.ceil((bbox.max_lon - bbox.min_lon) / resolution_deg))
    n_lat = int(math.ceil((bbox.max_lat - bbox.min_lat) / resolution_deg))
    return GridMeta(
        min_lon=bbox.min_lon,
        max_lon=bbox.max_lon,
        min_lat=bbox.min_lat,
        max_lat=bbox.max_lat,
        resolution_deg=resolution_deg,
        n_lon=n_lon,
        n_lat=n_lat,
        M=n_lon * n_lat,
    )


def add_grid_index(df: pd.DataFrame, meta: GridMeta) -> pd.DataFrame:
    df = df.copy()
    lon = df["lon"].astype(float)
    lat = df["lat"].astype(float)
    in_bbox = (
        (lon >= meta.min_lon)
        & (lon < meta.max_lon)
        & (lat >= meta.min_lat)
        & (lat < meta.max_lat)
    )
    df = df.loc[in_bbox].copy()
    if df.empty:
        df["lon_idx"] = pd.Series(dtype="int64")
        df["lat_idx"] = pd.Series(dtype="int64")
        df["cell_id"] = pd.Series(dtype="int64")
        return df

    df["lon_idx"] = np.floor((df["lon"] - meta.min_lon) / meta.resolution_deg).astype(
        "int64"
    )
    df["lat_idx"] = np.floor((df["lat"] - meta.min_lat) / meta.resolution_deg).astype(
        "int64"
    )
    df = df.loc[
        (df["lon_idx"] >= 0)
        & (df["lon_idx"] < meta.n_lon)
        & (df["lat_idx"] >= 0)
        & (df["lat_idx"] < meta.n_lat)
    ].copy()
    df["cell_id"] = df["lat_idx"] * meta.n_lon + df["lon_idx"]
    return df


def _month_start_utc(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, utc=True)
    t_naive = t.dt.tz_convert("UTC").dt.tz_localize(None)
    month_start_naive = t_naive.dt.to_period("M").dt.start_time
    return month_start_naive.dt.tz_localize("UTC")


def _week_start_utc(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, utc=True).dt.tz_convert("UTC")
    day = t.dt.floor("D")
    return day - pd.to_timedelta(day.dt.weekday, unit="D")

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _align_to_bin_start(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    ts = _ensure_utc(ts)
    if freq == "monthly":
        ts_naive = ts.tz_convert("UTC").tz_localize(None)
        out_naive = ts_naive.to_period("M").start_time
        return _ensure_utc(out_naive)
    if freq == "weekly":
        day = ts.floor("D")
        return day - pd.Timedelta(days=day.weekday())
    raise ValueError("freq must be monthly or weekly")


def add_time_bin(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    if freq == "monthly":
        df["time_bin"] = _month_start_utc(df["time"])
    elif freq == "weekly":
        df["time_bin"] = _week_start_utc(df["time"])
    else:
        raise ValueError("freq must be monthly or weekly")
    return df


def build_bins(freq: str, start: str = "2000-01-01", end: str = "today") -> np.ndarray:
    start_ts = _align_to_bin_start(pd.Timestamp(start), freq)
    if end == "today":
        end_ts = _align_to_bin_start(pd.Timestamp.now(tz="UTC"), freq)
    else:
        end_ts = _align_to_bin_start(pd.Timestamp(end), freq)

    if freq == "monthly":
        date_index = pd.date_range(start_ts, end_ts, freq="MS", tz="UTC")
    elif freq == "weekly":
        date_index = pd.date_range(start_ts, end_ts, freq="W-MON", tz="UTC")
    else:
        raise ValueError("freq must be monthly or weekly")

    return date_index.to_numpy(dtype="datetime64[ns]")


@dataclass(frozen=True)
class TensorBuildConfig:
    freq: str = "monthly"
    count_clip: float = 10.0
    mag_scale: float = 10.0


def build_tensors(
    events: pd.DataFrame,
    grid: GridMeta,
    bins: np.ndarray,
    config: TensorBuildConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if events.empty:
        T = len(bins)
        X = np.zeros((T, 3, grid.n_lat, grid.n_lon), dtype=np.float32)
        Y = np.zeros((T, 1, grid.n_lat, grid.n_lon), dtype=np.float32)
        return X, Y

    if not {"lon_idx", "lat_idx", "cell_id"}.issubset(events.columns):
        events = add_grid_index(events, grid)
    else:
        events = events.copy()
        events = events.loc[
            (events["lon_idx"] >= 0)
            & (events["lon_idx"] < grid.n_lon)
            & (events["lat_idx"] >= 0)
            & (events["lat_idx"] < grid.n_lat)
        ].copy()
    if "time_bin" not in events.columns:
        events = add_time_bin(events, config.freq)
    if events.empty:
        T = len(bins)
        X = np.zeros((T, 3, grid.n_lat, grid.n_lon), dtype=np.float32)
        Y = np.zeros((T, 1, grid.n_lat, grid.n_lon), dtype=np.float32)
        return X, Y

    events = events.copy()
    events["mag"] = pd.to_numeric(events["mag"], errors="coerce")
    events = events.dropna(subset=["mag", "lon_idx", "lat_idx", "time_bin"]).copy()

    bin_index = pd.DatetimeIndex(pd.to_datetime(bins, utc=True))
    events["time_bin"] = pd.to_datetime(events["time_bin"], utc=True)
    t_idx = bin_index.get_indexer(events["time_bin"])
    events = events.loc[t_idx >= 0].copy()
    if events.empty:
        T = len(bins)
        X = np.zeros((T, 3, grid.n_lat, grid.n_lon), dtype=np.float32)
        Y = np.zeros((T, 1, grid.n_lat, grid.n_lon), dtype=np.float32)
        return X, Y

    t_index = t_idx[t_idx >= 0].astype("int64")
    y = events["lat_idx"].astype("int64").to_numpy()
    x = events["lon_idx"].astype("int64").to_numpy()
    mag = events["mag"].astype(float).to_numpy()
    finite = np.isfinite(mag)
    if not finite.all():
        t_index = t_index[finite]
        y = y[finite]
        x = x[finite]
        mag = mag[finite]

    T = len(bins)
    count = np.zeros((T, grid.n_lat, grid.n_lon), dtype=np.float32)
    max_mag = np.zeros((T, grid.n_lat, grid.n_lon), dtype=np.float32)
    sum_energy = np.zeros((T, grid.n_lat, grid.n_lon), dtype=np.float64)

    np.add.at(count, (t_index, y, x), 1.0)
    np.maximum.at(max_mag, (t_index, y, x), mag.astype(np.float32))

    mag_energy = np.clip(mag, -2.0, 10.0)
    energy = np.power(10.0, 1.5 * mag_energy).astype(np.float64)
    np.add.at(sum_energy, (t_index, y, x), energy)

    Y = (count > 0).astype(np.float32)[:, None, :, :]

    count_norm = np.clip(count, 0.0, config.count_clip) / float(config.count_clip)
    max_mag_norm = max_mag / float(config.mag_scale)
    sum_energy_norm = np.log1p(sum_energy).astype(np.float32)

    X = np.stack([count_norm, max_mag_norm, sum_energy_norm], axis=1).astype(np.float32)
    return X, Y


def save_grid_meta(path: Path, meta: GridMeta, config: TensorBuildConfig) -> None:
    payload = {"grid": asdict(meta), "tensor_config": asdict(config)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
