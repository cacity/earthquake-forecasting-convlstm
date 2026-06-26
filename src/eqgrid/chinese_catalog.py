"""Load Chinese earthquake catalog from EQT and Excel files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_eqt_file(path: str | Path) -> pd.DataFrame:
    """Parse EQ09 .EQT fixed-width format file.

    Format per line (40 chars):
        Pos 1-14:  datetime YYYYMMDDHHMMSS
        Pos 16-20: latitude (xx.xx)
        Pos 22-27: longitude (xxx.xx)
        Pos 28-31: magnitude (x.xx)
        Pos 32-35: depth code in 0.1 km (9999 means no depth)
        Pos 37:    flag
    """
    path = Path(path)
    records = []

    with open(path, "r", encoding="gbk") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 38:
                continue

            datetime_str = line[1:15]
            lat = float(line[16:21])
            lon = float(line[22:28])
            mag = float(line[28:32])
            depth_code = int(line[32:36])
            flag = line[37] if len(line) > 37 else None

            # Parse datetime with validation using datetime module
            from datetime import datetime
            year = int(datetime_str[:4])
            month = int(datetime_str[4:6])
            day = int(datetime_str[6:8])
            hour = int(datetime_str[8:10])
            minute = int(datetime_str[10:12])
            second = int(datetime_str[12:14])

            # Validate and clamp values
            month = min(max(month, 1), 12)
            day = min(max(day, 1), 31)
            hour = min(hour, 23)
            minute = min(minute, 59)
            second = min(second, 59)

            # Use datetime to handle invalid dates (e.g. Feb 30)
            try:
                dt = datetime(year, month, day, hour, minute, second)
                dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                # Invalid date, use a safe fallback
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                day = min(day, last_day)
                dt = datetime(year, month, day, hour, minute, second)
                dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

            # Build ISO datetime string
            dt_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"

            records.append({
                "time": dt_str,
                "lat": lat,
                "lon": lon,
                "mag": mag,
                "depth_km": None if depth_code == 9999 else depth_code / 10.0,
                "flag": flag,
            })

    return pd.DataFrame(records)


def parse_excel_file(path: str | Path) -> pd.DataFrame:
    """Parse Excel earthquake catalog.

    Expected columns: 日期, 时间, 纬度, 经度, 深度, 震级, 参考地点
    """
    path = Path(path)
    df = pd.read_excel(path)

    # Rename columns (Chinese -> English)
    column_map = {
        "日期": "date",
        "时间": "time_str",
        "纬度": "lat",
        "经度": "lon",
        "深度": "depth_km",
        "震级": "mag",
        "参考地点": "location",
    }
    df = df.rename(columns=column_map)

    # Combine date and time into ISO datetime
    df["time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_str"].astype(str),
        errors="coerce",
    )
    # Drop rows where datetime parsing failed
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Select and reorder columns
    df = df[["time", "lat", "lon", "mag", "depth_km", "location"]].copy()
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    return df


def load_chinese_catalog(
    eqt_path: str | Path = None,
    excel_part1: str | Path = None,
    excel_part2: str | Path = None,
) -> pd.DataFrame:
    """Load and merge all Chinese earthquake catalog sources.

    Args:
        eqt_path: Path to EQ09 .EQT file (1965-2008)
        excel_part1: Path to Excel part 1 (2009+)
        excel_part2: Path to Excel part 2 (2009+)

    Returns:
        DataFrame with columns: time, lat, lon, mag, depth_km
    """
    data_dir = Path("data")

    if eqt_path is None:
        eqt_path = data_dir / "EQ09全国地震目录ML2.EQT"
    if excel_part1 is None:
        excel_part1 = data_dir / "eq_20090101_000000_20251231_235959_xls_xin1_ms_20260306_100558_part1.xlsx"
    if excel_part2 is None:
        excel_part2 = data_dir / "eq_20090101_000000_20251231_235959_xls_xin1_ms_20260306_100558_part2.xlsx"

    # Load EQT file
    print(f"Loading EQT file: {eqt_path}")
    df_eqt = parse_eqt_file(eqt_path)
    print(f"  -> {len(df_eqt)} records")

    # Load Excel files
    print(f"Loading Excel part 1: {excel_part1}")
    df_excel1 = parse_excel_file(excel_part1)
    print(f"  -> {len(df_excel1)} records")

    print(f"Loading Excel part 2: {excel_part2}")
    df_excel2 = parse_excel_file(excel_part2)
    print(f"  -> {len(df_excel2)} records")

    # Combine Excel files
    df_excel = pd.concat([df_excel1, df_excel2], ignore_index=True)

    # Standardize columns for merging
    df_eqt = df_eqt[["time", "lat", "lon", "mag", "depth_km"]].copy()
    df_excel = df_excel[["time", "lat", "lon", "mag", "depth_km"]].copy()

    # Combine all sources
    df_all = pd.concat([df_eqt, df_excel], ignore_index=True)

    # Remove rows with missing critical data
    df_all = df_all.dropna(subset=["time", "lat", "lon", "mag"])

    # Sort by time
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all = df_all.sort_values("time").reset_index(drop=True)

    print(f"\nTotal combined records: {len(df_all)}")
    print(f"Time range: {df_all['time'].min()} to {df_all['time'].max()}")

    return df_all


if __name__ == "__main__":
    df = load_chinese_catalog()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 records:")
    print(df.head())
    print("\nLast 5 records:")
    print(df.tail())
