from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def summarize_ts(ts: np.ndarray) -> dict[str, Any]:
    if ts.size == 0:
        return {"n": 0, "min": None, "max": None}
    ts_pd = pd.to_datetime(ts, utc=True)
    return {
        "n": int(ts.size),
        "min": ts_pd.min().isoformat(),
        "max": ts_pd.max().isoformat(),
    }


def load_split_ts(splits_dir: Path) -> dict[str, np.ndarray]:
    return {
        "train": np.load(splits_dir / "train_TS.npy"),
        "val": np.load(splits_dir / "val_TS.npy"),
        "test": np.load(splits_dir / "test_TS.npy"),
    }


def assert_no_overlap(ts_by_split: dict[str, np.ndarray]) -> None:
    train_ts = ts_by_split["train"]
    val_ts = ts_by_split["val"]
    test_ts = ts_by_split["test"]
    if train_ts.size and val_ts.size:
        if not (pd.to_datetime(train_ts, utc=True).max() < pd.to_datetime(val_ts, utc=True).min()):
            raise ValueError("overlap/leakage: max(train.TS) must be < min(val.TS)")
    if val_ts.size and test_ts.size:
        if not (pd.to_datetime(val_ts, utc=True).max() < pd.to_datetime(test_ts, utc=True).min()):
            raise ValueError("overlap/leakage: max(val.TS) must be < min(test.TS)")


def load_meta(splits_dir: Path) -> dict[str, Any] | None:
    meta_path = splits_dir / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def print_split_summary(splits_dir: Path, require_boundaries_in_ts: bool = False) -> None:
    ts_by = load_split_ts(splits_dir)
    train_s = summarize_ts(ts_by["train"])
    val_s = summarize_ts(ts_by["val"])
    test_s = summarize_ts(ts_by["test"])
    print(f"train: n={train_s['n']} min={train_s['min']} max={train_s['max']}")
    print(f"val:   n={val_s['n']} min={val_s['min']} max={val_s['max']}")
    print(f"test:  n={test_s['n']} min={test_s['min']} max={test_s['max']}")

    meta = load_meta(splits_dir)
    if meta is not None:
        te = meta.get("train_end_resolved")
        ve = meta.get("val_end_resolved")
        tin = meta.get("train_end_in_TS")
        vin = meta.get("val_end_in_TS")
        if te is not None and ve is not None:
            print(f"train_end_resolved: {te} (in_TS={tin})")
            print(f"val_end_resolved:   {ve} (in_TS={vin})")
        if require_boundaries_in_ts and ((tin is False) or (vin is False)):
            raise ValueError("split boundaries are not aligned to TS bins (train_end_in_TS/val_end_in_TS is false)")

    assert_no_overlap(ts_by)

