from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    val_end: str


def time_splits(ts: np.ndarray, cfg: SplitConfig) -> dict[str, np.ndarray]:
    ts_pd = pd.to_datetime(ts, utc=True)
    train_end = pd.Timestamp(cfg.train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.val_end, tz="UTC")
    if not (train_end < val_end):
        raise ValueError("train_end must be < val_end")

    train_idx = np.where(ts_pd <= train_end)[0]
    val_idx = np.where((ts_pd > train_end) & (ts_pd <= val_end))[0]
    test_idx = np.where(ts_pd > val_end)[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(
    out_dir: Path,
    XS: np.ndarray,
    YS: np.ndarray,
    TS: np.ndarray,
    splits: dict[str, np.ndarray],
    cfg: SplitConfig,
    require_non_empty: tuple[str, ...] = (),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_pd_all = pd.to_datetime(TS, utc=True)
    unique_ts = pd.DatetimeIndex(ts_pd_all.unique()).sort_values()
    train_end_resolved = pd.Timestamp(cfg.train_end, tz="UTC")
    val_end_resolved = pd.Timestamp(cfg.val_end, tz="UTC")
    split_meta: dict[str, dict[str, object]] = {}
    for k, idx in splits.items():
        np.save(out_dir / f"{k}_X.npy", XS[idx])
        np.save(out_dir / f"{k}_Y.npy", YS[idx])
        np.save(out_dir / f"{k}_TS.npy", TS[idx])
        np.save(out_dir / f"{k}_idx.npy", idx.astype(np.int64))
        if len(idx) == 0:
            split_meta[k] = {"n_samples": 0, "ts_min": None, "ts_max": None}
        else:
            ts_pd = ts_pd_all[idx]
            split_meta[k] = {
                "n_samples": int(len(idx)),
                "ts_min": ts_pd.min().isoformat(),
                "ts_max": ts_pd.max().isoformat(),
            }
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "split": asdict(cfg),
                "train_end_resolved": train_end_resolved.isoformat(),
                "val_end_resolved": val_end_resolved.isoformat(),
                "train_end_in_TS": bool(train_end_resolved in unique_ts),
                "val_end_in_TS": bool(val_end_resolved in unique_ts),
                "boundary_semantics": {
                    "train": "TS <= train_end",
                    "val": "train_end < TS <= val_end",
                    "test": "TS > val_end",
                },
                "splits": split_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for k in require_non_empty:
        if int(split_meta.get(k, {}).get("n_samples", 0)) == 0:
            raise ValueError(f"required split is empty: {k}")
