from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleConfig:
    lookback: int
    horizon: int


def make_samples(
    X: np.ndarray, Y: np.ndarray, bins: np.ndarray, config: SampleConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised samples with strict time ordering (no label leakage).

    Conventions:
    - X is aligned to time bins: X[t] uses events aggregated inside bin t.
    - Each sample i predicts a target bin t_target that is strictly AFTER the last input bin.

    For lookback=L and horizon=H:
    - input bins:  [t0-L, ..., t0-1]  (length L)
    - target bin:  t_target = t0 + (H - 1)
      - H=1 means "next bin" relative to the last observed bin.
    - TS records bins[t_target].
    """
    if X.ndim != 4:
        raise ValueError("X must be [T,C,H,W]")
    if Y.ndim != 4:
        raise ValueError("Y must be [T,1,H,W]")
    if X.shape[0] != Y.shape[0] or X.shape[0] != bins.shape[0]:
        raise ValueError("X/Y/bins must share T")

    T = X.shape[0]
    L = int(config.lookback)
    H = int(config.horizon)
    if L <= 0 or H <= 0:
        raise ValueError("lookback/horizon must be positive")

    n = T - (L + H - 1)
    if n <= 0:
        raise ValueError("not enough time bins for given lookback/horizon")

    xs = np.zeros((n, L, *X.shape[1:]), dtype=X.dtype)
    ys = np.zeros((n, *Y.shape[1:]), dtype=Y.dtype)
    ts = np.zeros((n,), dtype=bins.dtype)

    for i in range(n):
        t0 = i + L
        t1 = t0 + H - 1
        if t1 <= (t0 - 1):
            raise RuntimeError("invalid alignment: target must be after last input bin")
        xs[i] = X[t0 - L : t0]
        ys[i] = Y[t1]
        ts[i] = bins[t1]

    return xs, ys, ts


def save_samples(out_dir: Path, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, cfg: SampleConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "XS.npy", xs)
    np.save(out_dir / "YS.npy", ys)
    np.save(out_dir / "TS.npy", ts)
    (out_dir / "meta.json").write_text(
        json.dumps({"samples": asdict(cfg), "shapes": {"XS": list(xs.shape), "YS": list(ys.shape)}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
