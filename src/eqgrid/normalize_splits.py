from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _channel_stats_train(train_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    if train_x.ndim != 5:
        raise ValueError("train_X must be [N,L,C,H,W]")
    # stats over N,L,H,W per channel
    mean = train_x.mean(axis=(0, 1, 3, 4), keepdims=False).astype(np.float32)
    std = train_x.std(axis=(0, 1, 3, 4), keepdims=False).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


def _apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.ndim != 5:
        raise ValueError("X must be [N,L,C,H,W]")
    mean_b = mean.reshape(1, 1, -1, 1, 1)
    std_b = std.reshape(1, 1, -1, 1, 1)
    return ((x - mean_b) / std_b).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args(argv)

    train_x = np.load(args.splits_dir / "train_X.npy")
    mean, std = _channel_stats_train(train_x, eps=float(args.eps))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stats.json").write_text(
        json.dumps(
            {"mean": mean.tolist(), "std": std.tolist(), "eps": float(args.eps)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for split in ["train", "val", "test"]:
        x = np.load(args.splits_dir / f"{split}_X.npy")
        y = np.load(args.splits_dir / f"{split}_Y.npy")
        ts = np.load(args.splits_dir / f"{split}_TS.npy")
        if not (x.ndim == 5 and y.ndim == 4 and ts.ndim == 1):
            raise ValueError(
                f"{split} shapes invalid: X{tuple(x.shape)} Y{tuple(y.shape)} TS{tuple(ts.shape)}"
            )
        if not (x.shape[0] == y.shape[0] == ts.shape[0]):
            raise ValueError(
                f"{split} N mismatch: X{tuple(x.shape)} Y{tuple(y.shape)} TS{tuple(ts.shape)}"
            )
        idx_path = args.splits_dir / f"{split}_idx.npy"
        idx = np.load(idx_path) if idx_path.exists() else None

        x_n = _apply_norm(x.astype(np.float32), mean, std)
        np.save(args.out_dir / f"{split}_X.npy", x_n)
        np.save(args.out_dir / f"{split}_Y.npy", y)
        np.save(args.out_dir / f"{split}_TS.npy", ts)
        if idx is not None:
            np.save(args.out_dir / f"{split}_idx.npy", idx)

    # carry meta.json if present (but don't overwrite stats)
    meta_path = args.splits_dir / "meta.json"
    if meta_path.exists():
        (args.out_dir / "meta.json").write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
