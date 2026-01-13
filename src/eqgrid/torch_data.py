from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os


@dataclass(frozen=True)
class SplitArrays:
    X: np.ndarray
    Y: np.ndarray
    TS: np.ndarray


def load_split_arrays(splits_dir, split: Literal["train", "val", "test"]) -> SplitArrays:
    d = Path(splits_dir)
    x_path = d / f"{split}_X.npy"
    y_path = d / f"{split}_Y.npy"
    ts_path = d / f"{split}_TS.npy"
    if not x_path.exists() or not y_path.exists() or not ts_path.exists():
        cwd = os.getcwd()
        if d.exists():
            contents = sorted(p.name for p in d.glob("*"))
            contents_preview = contents[:50]
        else:
            contents_preview = None

        looks_like_samples_dir = (
            d.exists()
            and (d / "XS.npy").exists()
            and (d / "YS.npy").exists()
            and (d / "TS.npy").exists()
        )
        hint = (
            "It looks like you passed a *samples* directory (has XS/YS/TS).\n"
            "Fix:\n"
            "- Run `python -m eqgrid.make_splits --samples-dir <this_dir> --out-dir <splits_dir>`\n"
            "- Then train with `--splits-dir <splits_dir>`.\n"
            if looks_like_samples_dir
            else ""
        )
        raise FileNotFoundError(
            "Split files not found.\n"
            f"- splits_dir: {d}\n"
            f"- cwd: {cwd}\n"
            f"- expected: {x_path.name}, {y_path.name}, {ts_path.name}\n"
            f"- splits_dir exists: {d.exists()}\n"
            f"- splits_dir contents (first 50): {contents_preview}\n"
            f"{hint}"
            "Fix:\n"
            "- Ensure you ran `python -m eqgrid.make_samples ...` then `python -m eqgrid.make_splits ...`\n"
            "- Or pass the correct path (if running from `src/`, use `--splits-dir ../data/processed/splits_L12_H1`).\n"
        )

    X = np.load(x_path)
    Y = np.load(y_path)
    TS = np.load(ts_path)
    return SplitArrays(X=X, Y=Y, TS=TS)


class NumpySeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        if X.ndim != 5:
            raise ValueError("X must be [N,L,C,H,W]")
        if Y.ndim != 4:
            raise ValueError("Y must be [N,1,H,W]")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X/Y must share N")
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


def compute_pos_weight(y: np.ndarray, cap: float | None = None) -> float:
    pos = float(y.sum())
    total = float(y.size)
    neg = total - pos
    if pos <= 0.0:
        return 1.0
    w = neg / pos
    if cap is not None:
        w = min(w, float(cap))
    return float(w)
