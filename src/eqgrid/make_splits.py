from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .splits import SplitConfig, save_splits, time_splits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-end", type=str, default="2018-12-01")
    p.add_argument("--val-end", type=str, default="2022-12-01")
    p.add_argument(
        "--require-non-empty",
        action="store_true",
        help="Fail if any of train/val/test is empty",
    )
    args = p.parse_args(argv)

    XS = np.load(args.samples_dir / "XS.npy")
    YS = np.load(args.samples_dir / "YS.npy")
    TS = np.load(args.samples_dir / "TS.npy")

    cfg = SplitConfig(train_end=args.train_end, val_end=args.val_end)
    splits = time_splits(TS, cfg)
    require = ("train", "val", "test") if args.require_non_empty else ()
    save_splits(args.out_dir, XS, YS, TS, splits, cfg, require_non_empty=require)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
