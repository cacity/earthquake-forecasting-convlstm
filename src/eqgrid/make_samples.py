from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .samples import SampleConfig, make_samples, save_samples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=Path, required=True)
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args(argv)

    X = np.load(args.x)
    Y = np.load(args.y)
    bins = np.load(args.bins)

    cfg = SampleConfig(lookback=args.lookback, horizon=args.horizon)
    xs, ys, ts = make_samples(X=X, Y=Y, bins=bins, config=cfg)
    save_samples(args.out_dir, xs, ys, ts, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

