from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--require-boundaries-in-ts", action="store_true")
    args = p.parse_args(argv)
    try:
        from .split_sanity import print_split_summary
    except ImportError:  # allow running as a plain script
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eqgrid.split_sanity import print_split_summary

    try:
        print_split_summary(
            args.splits_dir, require_boundaries_in_ts=bool(args.require_boundaries_in_ts)
        )
    except ValueError as e:
        raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
