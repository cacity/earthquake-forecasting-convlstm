from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


CHANNEL_SETS = {
    "count": {
        "indices": [0],
        "names": ["count_norm"],
    },
    "activity": {
        "indices": [0, 1, 2, 7, 8],
        "names": ["count_norm", "max_mag_norm", "log_energy", "recency_weight", "rate_change"],
    },
}


def _copy_or_slice(src_dir: Path, out_dir: Path, split: str, indices: list[int]) -> None:
    x = np.load(src_dir / f"{split}_X.npy")
    np.save(out_dir / f"{split}_X.npy", x[:, :, indices, :, :].astype(np.float32))
    for suffix in ["Y", "TS", "idx"]:
        src = src_dir / f"{split}_{suffix}.npy"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-splits", type=Path, default=Path("data/processed_china_10ch/splits_L12_H1"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed_china_10ch/ablations"))
    args = parser.parse_args()

    src_dir = args.src_splits.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for name, spec in CHANNEL_SETS.items():
        out_dir = out_root / f"splits_L12_H1_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = list(spec["indices"])
        for split in ["train", "val", "test"]:
            _copy_or_slice(src_dir, out_dir, split, indices)

        src_meta = src_dir / "meta.json"
        meta = json.loads(src_meta.read_text(encoding="utf-8")) if src_meta.exists() else {}
        meta["channel_ablation"] = {
            "name": name,
            "indices": indices,
            "names": list(spec["names"]),
            "source_splits": str(src_dir),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_dir} with channels {indices}: {', '.join(spec['names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
