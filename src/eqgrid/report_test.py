from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--ts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid-meta", type=Path, help="Optional grid_meta.json from build_tensors")
    p.add_argument("--threshold-json", type=Path)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args(argv)

    preds = np.load(args.preds)  # [N,1,H,W]
    ts = np.load(args.ts)
    ts_pd = pd.to_datetime(ts, utc=True)
    if preds.ndim != 4 or preds.shape[1] != 1:
        raise ValueError(f"preds must be [N,1,H,W], got {tuple(preds.shape)}")
    n, _, h, w_ = preds.shape

    grid_info: dict[str, object] = {}
    if args.grid_meta is not None:
        payload = json.loads(args.grid_meta.read_text(encoding="utf-8"))
        grid = payload.get("grid") or {}
        tcfg = payload.get("tensor_config") or {}
        grid_info = {
            "min_lon": grid.get("min_lon"),
            "max_lon": grid.get("max_lon"),
            "min_lat": grid.get("min_lat"),
            "max_lat": grid.get("max_lat"),
            "resolution_deg": grid.get("resolution_deg"),
            "n_lon": grid.get("n_lon"),
            "n_lat": grid.get("n_lat"),
            "freq": tcfg.get("freq"),
        }

    thr = None
    if args.thr is not None:
        thr = float(args.thr)
    elif args.threshold_json and args.threshold_json.exists():
        payload = json.loads(args.threshold_json.read_text(encoding="utf-8"))
        thr = float(payload.get("best_threshold", 0.5))
    else:
        thr = 0.5

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_out = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_out.write_text(
        json.dumps(
            {
                "preds_shape": [int(x) for x in preds.shape],
                "n_samples": int(n),
                "grid_info": grid_info,
                "threshold": float(thr),
                "topk": int(args.topk),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "mean_prob",
                "alarm_rate",
                "max_prob",
                "topk_mean_prob",
                "thr",
                "H",
                "W",
                "freq",
                "min_lon",
                "max_lon",
                "min_lat",
                "max_lat",
                "n_lat",
                "n_lon",
            ],
        )
        writer.writeheader()
        for i in range(preds.shape[0]):
            pmap = preds[i, 0]
            flat = pmap.reshape(-1)
            topk = min(int(args.topk), flat.size)
            topk_mean = float(np.sort(flat)[-topk:].mean()) if topk > 0 else 0.0
            writer.writerow(
                {
                    "ts": ts_pd[i].isoformat(),
                    "mean_prob": float(flat.mean()),
                    "alarm_rate": float((flat >= thr).mean()),
                    "max_prob": float(flat.max(initial=0.0)),
                    "topk_mean_prob": topk_mean,
                    "thr": float(thr),
                    "H": int(h),
                    "W": int(w_),
                    "freq": grid_info.get("freq"),
                    "min_lon": grid_info.get("min_lon"),
                    "max_lon": grid_info.get("max_lon"),
                    "min_lat": grid_info.get("min_lat"),
                    "max_lat": grid_info.get("max_lat"),
                    "n_lat": grid_info.get("n_lat"),
                    "n_lon": grid_info.get("n_lon"),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
