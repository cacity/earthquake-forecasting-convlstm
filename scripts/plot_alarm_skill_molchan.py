from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must be a JSON object")
    return payload


def _topk_mask(p: np.ndarray, k: int) -> np.ndarray:
    flat = p.reshape(-1)
    k = int(max(0, min(k, flat.size)))
    if k <= 0:
        return np.zeros_like(flat, dtype=bool).reshape(p.shape)
    # Top-k by value (stable), then mark those indices.
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    return mask.reshape(p.shape)


def _counts_mask(mask: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    # mask: True means "alarm"/positive prediction
    pred = mask.astype(np.float32)
    yy = y.astype(np.float32)
    tp = float((pred * yy).sum())
    fp = float((pred * (1 - yy)).sum())
    fn = float(((1 - pred) * yy).sum())
    tn = float(((1 - pred) * (1 - yy)).sum())
    return tp, fp, fn, tn


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = (2.0 * tp + fp + fn)
    return float((2.0 * tp) / denom) if denom > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("."), help="Resolve default paths under this dir")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("figures/alarm_skill"))
    p.add_argument("--areas", type=str, default="0.01:0.50:0.01", help="alarm area fractions a:b:step")
    p.add_argument("--no-fig", action="store_true", help="Only write JSON, do not write PNG")
    args = p.parse_args()

    base_dir = args.base_dir.resolve()
    run_dir = (base_dir / "outputs/convlstm_L12") if args.run_dir is None else args.run_dir
    splits_dir = (base_dir / "data/processed/splits_L12_H1") if args.splits_dir is None else args.splits_dir
    run_dir = run_dir.resolve()
    splits_dir = splits_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds = np.load(run_dir / "preds_test.npy").astype(np.float32)  # [N,1,H,W]
    y = np.load(splits_dir / "test_Y.npy").astype(np.float32)  # [N,1,H,W]
    if preds.shape != y.shape:
        raise ValueError(f"preds/y shape mismatch: {tuple(preds.shape)} vs {tuple(y.shape)}")
    pmap = preds[:, 0]
    ymap = y[:, 0]
    n, h, w = pmap.shape
    m = int(h * w)

    # Parse areas a:b:step
    a0, a1, astep = (float(x) for x in args.areas.split(":"))
    areas = []
    a = a0
    while a <= a1 + 1e-12:
        areas.append(round(a, 10))
        a += astep
    if not areas:
        raise ValueError("no alarm areas")

    rows: list[dict[str, Any]] = []
    for frac in areas:
        k = int(np.ceil(frac * m))
        tp = fp = fn = tn = 0.0
        for i in range(n):
            mask = _topk_mask(pmap[i], k)
            tpi, fpi, fni, tni = _counts_mask(mask, ymap[i])
            tp += tpi
            fp += fpi
            fn += fni
            tn += tni
        miss_rate = fn / max(tp + fn, 1e-12)
        hit_rate = tp / max(tp + fn, 1e-12)
        precision = tp / max(tp + fp, 1e-12)
        f1 = _f1_from_counts(tp, fp, fn)
        rows.append(
            {
                "alarm_area": float(k / m) if m else 0.0,
                "k_cells": int(k),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "miss_rate": float(miss_rate),
                "hit_rate": float(hit_rate),
                "precision": float(precision),
                "f1": float(f1),
            }
        )

    (args.out_dir / "molchan_alarm_skill.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    x = np.array([r["alarm_area"] for r in rows], dtype=float)
    miss = np.array([r["miss_rate"] for r in rows], dtype=float)
    hit = np.array([r["hit_rate"] for r in rows], dtype=float)
    prec = np.array([r["precision"] for r in rows], dtype=float)
    f1 = np.array([r["f1"] for r in rows], dtype=float)

    best_i = int(np.nanargmax(f1)) if f1.size else 0
    (args.out_dir / "molchan_alarm_skill_summary.json").write_text(
        json.dumps(
            {
                "best_by_f1": rows[best_i] if rows else None,
                "n_test_months": int(n),
                "grid_cells": int(m),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.no_fig:
        return 0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(x, miss, lw=1.8, label="Molchan: miss rate")
    ax.plot([0, 1], [1, 0], "k--", lw=1.0, alpha=0.6, label="random ref")
    ax.set_xlabel("Alarm area fraction (top-k grid cells)")
    ax.set_ylabel("Miss rate (FN / (TP+FN))")
    ax.set_title("Molchan-style error diagram (test)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "molchan_error.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(x, hit, lw=1.8, label="Hit rate")
    ax.plot(x, prec, lw=1.8, label="Precision@TopArea")
    ax.set_xlabel("Alarm area fraction (top-k grid cells)")
    ax.set_ylabel("Score")
    ax.set_title("Alarm skill vs alarm area (test)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "alarm_skill_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Combined figure for main text (Molchan + TopArea skill)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    ax0, ax1 = axes
    ax0.plot(x, miss, lw=1.8, label="miss rate")
    ax0.plot([0, 1], [1, 0], "k--", lw=1.0, alpha=0.6, label="random ref")
    ax0.set_xlabel("Alarm area fraction (top-k cells)")
    ax0.set_ylabel("Miss rate")
    ax0.set_title("Molchan-style diagram (test)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False)

    ax1.plot(x, hit, lw=1.8, label="hit rate")
    ax1.plot(x, prec, lw=1.8, label="precision@TopArea")
    ax1.plot(x, f1, lw=1.8, label="F1@TopArea")
    ax1.axvline(x[best_i], color="k", lw=1.0, alpha=0.6)
    ax1.set_xlabel("Alarm area fraction (top-k cells)")
    ax1.set_ylabel("Score")
    ax1.set_title("Alarm skill vs alarm area (test)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out_dir / "molchan_toparea_combined.png", bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
