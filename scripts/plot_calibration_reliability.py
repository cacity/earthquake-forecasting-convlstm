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


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float32).reshape(-1)
    p = np.clip(p.astype(np.float32).reshape(-1), 0.0, 1.0)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = y.astype(np.float32).reshape(-1)
    p = np.clip(p.astype(np.float32).reshape(-1), eps, 1.0 - eps)
    if p.size == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[float, list[dict[str, Any]]]:
    y = y.astype(np.float32).reshape(-1)
    p = p.astype(np.float32).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    n = float(p.size)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "p_mean": None, "y_mean": None})
            continue
        p_mean = float(p[m].mean())
        y_mean = float(y[m].mean())
        w = float(m.sum()) / max(n, 1.0)
        ece += w * abs(y_mean - p_mean)
        rows.append(
            {"bin_lo": lo, "bin_hi": hi, "count": int(m.sum()), "p_mean": p_mean, "y_mean": y_mean}
        )
    return float(ece), rows


def _load_run_cfg(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("run_dir/config.json must be a JSON object")
    return payload


def _build_model(run_dir: Path):
    import torch

    from eqgrid.models import CNNOnly, ConvLSTMForecaster

    cfg = _load_run_cfg(run_dir)
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    if not isinstance(train_cfg, dict) or not isinstance(model_cfg, dict):
        raise TypeError("run_dir/config.json must contain 'train' and 'model' objects")

    model_name = str(train_cfg.get("model", "convlstm"))
    in_channels = int(model_cfg["in_channels"])
    lookback = int(model_cfg["lookback"])
    if model_name == "cnn":
        model = CNNOnly(in_channels=in_channels * lookback)
    else:
        model = ConvLSTMForecaster(
            input_channels=in_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            num_layers=int(model_cfg.get("num_layers", 1)),
        )
    return model, cfg


def _predict_split_probs(run_dir: Path, splits_dir: Path, split: str, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    from eqgrid.torch_data import NumpySeqDataset, load_split_arrays

    model, cfg = _build_model(run_dir)
    train_cfg = cfg.get("train") or {}
    batch_size = int(train_cfg.get("batch_size", 16)) if isinstance(train_cfg, dict) else 16

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()

    arr = load_split_arrays(splits_dir, split)
    loader = DataLoader(
        NumpySeqDataset(arr.X, arr.Y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )

    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(dev)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probs_list.append(probs)
    probs_all = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0,), dtype=np.float32)
    return probs_all, arr.TS.astype("datetime64[ns]")


def _platt_fit_from_probs(p: np.ndarray, y: np.ndarray, device: str) -> dict[str, Any]:
    import torch

    y = y.astype(np.float32).reshape(-1)
    p = np.clip(p.astype(np.float32).reshape(-1), 1e-6, 1.0 - 1e-6)
    x = np.log(p / (1.0 - p)).astype(np.float32)  # logit(p)

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y).to(dev)

    a = torch.zeros((), device=dev, requires_grad=True)
    b = torch.zeros((), device=dev, requires_grad=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = a * xt + b
        loss = loss_fn(logits, yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        final = float(loss_fn(a * xt + b, yt).item())
    return {"a": float(a.detach().cpu().item()), "b": float(b.detach().cpu().item()), "val_logloss": final}


def _platt_apply(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)
    x = np.log(p / (1.0 - p))
    logits = a * x + b
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _xy_from_bins(bins_tbl: list[dict[str, Any]]) -> tuple[list[float], list[float], list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    ns: list[int] = []
    for r in bins_tbl:
        if r["count"] and r["p_mean"] is not None and r["y_mean"] is not None:
            xs.append(float(r["p_mean"]))
            ys.append(float(r["y_mean"]))
            ns.append(int(r["count"]))
    return xs, ys, ns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("."), help="Resolve default paths under this dir")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("figures/calibration"))
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--fit-platt", action="store_true", help="Fit Platt scaling on val and apply to test")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fig-out", type=Path, default=None, help="Optional explicit output path for the figure PNG")
    args = p.parse_args()

    base_dir = args.base_dir.resolve()
    run_dir = (base_dir / "outputs/convlstm_L12") if args.run_dir is None else args.run_dir
    splits_dir = (base_dir / "data/processed/splits_L12_H1") if args.splits_dir is None else args.splits_dir
    run_dir = run_dir.resolve()
    splits_dir = splits_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds = np.load(run_dir / "preds_test.npy").astype(np.float32)
    y = np.load(splits_dir / "test_Y.npy").astype(np.float32)
    if preds.shape != y.shape:
        raise ValueError(f"preds/y shape mismatch: {tuple(preds.shape)} vs {tuple(y.shape)}")

    pflat = np.clip(preds.reshape(-1), 0.0, 1.0).astype(np.float32)
    yflat = y.reshape(-1).astype(np.float32)

    raw_ece, raw_bins = _ece(yflat, pflat, n_bins=int(args.bins))
    raw = {
        "brier": _brier(yflat, pflat),
        "logloss": _logloss(yflat, pflat),
        "ece": float(raw_ece),
        "bins": raw_bins,
        "n_points": int(pflat.size),
        "pos_rate": float(yflat.mean()) if yflat.size else 0.0,
    }

    # Reference: per-cell historical rate from train split.
    train_y = np.load(splits_dir / "train_Y.npy").astype(np.float32)  # [N,1,H,W]
    hist_rate = train_y.mean(axis=0, keepdims=True)  # [1,1,H,W]
    hist_pred = np.repeat(hist_rate, repeats=y.shape[0], axis=0)  # [N,1,H,W]
    hist_flat = np.clip(hist_pred.reshape(-1), 0.0, 1.0).astype(np.float32)
    ref_ece, _ref_bins = _ece(yflat, hist_flat, n_bins=int(args.bins))
    ref = {
        "brier": _brier(yflat, hist_flat),
        "logloss": _logloss(yflat, hist_flat),
        "ece": float(ref_ece),
    }

    out: dict[str, Any] = {
        "n_bins": int(args.bins),
        "raw": raw,
        "reference_hist_rate": ref,
        "brier_skill_vs_hist_rate_raw": (1.0 - raw["brier"] / ref["brier"]) if ref["brier"] > 0 else None,
        "logloss_skill_vs_hist_rate_raw": (1.0 - raw["logloss"] / ref["logloss"]) if ref["logloss"] > 0 else None,
    }

    platt: dict[str, Any] | None = None
    if args.fit_platt:
        platt_path = run_dir / "platt_calibration.json"
        val_probs_path = run_dir / "preds_val.npy"
        if platt_path.exists():
            platt_params = json.loads(platt_path.read_text(encoding="utf-8"))
        else:
            if val_probs_path.exists():
                val_probs = np.load(val_probs_path).astype(np.float32)
            else:
                val_probs, _ = _predict_split_probs(run_dir, splits_dir, "val", device=str(args.device))
                np.save(val_probs_path, val_probs.astype(np.float32))
            val_y = np.load(splits_dir / "val_Y.npy").astype(np.float32)
            platt_params = _platt_fit_from_probs(val_probs.reshape(-1), val_y.reshape(-1), device=str(args.device))
            platt_path.write_text(json.dumps(platt_params, ensure_ascii=False, indent=2), encoding="utf-8")

        a = float(platt_params["a"])
        b = float(platt_params["b"])
        p_cal = _platt_apply(pflat, a=a, b=b)
        p_cal_4d = p_cal.reshape(preds.shape).astype(np.float32)
        np.save(run_dir / "preds_test_platt.npy", p_cal_4d)

        platt_ece, platt_bins = _ece(yflat, p_cal, n_bins=int(args.bins))
        platt = {
            "a": a,
            "b": b,
            "brier": _brier(yflat, p_cal),
            "logloss": _logloss(yflat, p_cal),
            "ece": float(platt_ece),
            "bins": platt_bins,
        }
        out["platt"] = platt
        out["brier_skill_vs_hist_rate_platt"] = (1.0 - platt["brier"] / ref["brier"]) if ref["brier"] > 0 else None
        out["logloss_skill_vs_hist_rate_platt"] = (1.0 - platt["logloss"] / ref["logloss"]) if ref["logloss"] > 0 else None
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="perfect")

    xs, ys, ns = _xy_from_bins(raw["bins"])
    if xs:
        size = np.clip(np.array(ns, dtype=float) / max(np.max(ns), 1.0), 0.05, 1.0) * 180
        ax.scatter(xs, ys, s=size, alpha=0.85, label="raw")
        ax.plot(xs, ys, lw=1.2)

    if platt is not None:
        xs2, ys2, _ns2 = _xy_from_bins(platt["bins"])
        if xs2:
            ax.plot(xs2, ys2, lw=1.6, label="platt", color="#C44E52")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    title = f"Reliability (test)\nraw ECE={raw['ece']:.3f}, Brier={raw['brier']:.4f}"
    if platt is not None:
        title += f" | platt ECE={platt['ece']:.3f}, Brier={platt['brier']:.4f}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_name = "reliability_before_after.png" if platt is not None else "reliability_diagram.png"
    fig_path = (args.out_dir / out_name) if args.fig_out is None else args.fig_out
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
