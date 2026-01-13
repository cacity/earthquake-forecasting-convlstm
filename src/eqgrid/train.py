from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import CNNOnly, ConvLSTMForecaster
from .torch_data import NumpySeqDataset, compute_pos_weight, load_split_arrays


@dataclass(frozen=True)
class TrainConfig:
    model: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    patience: int
    seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).to(dtype=y.dtype)
    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    tn = ((1 - pred) * (1 - y)).sum().item()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    pos_rate = y.mean().item()
    pred_rate = pred.mean().item()
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pos_rate": float(pos_rate),
        "pred_rate": float(pred_rate),
    }


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    pos = int(y_true.sum())
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)

def _counts_at_threshold(probs: torch.Tensor, y: torch.Tensor, thr: float) -> tuple[float, float, float, float]:
    pred = (probs >= thr).to(dtype=y.dtype)
    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    tn = ((1 - pred) * (1 - y)).sum().item()
    return float(tp), float(fp), float(fn), float(tn)


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float(2 * precision * recall / (precision + recall + eps))


def _select_threshold(val_probs: torch.Tensor, val_y: torch.Tensor, thresholds: list[float]) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        tp, fp, fn, _ = _counts_at_threshold(val_probs, val_y, thr)
        f1 = _f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="convlstm", choices=["cnn", "convlstm"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pos-weight-cap", type=float, default=100.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--threshold-scan", type=str, default="0.05:0.95:0.05")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load_split_arrays(args.splits_dir, "train")
    val = load_split_arrays(args.splits_dir, "val")
    test = load_split_arrays(args.splits_dir, "test")

    in_channels = int(train.X.shape[2])
    lookback = int(train.X.shape[1])
    if cfg.model == "cnn":
        model: nn.Module = CNNOnly(in_channels=in_channels * lookback)
    else:
        model = ConvLSTMForecaster(
            input_channels=in_channels,
            hidden_channels=int(args.hidden_channels),
            num_layers=int(args.num_layers),
        )
    model.to(device)

    pos_weight = compute_pos_weight(train.Y, cap=float(args.pos_weight_cap) if args.pos_weight_cap else None)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        NumpySeqDataset(train.X, train.Y),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        NumpySeqDataset(val.X, val.Y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "train": asdict(cfg),
                "model": {
                    "in_channels": in_channels,
                    "lookback": lookback,
                    "hidden_channels": int(args.hidden_channels),
                    "num_layers": int(args.num_layers),
                },
                "pos_weight": pos_weight,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n_train += int(xb.size(0))
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                n_val += int(xb.size(0))
        val_loss /= max(1, n_val)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (args.out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                args.out_dir / "ckpt_best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    ckpt = torch.load(args.out_dir / "ckpt_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Pick a validation threshold once, then reuse it for test metrics.
    def _parse_scan(s: str) -> list[float]:
        a, b, step = (float(x) for x in s.split(":"))
        vals = []
        v = a
        while v <= b + 1e-12:
            vals.append(round(v, 10))
            v += step
        return vals

    val_probs_list: list[torch.Tensor] = []
    val_logits_list: list[torch.Tensor] = []  # 保存 validation logits
    val_y_list: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            val_probs_list.append(torch.sigmoid(logits).cpu())
            val_logits_list.append(logits.cpu())  # 收集 logits
            val_y_list.append(yb.cpu())
    threshold_reason = "val_scan_f1"
    thresholds: list[float] = []
    f1_by_thr: list[float] = []
    if val_probs_list:
        val_probs = torch.cat(val_probs_list, dim=0)
        val_y = torch.cat(val_y_list, dim=0)
        if float(val_y.sum().item()) <= 0.0:
            best_thr = 0.95
            threshold_reason = "val_has_no_positives"
        else:
            thresholds = _parse_scan(args.threshold_scan)
            best_thr = _select_threshold(val_probs, val_y, thresholds)
    else:
        best_thr = 0.5
        threshold_reason = "val_empty"

    # Save scan curve for reproducible plotting (avoid re-running inference just to draw the curve).
    if thresholds and val_probs_list:
        for thr in thresholds:
            tp, fp, fn, _ = _counts_at_threshold(val_probs, val_y, float(thr))
            f1_by_thr.append(_f1_from_counts(tp, fp, fn))
    elif thresholds:
        f1_by_thr = [0.0 for _ in thresholds]
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {"best_threshold": best_thr, "scan": args.threshold_scan, "reason": threshold_reason},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "threshold_scan_f1.json").write_text(
        json.dumps(
            {
                "scan": args.threshold_scan,
                "thresholds": thresholds,
                "f1": f1_by_thr,
                "best_threshold": float(best_thr),
                "reason": threshold_reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 保存 validation logits（评估脚本需要用于校准）
    if val_logits_list:
        val_logits_cat = torch.cat(val_logits_list, dim=0)
        val_logits_arr = val_logits_cat.numpy().astype(np.float32)
        np.save(args.out_dir / "val_logits.npy", val_logits_arr)

    test_loader = DataLoader(
        NumpySeqDataset(test.X, test.Y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds: list[np.ndarray] = []
    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            preds.append(probs)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())

    preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
    np.save(args.out_dir / "preds_test.npy", preds_arr)
    np.save(args.out_dir / "test_preds.npy", preds_arr)  # 额外保存，评估脚本需要
    np.save(args.out_dir / "ts_test.npy", test.TS)

    if all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        # 保存 test logits（评估脚本需要用于校准）
        logits_arr = logits_cat.numpy().astype(np.float32)
        np.save(args.out_dir / "test_logits.npy", logits_arr)
        y_cat = torch.cat(all_y, dim=0)
        metrics = _metrics_from_logits(logits_cat, y_cat)
        probs_cat = torch.sigmoid(logits_cat)
        tp, fp, fn, tn = _counts_at_threshold(probs_cat, y_cat, best_thr)
        metrics.update(
            {
                "thr_best": float(best_thr),
                "f1_thr_best": _f1_from_counts(tp, fp, fn),
                "tp_thr_best": tp,
                "fp_thr_best": fp,
                "fn_thr_best": fn,
                "tn_thr_best": tn,
            }
        )
        tp05, fp05, fn05, tn05 = _counts_at_threshold(probs_cat, y_cat, 0.5)
        metrics.update(
            {
                "thr_0_5": 0.5,
                "tp_thr_0_5": tp05,
                "fp_thr_0_5": fp05,
                "fn_thr_0_5": fn05,
                "tn_thr_0_5": tn05,
            }
        )

        y_np = y_cat.numpy().astype(np.int64).reshape(-1)
        p_np = probs_cat.numpy().astype(np.float32).reshape(-1)
        metrics["pr_auc"] = _average_precision(y_np, p_np)
        roc = _roc_auc(y_np, p_np)
        if roc is not None:
            metrics["roc_auc"] = roc
        # Proper scoring rules (probabilistic quality).
        eps = 1e-12
        y_f = y_np.astype(np.float32)
        p_clip = np.clip(p_np, eps, 1.0 - eps)
        metrics["brier"] = float(np.mean((p_np - y_f) ** 2)) if p_np.size else 0.0
        metrics["logloss"] = float(-np.mean(y_f * np.log(p_clip) + (1.0 - y_f) * np.log(1.0 - p_clip))) if p_np.size else 0.0
    else:
        metrics = {}
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "test_size": int(test.X.shape[0]),
        }
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
