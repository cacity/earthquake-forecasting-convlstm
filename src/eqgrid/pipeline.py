from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STEP_ORDER: list[str] = [
    "download",
    "build_tensors",
    "make_samples",
    "make_splits",
    "normalize_splits",
    "train",
    "report_test",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config must be a JSON object at top-level")
    return payload


def _resolve_path(base_dir: Path, p: str | Path) -> Path:
    pp = p if isinstance(p, Path) else Path(p)
    if pp.is_absolute():
        return pp
    return base_dir / pp


def _is_enabled(cfg: dict[str, Any], key: str) -> bool:
    sec = cfg.get(key)
    if sec is None:
        return False
    if not isinstance(sec, dict):
        raise TypeError(f'"{key}" must be an object')
    return bool(sec.get("enabled", True))


def _bbox(cfg: dict[str, Any], sec: dict[str, Any], step: str) -> list[float]:
    bbox = sec.get("bbox", cfg.get("bbox"))
    if bbox is None:
        raise ValueError(f'"bbox" is required for step "{step}" (or set top-level "bbox")')
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise TypeError(f'"bbox" must be a list of 4 numbers for step "{step}"')
    return [float(x) for x in bbox]


def _maybe_add(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, list):
        argv.append(flag)
        argv.extend([str(x) for x in value])
        return
    argv.extend([flag, str(value)])


def _require_str(sec: dict[str, Any], key: str, step: str) -> str:
    v = sec.get(key)
    if v is None:
        raise ValueError(f'"{key}" is required for step "{step}"')
    if not isinstance(v, str):
        raise TypeError(f'"{key}" must be a string for step "{step}"')
    return v


def _require_path(base_dir: Path, sec: dict[str, Any], key: str, step: str) -> Path:
    v = sec.get(key)
    if v is None:
        raise ValueError(f'"{key}" is required for step "{step}"')
    if not isinstance(v, str):
        raise TypeError(f'"{key}" must be a string path for step "{step}"')
    return _resolve_path(base_dir, v).resolve()


def _opt_path(base_dir: Path, sec: dict[str, Any], key: str) -> Path | None:
    v = sec.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(f'"{key}" must be a string path')
    return _resolve_path(base_dir, v).resolve()


def _step_download(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import download

    step = "download"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []
    out_dir = _require_path(base_dir, sec, "out_dir", step)
    _maybe_add(argv, "--out-dir", out_dir)
    _maybe_add(argv, "--start", sec.get("start", "2000-01-01"))
    _maybe_add(argv, "--end", sec.get("end", "today"))
    _maybe_add(argv, "--min-mag", sec.get("min_mag", 3.5))
    _maybe_add(argv, "--bbox", _bbox(cfg, sec, step))
    _maybe_add(argv, "--segment", sec.get("segment", "yearly"))

    events_out = sec.get("events_out", "data/interim/events.parquet")
    if not isinstance(events_out, str):
        raise TypeError('"download.events_out" must be a string path')
    _maybe_add(argv, "--events-out", _resolve_path(base_dir, events_out).resolve())

    if dry_run:
        print(f"[dry-run] python -m eqgrid.download {' '.join(map(str, argv))}")
        return 0
    return int(download.main(argv))


def _step_build_tensors(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import build_tensors

    step = "build_tensors"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    events = sec.get("events")
    if events is None:
        d = cfg.get("download") or {}
        if isinstance(d, dict):
            events = d.get("events_out", "data/interim/events.parquet")
    if not isinstance(events, str):
        raise ValueError('"build_tensors.events" is required (or set "download.events_out")')
    _maybe_add(argv, "--events", _resolve_path(base_dir, events).resolve())

    out_dir = sec.get("out_dir", "data/processed")
    if not isinstance(out_dir, str):
        raise TypeError('"build_tensors.out_dir" must be a string path')
    _maybe_add(argv, "--out-dir", _resolve_path(base_dir, out_dir).resolve())

    _maybe_add(argv, "--bbox", _bbox(cfg, sec, step))
    _maybe_add(argv, "--freq", sec.get("freq", cfg.get("freq", "monthly")))
    _maybe_add(argv, "--bins-start", sec.get("bins_start", "2000-01-01"))
    _maybe_add(argv, "--bins-end", sec.get("bins_end", "today"))
    _maybe_add(argv, "--count-clip", sec.get("count_clip", 10.0))
    _maybe_add(argv, "--mag-scale", sec.get("mag_scale", 10.0))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.build_tensors {' '.join(map(str, argv))}")
        return 0
    return int(build_tensors.main(argv))


def _step_make_samples(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import make_samples

    step = "make_samples"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    bt = cfg.get("build_tensors") or {}
    bt_out = bt.get("out_dir", "data/processed") if isinstance(bt, dict) else "data/processed"
    if not isinstance(bt_out, str):
        bt_out = "data/processed"
    bt_out_p = _resolve_path(base_dir, bt_out).resolve()

    x = sec.get("x", str(bt_out_p / "X.npy"))
    y = sec.get("y", str(bt_out_p / "Y.npy"))
    bins = sec.get("bins", str(bt_out_p / "bins.npy"))
    if not (isinstance(x, str) and isinstance(y, str) and isinstance(bins, str)):
        raise TypeError('"make_samples.x/y/bins" must be string paths')

    _maybe_add(argv, "--x", _resolve_path(base_dir, x).resolve())
    _maybe_add(argv, "--y", _resolve_path(base_dir, y).resolve())
    _maybe_add(argv, "--bins", _resolve_path(base_dir, bins).resolve())

    out_dir = _require_str(sec, "out_dir", step)
    _maybe_add(argv, "--out-dir", _resolve_path(base_dir, out_dir).resolve())
    _maybe_add(argv, "--lookback", sec.get("lookback", 12))
    _maybe_add(argv, "--horizon", sec.get("horizon", 1))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.make_samples {' '.join(map(str, argv))}")
        return 0
    return int(make_samples.main(argv))


def _step_make_splits(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import make_splits

    step = "make_splits"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    ms = cfg.get("make_samples") or {}
    ms_out = ms.get("out_dir") if isinstance(ms, dict) else None
    samples_dir = sec.get("samples_dir", ms_out)
    if not isinstance(samples_dir, str):
        raise ValueError('"make_splits.samples_dir" is required (or set "make_samples.out_dir")')
    _maybe_add(argv, "--samples-dir", _resolve_path(base_dir, samples_dir).resolve())

    out_dir = _require_str(sec, "out_dir", step)
    _maybe_add(argv, "--out-dir", _resolve_path(base_dir, out_dir).resolve())
    _maybe_add(argv, "--train-end", sec.get("train_end", "2018-12-01"))
    _maybe_add(argv, "--val-end", sec.get("val_end", "2022-12-01"))
    _maybe_add(argv, "--require-non-empty", bool(sec.get("require_non_empty", False)))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.make_splits {' '.join(map(str, argv))}")
        return 0
    return int(make_splits.main(argv))


def _step_normalize_splits(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import normalize_splits

    step = "normalize_splits"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    mk = cfg.get("make_splits") or {}
    mk_out = mk.get("out_dir") if isinstance(mk, dict) else None
    splits_dir = sec.get("splits_dir", mk_out)
    if not isinstance(splits_dir, str):
        raise ValueError('"normalize_splits.splits_dir" is required (or set "make_splits.out_dir")')
    _maybe_add(argv, "--splits-dir", _resolve_path(base_dir, splits_dir).resolve())

    out_dir = _require_str(sec, "out_dir", step)
    _maybe_add(argv, "--out-dir", _resolve_path(base_dir, out_dir).resolve())
    _maybe_add(argv, "--eps", sec.get("eps", 1e-6))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.normalize_splits {' '.join(map(str, argv))}")
        return 0
    return int(normalize_splits.main(argv))


def _step_train(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import train

    step = "train"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    splits_dir = sec.get("splits_dir")
    if splits_dir is None:
        ns = cfg.get("normalize_splits") or {}
        mk = cfg.get("make_splits") or {}
        if isinstance(ns, dict) and bool(ns.get("enabled", False)):
            splits_dir = ns.get("out_dir")
        if splits_dir is None and isinstance(mk, dict):
            splits_dir = mk.get("out_dir")
    if not isinstance(splits_dir, str):
        raise ValueError(
            '"train.splits_dir" is required (or set "make_splits.out_dir", or enable "normalize_splits" with "out_dir")'
        )
    _maybe_add(argv, "--splits-dir", _resolve_path(base_dir, splits_dir).resolve())

    out_dir = _require_str(sec, "out_dir", step)
    _maybe_add(argv, "--out-dir", _resolve_path(base_dir, out_dir).resolve())

    _maybe_add(argv, "--model", sec.get("model", "convlstm"))
    _maybe_add(argv, "--epochs", sec.get("epochs", 50))
    _maybe_add(argv, "--batch-size", sec.get("batch_size", 16))
    _maybe_add(argv, "--lr", sec.get("lr", 1e-3))
    _maybe_add(argv, "--weight-decay", sec.get("weight_decay", 1e-2))
    _maybe_add(argv, "--grad-clip", sec.get("grad_clip", 1.0))
    _maybe_add(argv, "--patience", sec.get("patience", 5))
    _maybe_add(argv, "--seed", sec.get("seed", 42))
    _maybe_add(argv, "--hidden-channels", sec.get("hidden_channels", 64))
    _maybe_add(argv, "--num-layers", sec.get("num_layers", 1))
    _maybe_add(argv, "--num-workers", sec.get("num_workers", 0))
    _maybe_add(argv, "--pos-weight-cap", sec.get("pos_weight_cap", 100.0))
    _maybe_add(argv, "--deterministic", bool(sec.get("deterministic", False)))
    _maybe_add(argv, "--threshold-scan", sec.get("threshold_scan", "0.05:0.95:0.05"))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.train {' '.join(map(str, argv))}")
        return 0
    return int(train.main(argv))


def _step_report_test(cfg: dict[str, Any], base_dir: Path, dry_run: bool) -> int:
    from . import report_test

    step = "report_test"
    sec = cfg.get(step) or {}
    if not isinstance(sec, dict):
        raise TypeError(f'"{step}" must be an object')
    argv: list[str] = []

    tr = cfg.get("train") or {}
    tr_out = tr.get("out_dir") if isinstance(tr, dict) else None

    preds = sec.get("preds", (str(Path(tr_out) / "preds_test.npy") if isinstance(tr_out, str) else None))
    ts = sec.get("ts", (str(Path(tr_out) / "ts_test.npy") if isinstance(tr_out, str) else None))
    if not (isinstance(preds, str) and isinstance(ts, str)):
        raise ValueError('"report_test.preds" and "report_test.ts" are required (or set "train.out_dir")')
    _maybe_add(argv, "--preds", _resolve_path(base_dir, preds).resolve())
    _maybe_add(argv, "--ts", _resolve_path(base_dir, ts).resolve())

    out = sec.get("out", (str(Path(tr_out) / "report_test.csv") if isinstance(tr_out, str) else None))
    if not isinstance(out, str):
        raise ValueError('"report_test.out" is required (or set "train.out_dir")')
    _maybe_add(argv, "--out", _resolve_path(base_dir, out).resolve())

    bt = cfg.get("build_tensors") or {}
    bt_out = bt.get("out_dir", "data/processed") if isinstance(bt, dict) else "data/processed"
    if not isinstance(bt_out, str):
        bt_out = "data/processed"
    grid_meta = sec.get("grid_meta", str(_resolve_path(base_dir, bt_out).resolve() / "grid_meta.json"))
    if grid_meta is not None:
        if not isinstance(grid_meta, str):
            raise TypeError('"report_test.grid_meta" must be a string path')
        _maybe_add(argv, "--grid-meta", _resolve_path(base_dir, grid_meta).resolve())

    threshold_json = sec.get(
        "threshold_json",
        (str(Path(tr_out) / "threshold.json") if isinstance(tr_out, str) else None),
    )
    if threshold_json is not None:
        if not isinstance(threshold_json, str):
            raise TypeError('"report_test.threshold_json" must be a string path')
        _maybe_add(argv, "--threshold-json", _resolve_path(base_dir, threshold_json).resolve())

    _maybe_add(argv, "--thr", sec.get("thr"))
    _maybe_add(argv, "--topk", sec.get("topk", 20))

    if dry_run:
        print(f"[dry-run] python -m eqgrid.report_test {' '.join(map(str, argv))}")
        return 0
    return int(report_test.main(argv))


STEP_FUNCS = {
    "download": _step_download,
    "build_tensors": _step_build_tensors,
    "make_samples": _step_make_samples,
    "make_splits": _step_make_splits,
    "normalize_splits": _step_normalize_splits,
    "train": _step_train,
    "report_test": _step_report_test,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the full eqgrid pipeline from a JSON config.")
    p.add_argument("--config", type=Path, required=True, help="Path to pipeline JSON config file")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for resolving relative paths (overrides config base_dir)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print resolved commands and exit")
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help='Comma-separated steps to run (default: enabled steps in built-in order: "download,build_tensors,...")',
    )
    args = p.parse_args(argv)

    cfg = _load_json(args.config)
    cfg_dir = args.config.parent.resolve()
    if args.base_dir is not None:
        base_dir = _resolve_path(Path.cwd(), args.base_dir).resolve()
    else:
        base_dir_raw = cfg.get("base_dir")
        if base_dir_raw is None:
            base_dir = Path.cwd().resolve()
        else:
            if not isinstance(base_dir_raw, str):
                raise TypeError('"base_dir" must be a string path')
            base_dir = _resolve_path(cfg_dir, base_dir_raw).resolve()

    if args.steps is not None:
        requested = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = [s for s in requested if s not in STEP_FUNCS]
        if unknown:
            raise ValueError(f"unknown step(s): {', '.join(unknown)}")
        steps = requested
    else:
        steps = [s for s in STEP_ORDER if _is_enabled(cfg, s)]

    if not steps:
        raise ValueError("no steps selected (either add step configs, or pass --steps)")

    for step in steps:
        fn = STEP_FUNCS[step]
        try:
            rc = int(fn(cfg, base_dir, bool(args.dry_run)))
        except Exception as e:
            raise RuntimeError(f"step failed: {step}") from e
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

