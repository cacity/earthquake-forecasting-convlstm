from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS = [
    {
        "model": "ConvLSTM-count",
        "input_channels": "count_norm only",
        "n_channels": 1,
        "run_dir": Path("outputs/channel_ablation/convlstm_count"),
    },
    {
        "model": "ConvLSTM-activity",
        "input_channels": "count_norm, max_mag_norm, log_energy, recency_weight, rate_change",
        "n_channels": 5,
        "run_dir": Path("outputs/channel_ablation/convlstm_activity"),
    },
    {
        "model": "ConvLSTM-full",
        "input_channels": "all 10 channels",
        "n_channels": 10,
        "run_dir": Path("outputs/convlstm_china_10ch"),
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/channel_ablation"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run in RUNS:
        metrics = _load_json(run["run_dir"] / "metrics.json")
        threshold = _load_json(run["run_dir"] / "threshold.json")
        config = _load_json(run["run_dir"] / "config.json")
        rows.append(
            {
                "model": run["model"],
                "input_channels": run["input_channels"],
                "n_channels": run["n_channels"],
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "brier": float(metrics["brier"]),
                "log_loss": float(metrics["logloss"]),
                "validation_selected_threshold": float(threshold["best_threshold"]),
                "test_f1_at_validation_threshold": float(metrics["f1_thr_best"]),
                "best_epoch": int(metrics["best_epoch"]),
                "in_channels_recorded": int(config["model"]["in_channels"]),
            }
        )

    json_path = args.out_dir / "channel_ablation_summary.json"
    csv_path = args.out_dir / "channel_ablation_summary.csv"
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']}: ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
            f"Brier={row['brier']:.4f}, log loss={row['log_loss']:.4f}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
