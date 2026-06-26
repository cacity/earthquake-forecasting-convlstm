# Reproducibility Notes

This document describes the public code path for the manuscript:

**Benchmarking ConvLSTM Against Spatial-Rate and Aftershock-Decay Baselines for Monthly Gridded Earthquake Forecasting in Southwestern China**

The repository does not include raw CENC catalog files, processed tensors, trained model checkpoints, or prediction arrays. Place catalog files in an ignored local directory such as `data/private/`.

## Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Use Python 3.10 or later. CUDA is optional; CPU training is possible but slower.

## Catalog Preparation

The manuscript workflow used CENC catalog files with two source periods:

```text
Before 2009: fixed-width EQT catalog
2009-2025: table/parquet catalog
```

The public script `scripts/build_china_depth_filtered_tensors.py` implements the catalog handling used in the manuscript:

```text
1. Convert pre-2009 EQT depth codes from 0.1 km units to kilometers.
2. Use table/parquet records from 2009 onward.
3. Deduplicate events with time, epicentral, and magnitude tolerances.
4. Keep depth=0 km records because they indicate unspecified depth in this catalog.
5. Exclude events with known focal depth greater than 70 km.
6. Select M>=4.0 events in 95-110E and 22-35N from 2000 through 2025.
```

## Tensor Construction

```bash
python scripts/build_china_depth_filtered_tensors.py ^
  --data-dir data/private ^
  --out-dir data/processed_china_10ch_d0_70_corrected ^
  --bbox 95 110 22 35 ^
  --start 2000-01-01 ^
  --end-exclusive 2026-01-01 ^
  --bins-end 2025-12-01 ^
  --min-mag 4.0 ^
  --depth-max 70
```

Expected tensor shape for the manuscript setting:

```text
X: [312, 10, 13, 15]
Y: [312, 1, 13, 15]
```

The grid contains 195 cells. The ten feature channels are documented in `README.md` and implemented in `src/eqgrid/tensors_enhanced.py`.

## Supervised Samples And Splits

```bash
python -m eqgrid.make_samples ^
  --x data/processed_china_10ch_d0_70_corrected/X.npy ^
  --y data/processed_china_10ch_d0_70_corrected/Y.npy ^
  --bins data/processed_china_10ch_d0_70_corrected/bins.npy ^
  --out-dir data/processed_china_10ch_d0_70_corrected/samples_L12_H1 ^
  --lookback 12 ^
  --horizon 1

python -m eqgrid.make_splits ^
  --samples-dir data/processed_china_10ch_d0_70_corrected/samples_L12_H1 ^
  --out-dir data/processed_china_10ch_d0_70_corrected/splits_L12_H1 ^
  --train-end 2018-12-01 ^
  --val-end 2022-12-01 ^
  --require-non-empty
```

Manuscript split sizes:

| Period | Target months | Grid-months | Positive cell-months | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| Train, 2001-2018 | 216 | 42,120 | 752 | 1.79% |
| Validation, 2019-2022 | 48 | 9,360 | 230 | 2.46% |
| Test, 2023-2025 | 36 | 7,020 | 109 | 1.55% |

## ConvLSTM Training

```bash
python -m eqgrid.train ^
  --splits-dir data/processed_china_10ch_d0_70_corrected/splits_L12_H1 ^
  --out-dir outputs/convlstm_china_10ch_d0_70_corrected ^
  --model convlstm ^
  --epochs 50 ^
  --batch-size 16 ^
  --lr 0.001 ^
  --weight-decay 0.01 ^
  --grad-clip 1.0 ^
  --patience 5 ^
  --hidden-channels 64 ^
  --num-layers 1 ^
  --pos-weight-cap 100 ^
  --threshold-scan 0.01:0.95:0.01 ^
  --seed 42
```

The model uses one ConvLSTM layer with 64 hidden channels and a 3 by 3 kernel, followed by a 1 by 1 convolution. The input channel count is inferred from the split tensors.

## Channel Ablation

```bash
python scripts/make_channel_ablation_splits.py ^
  --src-splits data/processed_china_10ch_d0_70_corrected/splits_L12_H1 ^
  --out-root data/processed_china_10ch_d0_70_corrected/ablations
```

Then train the `count` and `activity` split directories using the same `eqgrid.train` command with different output directories. Summaries can be created with:

```bash
python scripts/summarize_channel_ablation.py ^
  --out-dir outputs/channel_ablation
```

## Baseline Evaluation

Baseline implementations are in `src/eqgrid/baselines.py`:

```text
SPR          training-period stationary spatial rate
Persistence last-month normalized count
KDE          Gaussian-smoothed spatial rate
DAR          Omori-decay aftershock-rate baseline
```

DAR is designed to capture the Omori-Utsu decay component in the same gridded monthly resolution. It is not a full ETAS maximum-likelihood model.

## Leakage Controls

The key leakage controls are:

```text
1. Chronological train/validation/test splits.
2. Positive-class weighting computed from the training split only.
3. Threshold selection performed on validation data only.
4. Calibration fitted on validation predictions only.
5. No full-catalog z-score normalization.
6. rate_change and spatial_density scaled with fixed analytic bounds.
```

## Notes On Exact Reproduction

Exact numerical reproduction requires the same CENC catalog extract, duplicate-screening rules, software versions, and random seed. The catalog itself is not distributed here. The code release is intended to make the preprocessing, modeling, and evaluation logic inspectable and reusable without publishing restricted or large data files.
