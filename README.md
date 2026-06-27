# eqgrid: Monthly Gridded Earthquake Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for:

**Benchmarking ConvLSTM Against Spatial-Rate and Aftershock-Decay Baselines for Monthly Gridded Earthquake Forecasting in Southwestern China**

Authors: Feng Gao, Mei Li, Yongsheng Li, Shuang Liu, and Changsheng Liu.

Manuscript status: prepared for submission to *Seismological Research Letters*.

Zenodo DOI: [10.5281/zenodo.20926491](https://doi.org/10.5281/zenodo.20926491).

## Overview

This repository contains the public code for a leakage-controlled benchmark of monthly gridded earthquake occurrence forecasting in Southwestern China. The workflow converts an earthquake catalog into monthly 1 degree by 1 degree tensors, builds 12-month supervised windows, trains a ConvLSTM forecaster, and evaluates it against spatial-rate and aftershock-decay baselines.

The repository does not include the CENC earthquake catalog, processed tensors, model checkpoints, or prediction arrays. Users must obtain catalog data from the appropriate data provider and place it locally under ignored data directories.

## What Is Included

```text
src/eqgrid/                         Core Python package
src/eqgrid/models/convlstm.py       ConvLSTM model
src/eqgrid/tensors_enhanced.py      Ten-channel tensor construction
src/eqgrid/baselines.py             SPR, persistence, KDE, and DAR baselines
src/eqgrid/evaluation.py            Metrics, bootstrap, and calibration utilities
scripts/build_china_depth_filtered_tensors.py
                                    CENC catalog preprocessing for the manuscript workflow
scripts/make_channel_ablation_splits.py
scripts/summarize_channel_ablation.py
configs/convlstm_default.json       Default ConvLSTM training configuration
docs/REPRODUCIBILITY.md             Step-by-step reproduction notes
examples/                           Minimal usage examples
```

## Installation

```bash
git clone https://github.com/cacity/earthquake-forecasting-convlstm.git
cd earthquake-forecasting-convlstm
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On Linux or macOS, use `source venv/bin/activate` instead of `venv\Scripts\activate`.

## Data Policy

The manuscript uses an earthquake catalog obtained from the China Earthquake Networks Center (CENC) through the National Earthquake Data Center service. The catalog is not redistributed in this repository.

The public preprocessing script expects local catalog files in a private data directory. For the manuscript workflow, the inputs were:

```text
data/private/*.EQT
data/private/eq_20090101_*.parquet
```

These paths are examples only. The `data/` directory is ignored by Git.

## Reproducing The Main Workflow

The commands below reproduce the code path used for the manuscript, provided that the required catalog files are available locally.

### 1. Build Ten-Channel Monthly Tensors

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

This step applies the manuscript catalog policy: records are harmonized into a common `depth_km` field; table/parquet records are used from 2009 onward; duplicate events are removed; records with depth equal to 0 km are retained as unspecified-depth entries for occurrence counts and labels; and events with known depth greater than 70 km are excluded.

### 2. Create Supervised Windows

```bash
python -m eqgrid.make_samples ^
  --x data/processed_china_10ch_d0_70_corrected/X.npy ^
  --y data/processed_china_10ch_d0_70_corrected/Y.npy ^
  --bins data/processed_china_10ch_d0_70_corrected/bins.npy ^
  --out-dir data/processed_china_10ch_d0_70_corrected/samples_L12_H1 ^
  --lookback 12 ^
  --horizon 1
```

### 3. Build Chronological Splits

```bash
python -m eqgrid.make_splits ^
  --samples-dir data/processed_china_10ch_d0_70_corrected/samples_L12_H1 ^
  --out-dir data/processed_china_10ch_d0_70_corrected/splits_L12_H1 ^
  --train-end 2018-12-01 ^
  --val-end 2022-12-01 ^
  --require-non-empty
```

### 4. Train ConvLSTM

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

The training code computes the positive-class weight from the training split only, selects the threshold on the validation split, and writes test predictions, logits, metrics, and threshold scans to the output directory.

## Feature Channels

The ten input channels are:

1. `count_norm`
2. `max_mag_norm`
3. `log_energy`
4. `b_value`
5. `mean_mag`
6. `mag_std`
7. `mean_depth`
8. `recency_weight`
9. `rate_change`
10. `spatial_density`

The main leakage-control rule is that no full-catalog z-score normalization is fitted. `rate_change` and `spatial_density` use fixed analytic bounds. Sparse monthly cell features should be interpreted as auxiliary covariates, not stable source-parameter estimates.

## Baselines

The benchmark includes:

```text
SPR          Stationary Poisson Rate baseline
Persistence Last-month activity baseline
KDE          Gaussian-smoothed spatial-rate baseline
DAR          Omori-decay aftershock-rate baseline
ConvLSTM     Neural spatiotemporal benchmark
```

DAR is an Omori-decay baseline, not a full ETAS maximum-likelihood implementation.

## Manuscript-Scale Reference Values

For the depth-filtered CENC catalog used in the manuscript, with zero-depth entries treated as unspecified-depth records, the final retained catalog contains 3,814 duplicate-removed M>=4.0 events. The grid has 195 cells, and the 2023-2025 test set contains 7,020 cell-months with 109 positive cell-months.

The benchmark does not show clear ConvLSTM superiority. In the manuscript run, ConvLSTM achieved ROC-AUC about 0.640 and PR-AUC about 0.041. DAR had the highest PR-AUC, while KDE had the highest ROC-AUC and the lowest clipped log loss.

## Repository Hygiene

The following are intentionally ignored and should not be committed:

```text
data/
outputs/
paper_evaluation_results/
figures/data/
*.npy, *.npz, *.parquet, *.pt, *.pth, *.ckpt
*.xlsx, *.xls, *.csv, *.geojson, *.docx, *.pdf
```

## Citation

If you use this code, cite this repository and the accompanying manuscript:

```bibtex
@software{gao_eqgrid_2026,
  author = {Gao, Feng and Li, Mei and Li, Yongsheng and Liu, Shuang and Liu, Changsheng},
  title = {eqgrid: Grid-Based Earthquake Forecasting Package},
  year = {2026},
  version = {V2.0.0},
  doi = {10.5281/zenodo.20926491},
  url = {https://github.com/cacity/earthquake-forecasting-convlstm}
}
```

## License

This project is released under the MIT License.
