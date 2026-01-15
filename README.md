# Grid-Based Earthquake Forecasting with ConvLSTM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18241723.svg)](https://doi.org/10.5281/zenodo.18241723)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official code repository for**: "Grid-Based Monthly Earthquake Forecasting Using Convolutional Long Short-Term Memory Networks: A Reproducible Framework for Southwestern China"

Published in: *IEEE Access* (2026)

## Overview

This repository provides a complete, reproducible framework for spatiotemporal earthquake forecasting using deep learning. The code implements a ConvLSTM-based model that predicts monthly earthquake occurrence probabilities on a 1°×1° spatial grid.

### Key Features

- **End-to-end pipeline**: Data download → preprocessing → training → evaluation → visualization
- **Rigorous evaluation**: Bootstrap confidence intervals, paired significance tests, probability calibration
- **Reproducible**: Fixed seeds, comprehensive logging, version-controlled dependencies
- **Well-documented**: Detailed comments, type hints, example scripts
- **Publication-ready figures**: All figures from the paper can be regenerated

### Performance Highlights

On 36-month test set (2023-2025):
- **PR-AUC**: 0.0973 (45% improvement over spatial baseline)
- **ROC-AUC**: 0.8110
- **Calibration**: ECE 0.0010 after Platt scaling (99.7% improvement)
- **Top-K**: 35.9% hit rate at 3% alarm area (12× over random)

## Repository Structure

```
.
├── src/eqgrid/              # Core library
│   ├── download.py          # USGS catalog download
│   ├── build_tensors.py     # Feature engineering
│   ├── models/              # Neural network architectures
│   │   ├── convlstm.py      # ConvLSTM model
│   │   └── cnn_only.py      # CNN baseline
│   ├── train.py             # Training loop
│   ├── evaluation.py        # Metrics and calibration
│   ├── baselines.py         # Baseline forecasting methods
│   └── ...                  # Other utilities
├── scripts/                 # Analysis and visualization
│   ├── run_comprehensive_evaluation.py
│   ├── fix_paper_metrics_unified.py
│   ├── compute_robust_calibration_metrics.py
│   └── plot_*.py            # Figure generation
├── examples/                # Tutorial notebooks and scripts
├── configs/                 # Model configuration files
├── requirements.txt         # Python dependencies
├── run_training.py          # Main training script
└── README.md                # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm.git
cd earthquake-forecasting-convlstm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Download Data

```bash
python -m eqgrid.download \
    --start 2000-01-01 \
    --end 2025-12-31 \
    --minmag 4.0 \
    --lon_min 100 --lon_max 115 \
    --lat_min 25 --lat_max 45 \
    --output data/raw/catalog.csv
```

### 2. Preprocess Data

```bash
python -m eqgrid.build_tensors \
    --catalog data/raw/catalog.csv \
    --lookback 12 \
    --horizon 1 \
    --grid_size 1.0 \
    --output data/processed/
```

### 3. Train Model

```bash
python run_training.py \
    --config configs/convlstm_default.json \
    --data_dir data/processed/splits_L12_H1 \
    --output_dir outputs/convlstm_L12 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --pos_weight 83 \
    --seed 42
```

### 4. Evaluate

```bash
python scripts/run_comprehensive_evaluation.py \
    --pred_path outputs/convlstm_L12/preds_test.npy \
    --label_path data/processed/splits_L12_H1/test_Y.npy \
    --output_dir evaluation_results/
```

### 5. Generate Figures

```bash
python scripts/plot_calibration_reliability.py \
    --pred_path outputs/convlstm_L12/preds_test.npy \
    --label_path data/processed/splits_L12_H1/test_Y.npy \
    --output figures/reliability_diagrams.png
```

## Reproducibility

### Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# 1. Use the unified metrics computation script
python scripts/fix_paper_metrics_unified.py

# Output will be saved to:
# - paper_results_unified/unified_metrics.json
# - paper_results_unified/computation_log.txt
```

This script computes:
- All metrics with monthly block bootstrap (1000 iterations)
- Paired significance tests
- Calibration metrics with proper bin filtering
- Exactly matches Table I and Table II in the paper

### Key Parameters for Reproducibility

All experiments use:
- **Random seed**: 42 (PyTorch, NumPy, Python)
- **Data splits**: Chronological (train 2001-2018, val 2019-2022, test 2023-2025)
- **Grid resolution**: 1° × 1° (300 cells: 20 lat × 15 lon)
- **Temporal resolution**: Monthly
- **Lookback window**: 12 months
- **Forecast horizon**: 1 month
- **Magnitude threshold**: M ≥ 4.0
- **Positive class weight**: 83 (based on ~1.2% positive rate)

## Data

### Earthquake Catalog

Data is downloaded from the USGS Earthquake Catalog:
- **Source**: https://earthquake.usgs.gov/fdsnws/event/1/
- **Time range**: 2000-01-01 to 2025-12-31 (26 years)
- **Region**: Southwestern China (100°E-115°E, 25°N-45°N)
- **Magnitude**: M ≥ 4.0
- **Events**: ~3,587 earthquakes

**Citation**: United States Geological Survey (USGS). (2025). USGS Earthquake Catalog. Retrieved from https://earthquake.usgs.gov/

### Data Structure

After preprocessing, data is organized as:

```
data/processed/splits_L12_H1/
├── train_X.npy    # Training features [N, L=12, C=3, H=20, W=15]
├── train_Y.npy    # Training labels [N, 1, H=20, W=15]
├── train_TS.npy   # Training timestamps [N]
├── val_X.npy      # Validation features
├── val_Y.npy      # Validation labels
├── val_TS.npy     # Validation timestamps
├── test_X.npy     # Test features
├── test_Y.npy     # Test labels
└── test_TS.npy    # Test timestamps
```

**Feature channels** (C=3):
1. Normalized earthquake frequency
2. Maximum magnitude
3. Log cumulative energy

## Model Architecture

### ConvLSTM Network

```python
ConvLSTM(
  input_channels=3,
  hidden_channels=[32, 16],
  kernel_size=(3, 3),
  num_layers=2,
  output_channels=1,
  activation='tanh'
)
```

**Architecture details**:
- 2 ConvLSTM layers (32→16 hidden channels)
- 3×3 spatial kernels with padding=1
- Tanh activation in LSTM cells
- Final 1×1 conv → sigmoid for probability output
- Total parameters: ~15K

**Training**:
- Loss: Weighted binary cross-entropy (pos_weight=83)
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Early stopping: 20 epochs (validation PR-AUC)
- Batch size: 16
- Epochs: 100 (typically converges ~40-60 epochs)

## Evaluation Metrics

### Discrimination Metrics
- **ROC-AUC**: Area under ROC curve (ranking quality)
- **PR-AUC**: Precision-Recall AUC (imbalanced class performance)
- **F1 Score**: Harmonic mean of precision and recall
- **Top-K Hit Rate**: Fraction of positives captured in top-K predictions

### Calibration Metrics
- **Brier Score**: Mean squared error of probabilities
- **ECE**: Expected Calibration Error (10 bins)
- **MCE**: Maximum Calibration Error (with bin filtering for small samples)
- **Log Loss**: Negative log-likelihood

### Statistical Rigor
- **Bootstrap CI**: 1000 iterations, monthly block resampling
- **Paired Tests**: Paired bootstrap difference tests
- **Calibration**: Platt scaling, Isotonic regression, Temperature scaling

## Baselines

The repository includes implementations of:

1. **Historical Rate**: Time-invariant spatial probability map
2. **Kernel Density**: Gaussian-smoothed seismicity (σ=0.5, 1.0)
3. **Persistence**: Last month's activity as forecast
4. **CNN-only**: Spatial-only baseline (no temporal modeling)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your2026earthquake,
  title={Grid-Based Monthly Earthquake Forecasting Using Convolutional Long Short-Term Memory Networks: A Reproducible Framework for Southwestern China},
  author={Your Name and Coauthors},
  journal={IEEE Access},
  year={2026},
  volume={XX},
  pages={XXXXX--XXXXX},
  doi={10.1109/ACCESS.2026.XXXXXXX}
}
```

For the code repository:

```bibtex
@software{your2026code,
  author={Your Name},
  title={earthquake-forecasting-convlstm: Grid-Based Earthquake Forecasting},
  year={2026},
  publisher={Zenodo},
  version={v1.0.0},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://github.com/cacity/earthquake-forecasting-convlstm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
black src/ scripts/
flake8 src/ scripts/
```

## Support

- **Issues**: Please use the [GitHub Issues](https://github.com/cacity/earthquake-forecasting-convlstm/issues) tracker
- **Email**: gf7823332@gmail.com

## Acknowledgments

This research was supported by [Your Funding Sources].

We thank:
- USGS for providing earthquake catalog data
- [Other acknowledgments]

## Changelog

### Version 1.0.0 (2026-01-14)
- Initial release
- Code accompanying IEEE Access publication
- Includes all experiments and figures from the paper

---

**Last updated**: January 2026
