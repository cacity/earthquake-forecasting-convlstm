# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-XX

### Added

- Initial release accompanying IEEE Access publication
- Complete ConvLSTM implementation for earthquake forecasting
- End-to-end pipeline from data download to evaluation
- Comprehensive evaluation framework:
  - Bootstrap confidence intervals (monthly block resampling)
  - Paired significance tests
  - Probability calibration (Platt, Isotonic, Temperature)
  - Multiple baselines (Historical Rate, Kernel Density, Persistence, CNN-only)
- Visualization scripts for all paper figures
- Reproducibility documentation and example scripts
- CITATION.cff for Zenodo integration
- MIT License

### Core Modules

- `src/eqgrid/download.py` - USGS catalog download
- `src/eqgrid/build_tensors.py` - Feature engineering and data preprocessing
- `src/eqgrid/models/convlstm.py` - Main ConvLSTM architecture
- `src/eqgrid/train.py` - Training loop with early stopping
- `src/eqgrid/evaluation.py` - Metrics, calibration, statistical tests
- `src/eqgrid/baselines.py` - Baseline forecasting methods

### Scripts

- `run_training.py` - Main training script
- `scripts/fix_paper_metrics_unified.py` - Reproduce paper Tables I & II
- `scripts/run_comprehensive_evaluation.py` - Full evaluation pipeline
- `scripts/plot_*.py` - Figure generation scripts
- `examples/quickstart_example.py` - Tutorial example

### Documentation

- `README.md` - Main documentation
- `docs/REPRODUCIBILITY.md` - Step-by-step reproduction guide
- `CITATION.cff` - Citation metadata for Zenodo
- `LICENSE` - MIT License

### Configuration

- `configs/convlstm_default.json` - Default model configuration
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation script

## [Unreleased]

### Planned

- Jupyter notebook tutorials
- Pre-trained model weights
- Docker container for reproducibility
- Extended to other regions (California, Japan, etc.)
- Real-time forecasting interface
- Web-based visualization dashboard

---

## Version History

- **v1.0.0** (2026-01-XX): Initial public release with IEEE Access paper
