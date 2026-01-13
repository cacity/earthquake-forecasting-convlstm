# Reproducibility Guide

This document provides detailed instructions for reproducing all results from the IEEE Access paper.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Figure Generation](#figure-generation)
6. [Verification](#verification)

## Environment Setup

### Hardware Requirements

**Recommended**:
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA GPU with 8+ GB VRAM (optional but recommended)
- Storage: 10+ GB free space

**Tested on**:
- Ubuntu 20.04 LTS
- NVIDIA Tesla V100 (16GB)
- CUDA 11.8, cuDNN 8.6

### Software Dependencies

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install exact versions used in paper
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Preparation

### Step 1: Download USGS Catalog

```bash
python -m eqgrid.download \
    --start 2000-01-01 \
    --end 2025-12-31 \
    --minmag 4.0 \
    --lon_min 100 --lon_max 115 \
    --lat_min 25 --lat_max 45 \
    --output data/raw/catalog.csv
```

**Expected output**:
- File: `data/raw/catalog.csv`
- Events: ~3,587 earthquakes
- Columns: time, latitude, longitude, depth, mag, magType, ...

### Step 2: Build Feature Tensors

```bash
python -m eqgrid.build_tensors \
    --catalog data/raw/catalog.csv \
    --lookback 12 \
    --horizon 1 \
    --grid_size 1.0 \
    --lon_min 100 --lon_max 115 \
    --lat_min 25 --lat_max 45 \
    --output data/processed/
```

**Expected output**:
- Directory: `data/processed/splits_L12_H1/`
- Files: `train_X.npy`, `train_Y.npy`, `train_TS.npy`, etc.
- Train samples: 216 (2001-2018)
- Val samples: 48 (2019-2022)
- Test samples: 36 (2023-2025)
- Shape: X: [N, 12, 3, 20, 15], Y: [N, 1, 20, 15]

### Step 3: Verify Data Integrity

```bash
python -m eqgrid.check_splits \
    --data_dir data/processed/splits_L12_H1/
```

**Expected checks**:
- ✓ No temporal leakage between splits
- ✓ Chronological ordering
- ✓ Positive rate ~1.2%
- ✓ No NaN or inf values
- ✓ Correct shapes

## Model Training

### Training ConvLSTM (Main Model)

```bash
python run_training.py \
    --config configs/convlstm_default.json \
    --data_dir data/processed/splits_L12_H1 \
    --output_dir outputs/convlstm_L12 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --pos_weight 83 \
    --scheduler_patience 10 \
    --scheduler_factor 0.5 \
    --early_stopping_patience 20 \
    --seed 42 \
    --device cuda
```

**Expected training time**: ~30-60 minutes on V100 GPU

**Expected convergence**:
- Best epoch: ~40-60
- Final val PR-AUC: ~0.10-0.11
- Final val ROC-AUC: ~0.80-0.82

**Outputs**:
- `outputs/convlstm_L12/ckpt_best.pt` - Best model checkpoint
- `outputs/convlstm_L12/history.json` - Training history
- `outputs/convlstm_L12/test_preds.npy` - Test predictions
- `outputs/convlstm_L12/test_logits.npy` - Test logits (pre-sigmoid)
- `outputs/convlstm_L12/val_logits.npy` - Validation logits

### Training Baselines

#### CNN-only (No temporal modeling)

```bash
python run_training.py \
    --config configs/cnn_only.json \
    --data_dir data/processed/splits_L12_H1 \
    --output_dir outputs/cnn_only \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --pos_weight 83 \
    --seed 42
```

#### Historical Rate and Kernel Density

These don't require training - computed during evaluation.

## Evaluation

### Comprehensive Evaluation (All Metrics)

This script reproduces **Table I** and **Table II** from the paper.

```bash
python scripts/fix_paper_metrics_unified.py
```

**Runtime**: ~15-30 minutes (1000 bootstrap iterations)

**Outputs**:
- `paper_results_unified/unified_metrics.json` - All metrics with CIs
- `paper_results_unified/computation_log.txt` - Full log

**Expected results** (should match paper Table I):

| Method | ROC-AUC | PR-AUC | Brier |
|--------|---------|--------|-------|
| ConvLSTM | 0.8110 [0.7603, 0.8649] | 0.0973 [0.0678, 0.1457] | 0.1385 |
| Historical Rate | 0.8409 [0.7961, 0.8678] | 0.0669 [0.0483, 0.0941] | 0.0116 |

**Calibration results** (should match paper Table II):

| Method | Brier | ECE | MCE (n≥50) | ROC-AUC |
|--------|-------|-----|------------|---------|
| Uncalibrated | 0.1385 | 0.2959 | 0.8063 | 0.8110 |
| Platt | 0.0114 | 0.0010 | 0.0367 | 0.8110 |
| Isotonic | 0.0114 | 0.0014 | 0.0568 | 0.8152 |

### Verification Checklist

After running evaluation, verify:

```bash
# Check that values match paper
python scripts/verify_results.py \
    --results paper_results_unified/unified_metrics.json \
    --tolerance 0.001
```

Expected output:
```
✓ ConvLSTM ROC-AUC: 0.8110 (within tolerance)
✓ ConvLSTM PR-AUC: 0.0973 (within tolerance)
✓ Historical Rate ROC-AUC: 0.8409 (within tolerance)
✓ Calibrated Brier: 0.0114 (within tolerance)
✓ All metrics verified!
```

## Figure Generation

### Figure 5: ROC and PR Curves

```bash
python scripts/plot_main_pr_baseline.py \
    --pred_path outputs/convlstm_L12/test_preds.npy \
    --label_path data/processed/splits_L12_H1/test_Y.npy \
    --output figures/fig5_roc_pr_curves.pdf
```

### Figure 6: Reliability Diagrams

```bash
python scripts/plot_calibration_reliability.py \
    --pred_path outputs/convlstm_L12/test_preds.npy \
    --logits_path outputs/convlstm_L12/test_logits.npy \
    --label_path data/processed/splits_L12_H1/test_Y.npy \
    --val_pred_path outputs/convlstm_L12/val_preds.npy \
    --val_logits_path outputs/convlstm_L12/val_logits.npy \
    --val_label_path data/processed/splits_L12_H1/val_Y.npy \
    --output figures/fig6_reliability_diagrams.pdf
```

### Figure 7: Top-K Alarm Performance

```bash
python scripts/plot_alarm_skill_molchan.py \
    --pred_path outputs/convlstm_L12/test_preds.npy \
    --label_path data/processed/splits_L12_H1/test_Y.npy \
    --output figures/fig7_alarm_skill.pdf
```

## Verification

### Final Checklist

Run this to verify all results match the paper:

```bash
bash scripts/verify_reproducibility.sh
```

This checks:
- [ ] Data statistics match (3,587 events, 36 test months, etc.)
- [ ] Model architecture matches (15K parameters)
- [ ] Training converged properly (val PR-AUC ~0.10-0.11)
- [ ] Test metrics match paper (within 0.001 tolerance):
  - [ ] ConvLSTM ROC-AUC: 0.8110 ± 0.001
  - [ ] ConvLSTM PR-AUC: 0.0973 ± 0.001
  - [ ] Historical Rate ROC-AUC: 0.8409 ± 0.001
  - [ ] Calibrated Brier: 0.0114 ± 0.001
- [ ] Bootstrap CIs match (width within 10%)
- [ ] Figures generated successfully

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size
python run_training.py --batch_size 8 ...
```

**2. Different results (minor variations)**

Small numerical differences (<0.001) are expected due to:
- GPU non-determinism (even with seeds)
- PyTorch version differences
- Hardware differences (CPU vs GPU)

These should not affect conclusions.

**3. Bootstrap takes too long**

```bash
# Reduce iterations for testing (not for paper)
python scripts/fix_paper_metrics_unified.py --n_bootstrap 100
```

**4. Missing validation logits**

If validation logits aren't saved, run:
```bash
python scripts/generate_validation_logits.py \
    --checkpoint outputs/convlstm_L12/ckpt_best.pt \
    --data_dir data/processed/splits_L12_H1 \
    --output outputs/convlstm_L12/val_logits.npy
```

## Contact

If you encounter issues reproducing results:
1. Check [GitHub Issues](https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm/issues)
2. Email: your.email@example.com

## Computational Resources

**Total time to reproduce all results** (on V100 GPU):
- Data download: ~5 minutes
- Preprocessing: ~10 minutes
- Training ConvLSTM: ~45 minutes
- Training baselines: ~30 minutes
- Evaluation (1000 bootstrap): ~20 minutes
- Figure generation: ~15 minutes
- **Total: ~2-3 hours**

**Storage requirements**:
- Raw catalog: ~2 MB
- Processed tensors: ~50 MB
- Model checkpoints: ~1 MB
- Results: ~10 MB
- Figures: ~20 MB
- **Total: ~100 MB**
