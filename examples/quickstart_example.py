"""
Quickstart Example: Train and Evaluate ConvLSTM Earthquake Forecasting Model

This script demonstrates the complete workflow from data download to evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from eqgrid import download, build_tensors, train, evaluation
from eqgrid.models.convlstm import ConvLSTM

# Configuration
CONFIG = {
    # Data parameters
    'region': {
        'lon_min': 100.0,
        'lon_max': 115.0,
        'lat_min': 25.0,
        'lat_max': 45.0,
    },
    'time_range': {
        'start': '2000-01-01',
        'end': '2025-12-31',
    },
    'magnitude_threshold': 4.0,
    'grid_size': 1.0,  # degrees
    'lookback': 12,  # months
    'horizon': 1,  # month

    # Model parameters
    'model': {
        'input_channels': 3,
        'hidden_channels': [32, 16],
        'kernel_size': (3, 3),
        'num_layers': 2,
    },

    # Training parameters
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'pos_weight': 83,  # Based on ~1.2% positive rate
        'early_stopping_patience': 20,
    },

    # Paths
    'data_dir': Path('data'),
    'output_dir': Path('outputs/quickstart'),
    'seed': 42,
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("="*80)
    print("Earthquake Forecasting Quickstart Example")
    print("="*80)

    # Set seed for reproducibility
    set_seed(CONFIG['seed'])

    # Create directories
    CONFIG['data_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

    # Step 1: Download data (if not already exists)
    catalog_path = CONFIG['data_dir'] / 'raw' / 'catalog.csv'
    if not catalog_path.exists():
        print("\nStep 1: Downloading earthquake catalog from USGS...")
        print(f"  Region: {CONFIG['region']}")
        print(f"  Time: {CONFIG['time_range']['start']} to {CONFIG['time_range']['end']}")
        print(f"  Magnitude: M >= {CONFIG['magnitude_threshold']}")

        # Note: This is a placeholder - implement actual download using eqgrid.download
        print("  [Download would happen here - see src/eqgrid/download.py]")
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\nStep 1: Using existing catalog at {catalog_path}")

    # Step 2: Preprocess data
    processed_dir = CONFIG['data_dir'] / 'processed' / f"splits_L{CONFIG['lookback']}_H{CONFIG['horizon']}"
    if not (processed_dir / 'train_X.npy').exists():
        print("\nStep 2: Preprocessing data...")
        print(f"  Lookback window: {CONFIG['lookback']} months")
        print(f"  Forecast horizon: {CONFIG['horizon']} month")
        print(f"  Grid size: {CONFIG['grid_size']}°")

        # Note: This is a placeholder - implement using eqgrid.build_tensors
        print("  [Preprocessing would happen here - see src/eqgrid/build_tensors.py]")
        processed_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\nStep 2: Using existing processed data at {processed_dir}")

    # Step 3: Create model
    print("\nStep 3: Creating ConvLSTM model...")
    model = ConvLSTM(**CONFIG['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 4: Train model
    print("\nStep 4: Training model...")
    print("  [Training loop would happen here - see run_training.py]")
    print(f"  Epochs: {CONFIG['training']['epochs']}")
    print(f"  Batch size: {CONFIG['training']['batch_size']}")
    print(f"  Learning rate: {CONFIG['training']['lr']}")
    print(f"  Positive class weight: {CONFIG['training']['pos_weight']}")

    # For demonstration, we'll skip actual training
    # In practice, use run_training.py or implement training loop here

    # Step 5: Evaluate
    print("\nStep 5: Evaluating on test set...")
    print("  [Evaluation would happen here - see scripts/run_comprehensive_evaluation.py]")
    print("\n  Metrics to compute:")
    print("    - ROC-AUC, PR-AUC")
    print("    - Brier Score, Log Loss")
    print("    - ECE, MCE")
    print("    - Bootstrap confidence intervals (1000 iterations)")
    print("    - Paired significance tests")
    print("    - Calibration (Platt scaling, Isotonic regression)")

    # Step 6: Visualize
    print("\nStep 6: Generating figures...")
    print("  [Visualization would happen here - see scripts/plot_*.py]")
    print("  Figures to generate:")
    print("    - ROC and PR curves")
    print("    - Reliability diagrams")
    print("    - Top-K alarm curves")
    print("    - Prediction maps")

    print("\n" + "="*80)
    print("Quickstart Example Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Download real data using: python -m eqgrid.download")
    print("2. Preprocess using: python -m eqgrid.build_tensors")
    print("3. Train using: python run_training.py")
    print("4. Evaluate using: python scripts/run_comprehensive_evaluation.py")
    print("\nSee README.md for detailed instructions.")


if __name__ == '__main__':
    main()
