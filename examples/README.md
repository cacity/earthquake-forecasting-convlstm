# Examples

This directory contains example scripts and tutorials for using the earthquake forecasting framework.

## Available Examples

### 1. quickstart_example.py

**Description**: Complete workflow demonstration from data download to evaluation.

**Usage**:
```bash
python examples/quickstart_example.py
```

**What it covers**:
- Environment setup
- Data download and preprocessing
- Model creation and training
- Evaluation and visualization
- Best practices

**Runtime**: ~5 minutes (demonstration mode)

## Coming Soon

### Jupyter Notebooks
- [ ] `tutorial_01_data_preparation.ipynb` - Data download and preprocessing
- [ ] `tutorial_02_model_training.ipynb` - Training ConvLSTM from scratch
- [ ] `tutorial_03_evaluation.ipynb` - Comprehensive evaluation
- [ ] `tutorial_04_visualization.ipynb` - Creating publication-quality figures
- [ ] `tutorial_05_custom_region.ipynb` - Adapting to new geographic regions

### Advanced Examples
- [ ] `custom_architecture.py` - Modifying the ConvLSTM architecture
- [ ] `hyperparameter_tuning.py` - Systematic hyperparameter optimization
- [ ] `ensemble_forecasting.py` - Combining multiple models
- [ ] `real_time_forecasting.py` - Operational forecasting workflow

## Quick Start Template

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eqgrid.models.convlstm import ConvLSTM
import torch

# 1. Create model
model = ConvLSTM(
    input_channels=3,
    hidden_channels=[32, 16],
    kernel_size=(3, 3),
    num_layers=2
)

# 2. Load data (implement your data loader)
# train_loader, val_loader = create_dataloaders(...)

# 3. Train
# trainer = Trainer(model, ...)
# trainer.fit(train_loader, val_loader)

# 4. Evaluate
# results = evaluate(model, test_loader)

print("Model ready!")
```

## Data Requirements

All examples assume data is organized as:

```
data/
├── raw/
│   └── catalog.csv         # USGS catalog
└── processed/
    └── splits_L12_H1/      # Preprocessed tensors
        ├── train_X.npy
        ├── train_Y.npy
        └── ...
```

Download data using:
```bash
python -m eqgrid.download --start 2000-01-01 --end 2025-12-31 ...
python -m eqgrid.build_tensors --catalog data/raw/catalog.csv ...
```

## Tips

1. **Start simple**: Run `quickstart_example.py` first
2. **Check README**: Main README has detailed setup instructions
3. **Read docs**: See `docs/REPRODUCIBILITY.md` for step-by-step guide
4. **Use GPU**: Training is much faster on GPU
5. **Adjust batch size**: Reduce if you get CUDA out of memory errors

## Support

- **Issues**: https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm/issues
- **Email**: your.email@example.com

## Contributing

Have a useful example? Please submit a pull request!

Guidelines:
- Clear documentation
- Minimal dependencies
- Well-commented code
- Test before submitting
