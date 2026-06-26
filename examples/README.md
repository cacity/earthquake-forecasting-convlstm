# Examples

These examples show how to run the public code package. They do not include raw earthquake catalog data.

## Minimal Manuscript-Style Workflow

Place the local CENC catalog files in an ignored directory:

```text
data/private/
```

Then run:

```bash
python run_training.py --data-dir data/private --epochs 50 --seed 42
```

This command builds corrected depth-filtered ten-channel tensors, creates 12-month supervised windows, builds chronological train/validation/test splits, and trains the ConvLSTM model.

## Manual Steps

For greater control, run the commands in `docs/REPRODUCIBILITY.md` one by one.

## Data Reminder

Do not commit raw catalogs, processed tensors, checkpoints, or prediction arrays. The repository `.gitignore` excludes these files by default.
