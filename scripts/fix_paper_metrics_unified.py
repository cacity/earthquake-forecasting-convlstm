"""
Fix paper metrics by computing unified, consistent results for all tables.

This script:
1. Loads all test predictions and labels
2. Computes metrics with monthly-block bootstrap CI
3. Computes paired bootstrap significance tests
4. Recalculates calibration metrics with proper bin filtering
5. Generates unified results for Table I and Table II
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eqgrid import evaluation, baselines
from sklearn.isotonic import IsotonicRegression
from scipy.special import logit as sp_logit


def bootstrap_metric_monthly(y_true_per_month: list,
                             y_score_per_month: list,
                             metric_fn,
                             n_bootstrap: int = 1000,
                             confidence_level: float = 0.95,
                             seed: int = 42):
    """
    Bootstrap by resampling entire months (blocks), preserving spatial structure.

    Args:
        y_true_per_month: List of arrays, one per month
        y_score_per_month: List of arrays, one per month
        metric_fn: Function that takes (y_true, y_score) and returns scalar
        n_bootstrap: Number of bootstrap iterations
        confidence_level: CI level (default 0.95)
        seed: Random seed

    Returns:
        Dict with 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    rng = np.random.RandomState(seed)
    n_months = len(y_true_per_month)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample months with replacement
        month_indices = rng.choice(n_months, size=n_months, replace=True)

        # Concatenate resampled months
        y_true_boot = np.concatenate([y_true_per_month[i] for i in month_indices])
        y_score_boot = np.concatenate([y_score_per_month[i] for i in month_indices])

        try:
            score = metric_fn(y_true_boot, y_score_boot)
            if not np.isnan(score):
                bootstrap_scores.append(score)
        except:
            pass

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return {
        'mean': float(np.mean(bootstrap_scores)),
        'std': float(np.std(bootstrap_scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


def paired_bootstrap_test_monthly(y_true_per_month: list,
                                 y_score1_per_month: list,
                                 y_score2_per_month: list,
                                 metric_fn,
                                 n_bootstrap: int = 1000,
                                 seed: int = 42):
    """
    Paired bootstrap test by resampling months.

    Tests whether metric(score1) > metric(score2) significantly.

    Returns:
        Dict with 'observed_diff', 'p_value', 'ci_lower', 'ci_upper'
    """
    # Observed difference
    y_true_all = np.concatenate(y_true_per_month)
    y_score1_all = np.concatenate(y_score1_per_month)
    y_score2_all = np.concatenate(y_score2_per_month)

    observed_diff = metric_fn(y_true_all, y_score1_all) - metric_fn(y_true_all, y_score2_all)

    # Bootstrap
    rng = np.random.RandomState(seed)
    n_months = len(y_true_per_month)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        month_indices = rng.choice(n_months, size=n_months, replace=True)

        y_true_boot = np.concatenate([y_true_per_month[i] for i in month_indices])
        y_score1_boot = np.concatenate([y_score1_per_month[i] for i in month_indices])
        y_score2_boot = np.concatenate([y_score2_per_month[i] for i in month_indices])

        try:
            s1 = metric_fn(y_true_boot, y_score1_boot)
            s2 = metric_fn(y_true_boot, y_score2_boot)
            if not (np.isnan(s1) or np.isnan(s2)):
                bootstrap_diffs.append(s1 - s2)
        except:
            pass

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    # CI for the difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'significant': bool(p_value < 0.05)
    }


def compute_calibration_metrics_robust(y_true, y_prob, n_bins=10, min_bin_size=50):
    """
    Compute ECE and MCE with filtering for small bins.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        min_bin_size: Minimum samples per bin for MCE calculation

    Returns:
        Dict with 'ece', 'mce', 'mce_filtered' (filtered for large bins only)
    """
    y_true = y_true.reshape(-1).astype(np.float64)
    y_prob = y_prob.reshape(-1).astype(np.float64)

    # Equal-width binning
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    mce_filtered = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        n_i = mask.sum()

        if n_i > 0:
            acc_i = y_true[mask].mean()
            conf_i = y_prob[mask].mean()

            # ECE: weighted by bin size
            ece += (n_i / len(y_true)) * abs(acc_i - conf_i)

            # MCE: max error across all bins
            mce = max(mce, abs(acc_i - conf_i))

            # MCE filtered: only bins with sufficient samples
            if n_i >= min_bin_size:
                mce_filtered = max(mce_filtered, abs(acc_i - conf_i))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'mce_filtered': float(mce_filtered),
        'n_bins': n_bins,
        'min_bin_size': min_bin_size
    }


def main():
    print("="*80)
    print("UNIFIED METRICS COMPUTATION FOR PAPER")
    print("="*80)

    # Paths
    data_dir = Path("/mnt/f/work/网格地震预测/data/processed/splits_L12_H1")
    output_dir = Path("/mnt/f/work/网格地震预测/outputs/convlstm")
    results_dir = Path("/mnt/f/work/网格地震预测/paper_results_unified")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    print("\n1. Loading test data...")
    y_test = np.load(data_dir / "test_Y.npy")  # [N, 1, H, W]
    ts_test = np.load(data_dir / "test_TS.npy")  # [N]

    # Load ConvLSTM predictions
    preds_test = np.load(output_dir / "test_preds.npy")  # [N, 1, H, W]
    logits_test = np.load(output_dir / "test_logits.npy")  # [N, 1, H, W]

    # Load validation data
    y_val = np.load(data_dir / "val_Y.npy")
    logits_val = np.load(output_dir / "val_logits.npy")
    # Generate val predictions from logits
    preds_val = 1 / (1 + np.exp(-logits_val))

    print(f"  Test shape: {y_test.shape}")
    print(f"  Predictions shape: {preds_test.shape}")
    print(f"  Timestamps shape: {ts_test.shape}")
    print(f"  Date range: {ts_test[0]} to {ts_test[-1]}")
    print(f"  Positive rate: {y_test.mean():.4f}")
    print(f"  Val shape: {y_val.shape}, Val preds shape: {preds_val.shape}")

    # Flatten spatial dimensions, keep monthly structure
    n_months = len(ts_test)
    print(f"\n  Number of months: {n_months}")

    # Organize data by month
    y_true_per_month = []
    preds_per_month = []
    logits_per_month = []

    for i in range(n_months):
        y_true_per_month.append(y_test[i].reshape(-1))
        preds_per_month.append(preds_test[i].reshape(-1))
        logits_per_month.append(logits_test[i].reshape(-1))

    # Full arrays for point estimates
    y_true_flat = np.concatenate(y_true_per_month)
    preds_flat = np.concatenate(preds_per_month)
    logits_flat = np.concatenate(logits_per_month)

    # ========================================================================
    # 2. Compute ConvLSTM metrics with monthly bootstrap
    # ========================================================================
    print("\n2. Computing ConvLSTM metrics...")

    convlstm_results = {}

    # ROC-AUC
    print("  ROC-AUC...")
    roc_point = evaluation.compute_roc_auc(y_true_flat, preds_flat)
    roc_boot = bootstrap_metric_monthly(
        y_true_per_month, preds_per_month,
        evaluation.compute_roc_auc,
        n_bootstrap=1000, seed=42
    )
    convlstm_results['roc_auc'] = {
        'point': float(roc_point),
        **roc_boot
    }
    print(f"    Point: {roc_point:.4f}")
    print(f"    Bootstrap: {roc_boot['mean']:.4f} [{roc_boot['ci_lower']:.4f}, {roc_boot['ci_upper']:.4f}]")

    # PR-AUC
    print("  PR-AUC...")
    pr_point = evaluation.compute_pr_auc(y_true_flat, preds_flat)
    pr_boot = bootstrap_metric_monthly(
        y_true_per_month, preds_per_month,
        evaluation.compute_pr_auc,
        n_bootstrap=1000, seed=42
    )
    convlstm_results['pr_auc'] = {
        'point': float(pr_point),
        **pr_boot
    }
    print(f"    Point: {pr_point:.4f}")
    print(f"    Bootstrap: {pr_boot['mean']:.4f} [{pr_boot['ci_lower']:.4f}, {pr_boot['ci_upper']:.4f}]")

    # Brier Score
    print("  Brier Score...")
    brier_point = evaluation.compute_brier_score(y_true_flat, preds_flat)
    brier_boot = bootstrap_metric_monthly(
        y_true_per_month, preds_per_month,
        evaluation.compute_brier_score,
        n_bootstrap=1000, seed=42
    )
    convlstm_results['brier_score'] = {
        'point': float(brier_point),
        **brier_boot
    }
    print(f"    Point: {brier_point:.4f}")
    print(f"    Bootstrap: {brier_boot['mean']:.4f} [{brier_boot['ci_lower']:.4f}, {brier_boot['ci_upper']:.4f}]")

    # Log Loss
    print("  Log Loss...")
    logloss_point = evaluation.compute_log_loss(y_true_flat, preds_flat)
    logloss_boot = bootstrap_metric_monthly(
        y_true_per_month, preds_per_month,
        evaluation.compute_log_loss,
        n_bootstrap=1000, seed=42
    )
    convlstm_results['log_loss'] = {
        'point': float(logloss_point),
        **logloss_boot
    }
    print(f"    Point: {logloss_point:.4f}")

    # ========================================================================
    # 3. Compute baseline metrics (Historical Rate, Kernel Density, Persistence)
    # ========================================================================
    print("\n3. Computing baseline metrics...")

    baseline_results = {}

    # Historical Rate baseline (spatial map, not scalar)
    print("\n  Historical Rate baseline (spatial)...")
    # Need training data to compute historical rate map
    y_train = np.load(data_dir / "train_Y.npy")

    # Compute spatial historical rate: mean rate per cell over training period
    if y_train.ndim == 4 and y_train.shape[1] == 1:
        y_train_3d = y_train[:, 0, :, :]  # [N, H, W]
    else:
        y_train_3d = y_train

    historical_rate_map = y_train_3d.mean(axis=0)  # [H, W]
    print(f"    Historical rate map shape: {historical_rate_map.shape}")
    print(f"    Historical rate range: [{historical_rate_map.min():.6f}, {historical_rate_map.max():.6f}]")
    print(f"    Historical rate mean: {historical_rate_map.mean():.6f}")

    # Generate predictions for each test month (same spatial map repeated)
    hist_preds_per_month = [historical_rate_map.flatten() for _ in range(n_months)]
    hist_preds_flat = np.tile(historical_rate_map.flatten(), n_months)

    hist_roc = evaluation.compute_roc_auc(y_true_flat, hist_preds_flat)
    hist_pr = evaluation.compute_pr_auc(y_true_flat, hist_preds_flat)
    hist_brier = evaluation.compute_brier_score(y_true_flat, hist_preds_flat)
    hist_logloss = evaluation.compute_log_loss(y_true_flat, hist_preds_flat)

    hist_roc_boot = bootstrap_metric_monthly(
        y_true_per_month, hist_preds_per_month,
        evaluation.compute_roc_auc, n_bootstrap=1000, seed=42
    )
    hist_pr_boot = bootstrap_metric_monthly(
        y_true_per_month, hist_preds_per_month,
        evaluation.compute_pr_auc, n_bootstrap=1000, seed=42
    )
    hist_brier_boot = bootstrap_metric_monthly(
        y_true_per_month, hist_preds_per_month,
        evaluation.compute_brier_score, n_bootstrap=1000, seed=42
    )

    baseline_results['historical_rate'] = {
        'rate_mean': float(historical_rate_map.mean()),
        'rate_range': [float(historical_rate_map.min()), float(historical_rate_map.max())],
        'roc_auc': {'point': float(hist_roc), **hist_roc_boot},
        'pr_auc': {'point': float(hist_pr), **hist_pr_boot},
        'brier_score': {'point': float(hist_brier), **hist_brier_boot},
        'log_loss': {'point': float(hist_logloss)}
    }

    print(f"    ROC-AUC: {hist_roc:.4f} [{hist_roc_boot['ci_lower']:.4f}, {hist_roc_boot['ci_upper']:.4f}]")
    print(f"    PR-AUC: {hist_pr:.4f} [{hist_pr_boot['ci_lower']:.4f}, {hist_pr_boot['ci_upper']:.4f}]")
    print(f"    Brier: {hist_brier:.4f} [{hist_brier_boot['ci_lower']:.4f}, {hist_brier_boot['ci_upper']:.4f}]")

    # ========================================================================
    # 4. Paired bootstrap significance tests
    # ========================================================================
    print("\n4. Paired bootstrap significance tests...")

    significance_results = {}

    # ConvLSTM vs Historical Rate: PR-AUC
    print("\n  ConvLSTM vs Historical Rate (PR-AUC)...")
    pr_sig = paired_bootstrap_test_monthly(
        y_true_per_month, preds_per_month, hist_preds_per_month,
        evaluation.compute_pr_auc, n_bootstrap=1000, seed=42
    )
    significance_results['convlstm_vs_hist_pr'] = pr_sig
    print(f"    Observed diff: {pr_sig['observed_diff']:.4f}")
    print(f"    95% CI: [{pr_sig['ci_lower']:.4f}, {pr_sig['ci_upper']:.4f}]")
    print(f"    p-value: {pr_sig['p_value']:.4f}")
    print(f"    Significant: {pr_sig['significant']}")

    # ConvLSTM vs Historical Rate: ROC-AUC
    print("\n  ConvLSTM vs Historical Rate (ROC-AUC)...")
    roc_sig = paired_bootstrap_test_monthly(
        y_true_per_month, preds_per_month, hist_preds_per_month,
        evaluation.compute_roc_auc, n_bootstrap=1000, seed=42
    )
    significance_results['convlstm_vs_hist_roc'] = roc_sig
    print(f"    Observed diff: {roc_sig['observed_diff']:.4f}")
    print(f"    95% CI: [{roc_sig['ci_lower']:.4f}, {roc_sig['ci_upper']:.4f}]")
    print(f"    p-value: {roc_sig['p_value']:.4f}")

    # ========================================================================
    # 5. Calibration methods and metrics
    # ========================================================================
    print("\n5. Computing calibration metrics...")

    # Flatten validation data
    y_val_flat = y_val.reshape(-1)
    val_preds_flat = preds_val.reshape(-1)
    val_logits_flat = logits_val.reshape(-1)

    print(f"  Val data loaded: {y_val.shape}")

    # Fit calibration methods
    print("\n  Fitting calibration methods on validation set...")

    # Platt Scaling
    print("    Platt Scaling...")
    a_platt, b_platt = evaluation.platt_scaling(y_val_flat, val_logits_flat)

    # Apply to test set
    logits_to_probs = lambda x: 1 / (1 + np.exp(-x))
    probs_raw = logits_to_probs(logits_flat)
    probs_raw_clipped = np.clip(probs_raw, 1e-7, 1 - 1e-7)
    logit_probs = sp_logit(probs_raw_clipped)
    logits_transformed = a_platt * logit_probs + b_platt
    preds_platt = logits_to_probs(logits_transformed)

    # Isotonic Regression
    print("    Isotonic Regression...")
    iso_model = evaluation.isotonic_calibration(y_val_flat, val_preds_flat)
    preds_isotonic = iso_model.transform(preds_flat)

    # Temperature Scaling
    print("    Temperature Scaling...")
    T = evaluation.temperature_scaling(y_val_flat, val_logits_flat)
    preds_temp = logits_to_probs(logits_flat / T)

    # Compute calibration metrics for each
    print("\n  Computing calibration metrics on test set...")

    calibration_results = {}

    for name, preds_cal in [
        ('uncalibrated', preds_flat),
        ('platt', preds_platt),
        ('isotonic', preds_isotonic),
        ('temperature', preds_temp)
    ]:
        print(f"\n    {name.upper()}:")

        # Basic metrics
        roc = evaluation.compute_roc_auc(y_true_flat, preds_cal)
        pr = evaluation.compute_pr_auc(y_true_flat, preds_cal)
        brier = evaluation.compute_brier_score(y_true_flat, preds_cal)
        logloss = evaluation.compute_log_loss(y_true_flat, preds_cal)

        # Calibration metrics with robust MCE
        cal_metrics = compute_calibration_metrics_robust(
            y_true_flat, preds_cal, n_bins=10, min_bin_size=50
        )

        calibration_results[name] = {
            'roc_auc': float(roc),
            'pr_auc': float(pr),
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'ece': cal_metrics['ece'],
            'mce': cal_metrics['mce'],
            'mce_filtered_min50': cal_metrics['mce_filtered']
        }

        if name == 'platt':
            calibration_results[name]['platt_params'] = {
                'a': float(a_platt),
                'b': float(b_platt)
            }
        elif name == 'temperature':
            calibration_results[name]['temperature'] = float(T)

        print(f"      ROC-AUC: {roc:.4f}")
        print(f"      PR-AUC: {pr:.4f}")
        print(f"      Brier: {brier:.4f}")
        print(f"      Log Loss: {logloss:.4f}")
        print(f"      ECE: {cal_metrics['ece']:.4f}")
        print(f"      MCE (all bins): {cal_metrics['mce']:.4f}")
        print(f"      MCE (bins≥50): {cal_metrics['mce_filtered']:.4f}")

    # ========================================================================
    # 6. Save all results
    # ========================================================================
    print("\n6. Saving results...")

    all_results = {
        'convlstm': convlstm_results,
        'baselines': baseline_results,
        'significance_tests': significance_results,
        'calibration': calibration_results,
        'metadata': {
            'n_months': int(n_months),
            'n_samples': int(len(y_true_flat)),
            'positive_rate': float(y_true_flat.mean()),
            'bootstrap_iterations': 1000,
            'bootstrap_method': 'monthly_block_resampling'
        }
    }

    output_file = results_dir / "unified_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved to: {output_file}")

    # ========================================================================
    # 7. Print summary for Table I
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE I SUMMARY (Use these values)")
    print("="*80)

    print("\nConvLSTM:")
    print(f"  ROC-AUC: {convlstm_results['roc_auc']['point']:.4f} "
          f"[{convlstm_results['roc_auc']['ci_lower']:.4f}, "
          f"{convlstm_results['roc_auc']['ci_upper']:.4f}]")
    print(f"  PR-AUC: {convlstm_results['pr_auc']['point']:.4f} "
          f"[{convlstm_results['pr_auc']['ci_lower']:.4f}, "
          f"{convlstm_results['pr_auc']['ci_upper']:.4f}]")
    print(f"  Brier: {convlstm_results['brier_score']['point']:.4f} "
          f"[{convlstm_results['brier_score']['ci_lower']:.4f}, "
          f"{convlstm_results['brier_score']['ci_upper']:.4f}]")

    print("\nHistorical Rate:")
    hist = baseline_results['historical_rate']
    print(f"  ROC-AUC: {hist['roc_auc']['point']:.4f} "
          f"[{hist['roc_auc']['ci_lower']:.4f}, "
          f"{hist['roc_auc']['ci_upper']:.4f}]")
    print(f"  PR-AUC: {hist['pr_auc']['point']:.4f} "
          f"[{hist['pr_auc']['ci_lower']:.4f}, "
          f"{hist['pr_auc']['ci_upper']:.4f}]")
    print(f"  Brier: {hist['brier_score']['point']:.4f} "
          f"[{hist['brier_score']['ci_lower']:.4f}, "
          f"{hist['brier_score']['ci_upper']:.4f}]")

    print("\nSignificance Tests (paired bootstrap, monthly blocks):")
    print(f"  PR-AUC improvement: {pr_sig['observed_diff']:.4f} "
          f"(p={pr_sig['p_value']:.4f}, "
          f"95% CI: [{pr_sig['ci_lower']:.4f}, {pr_sig['ci_upper']:.4f}])")
    print(f"  ROC-AUC difference: {roc_sig['observed_diff']:.4f} "
          f"(p={roc_sig['p_value']:.4f})")

    if calibration_results:
        print("\n" + "="*80)
        print("TABLE II SUMMARY (Use these values)")
        print("="*80)

        for name in ['uncalibrated', 'platt', 'isotonic', 'temperature']:
            res = calibration_results[name]
            print(f"\n{name.upper()}:")
            print(f"  ROC-AUC: {res['roc_auc']:.4f}")
            print(f"  PR-AUC: {res['pr_auc']:.4f}")
            print(f"  Brier: {res['brier_score']:.4f}")
            print(f"  ECE: {res['ece']:.4f}")
            print(f"  MCE (all): {res['mce']:.4f}")
            print(f"  MCE (bins≥50): {res['mce_filtered_min50']:.4f}")
            print(f"  Log Loss: {res['log_loss']:.4f}")

    print("\n" + "="*80)
    print("DONE! Use the values above to update your paper.")
    print("="*80)


if __name__ == "__main__":
    main()
