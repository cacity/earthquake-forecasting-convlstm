"""
Comprehensive evaluation script for earthquake forecasting models.

This script runs all evaluations needed for the IEEE Access paper:
1. Bootstrap confidence intervals for metrics
2. Compare multiple loss functions
3. Compare multiple calibration methods
4. Generate reliability diagrams
5. Generate Molchan diagrams
6. Evaluate baseline models (kernel smoothing)
7. Analyze catalog completeness (Mc, b-value)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eqgrid import evaluation, baselines, completeness


def load_predictions_and_labels(pred_path: Path,
                               label_path: Path) -> tuple:
    """Load predictions and labels from numpy files."""
    print(f"Loading predictions from {pred_path}")
    preds = np.load(pred_path)  # [N, 1, H, W]

    print(f"Loading labels from {label_path}")
    labels = np.load(label_path)  # [N, 1, H, W]

    print(f"  Predictions shape: {preds.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Positive rate: {labels.mean():.4f}")

    return preds, labels


def evaluate_with_bootstrap(y_true: np.ndarray,
                           y_score: np.ndarray,
                           output_dir: Path,
                           n_bootstrap: int = 1000):
    """
    Compute metrics with bootstrap confidence intervals.

    Saves results to JSON file.
    """
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ROC-AUC
    print("\nComputing bootstrap CI for ROC-AUC...")
    roc_boot = evaluation.bootstrap_metric(
        y_true, y_score,
        evaluation.compute_roc_auc,
        n_bootstrap=n_bootstrap
    )
    results['roc_auc'] = roc_boot
    print(f"  ROC-AUC: {roc_boot['mean']:.4f} "
          f"(95% CI: [{roc_boot['ci_lower']:.4f}, {roc_boot['ci_upper']:.4f}])")

    # PR-AUC
    print("\nComputing bootstrap CI for PR-AUC...")
    pr_boot = evaluation.bootstrap_metric(
        y_true, y_score,
        evaluation.compute_pr_auc,
        n_bootstrap=n_bootstrap
    )
    results['pr_auc'] = pr_boot
    print(f"  PR-AUC: {pr_boot['mean']:.4f} "
          f"(95% CI: [{pr_boot['ci_lower']:.4f}, {pr_boot['ci_upper']:.4f}])")

    # Brier Score
    print("\nComputing bootstrap CI for Brier Score...")
    brier_boot = evaluation.bootstrap_metric(
        y_true, y_score,
        evaluation.compute_brier_score,
        n_bootstrap=n_bootstrap
    )
    results['brier_score'] = brier_boot
    print(f"  Brier Score: {brier_boot['mean']:.4f} "
          f"(95% CI: [{brier_boot['ci_lower']:.4f}, {brier_boot['ci_upper']:.4f}])")

    # Save results
    output_file = output_dir / "bootstrap_ci.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved bootstrap results to {output_file}")

    return results


def compare_with_baseline(y_true: np.ndarray,
                         y_score_model: np.ndarray,
                         y_score_baseline: np.ndarray,
                         baseline_name: str,
                         output_dir: Path,
                         n_permutations: int = 1000):
    """
    Statistical comparison between model and baseline.

    Uses permutation test to assess significance of improvement.
    """
    print(f"\n" + "="*80)
    print(f"STATISTICAL COMPARISON: MODEL vs {baseline_name.upper()}")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ROC-AUC comparison
    print("\nTesting ROC-AUC difference...")
    roc_test = evaluation.permutation_test(
        y_true, y_score_model, y_score_baseline,
        evaluation.compute_roc_auc,
        n_permutations=n_permutations
    )
    results['roc_auc_test'] = roc_test
    print(f"  Observed difference: {roc_test['observed_diff']:.4f}")
    print(f"  p-value: {roc_test['p_value']:.4f}")
    print(f"  Significant (p<0.05): {roc_test['significant_05']}")

    # PR-AUC comparison
    print("\nTesting PR-AUC difference...")
    pr_test = evaluation.permutation_test(
        y_true, y_score_model, y_score_baseline,
        evaluation.compute_pr_auc,
        n_permutations=n_permutations
    )
    results['pr_auc_test'] = pr_test
    print(f"  Observed difference: {pr_test['observed_diff']:.4f}")
    print(f"  p-value: {pr_test['p_value']:.4f}")
    print(f"  Significant (p<0.05): {pr_test['significant_05']}")

    # Save results
    output_file = output_dir / f"comparison_vs_{baseline_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison results to {output_file}")

    return results


def evaluate_calibration_methods(y_val: np.ndarray,
                                logits_val: np.ndarray,
                                y_test: np.ndarray,
                                logits_test: np.ndarray,
                                output_dir: Path):
    """
    Compare multiple calibration methods.

    Evaluates: Platt scaling, Isotonic regression, Temperature scaling
    """
    print("\n" + "="*80)
    print("CALIBRATION METHODS COMPARISON")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert logits to probabilities for uncalibrated baseline
    probs_test_uncal = 1 / (1 + np.exp(-logits_test.reshape(-1)))
    y_test_flat = y_test.reshape(-1)

    results = {}

    # Uncalibrated
    print("\nUncalibrated:")
    brier_uncal = evaluation.compute_brier_score(y_test_flat, probs_test_uncal)
    ece_uncal = evaluation.compute_ece(y_test_flat, probs_test_uncal)
    logloss_uncal = evaluation.compute_log_loss(y_test_flat, probs_test_uncal)
    print(f"  Brier: {brier_uncal:.4f}, ECE: {ece_uncal:.4f}, LogLoss: {logloss_uncal:.4f}")

    results['uncalibrated'] = {
        'brier': float(brier_uncal),
        'ece': float(ece_uncal),
        'log_loss': float(logloss_uncal)
    }

    # Platt scaling
    print("\nFitting Platt scaling...")
    a, b = evaluation.platt_scaling(y_val.reshape(-1), logits_val.reshape(-1))
    probs_test_platt = evaluation.apply_platt_scaling(logits_test, a, b).reshape(-1)

    brier_platt = evaluation.compute_brier_score(y_test_flat, probs_test_platt)
    ece_platt = evaluation.compute_ece(y_test_flat, probs_test_platt)
    logloss_platt = evaluation.compute_log_loss(y_test_flat, probs_test_platt)

    print(f"  Parameters: a={a:.4f}, b={b:.4f}")
    print(f"  Brier: {brier_platt:.4f}, ECE: {ece_platt:.4f}, LogLoss: {logloss_platt:.4f}")

    results['platt_scaling'] = {
        'params': {'a': float(a), 'b': float(b)},
        'brier': float(brier_platt),
        'ece': float(ece_platt),
        'log_loss': float(logloss_platt)
    }

    # Isotonic regression
    print("\nFitting Isotonic regression...")
    probs_val = 1 / (1 + np.exp(-logits_val.reshape(-1)))
    iso_model = evaluation.isotonic_calibration(y_val.reshape(-1), probs_val)

    probs_test_iso = iso_model.predict(probs_test_uncal)

    brier_iso = evaluation.compute_brier_score(y_test_flat, probs_test_iso)
    ece_iso = evaluation.compute_ece(y_test_flat, probs_test_iso)
    logloss_iso = evaluation.compute_log_loss(y_test_flat, probs_test_iso)

    print(f"  Brier: {brier_iso:.4f}, ECE: {ece_iso:.4f}, LogLoss: {logloss_iso:.4f}")

    results['isotonic'] = {
        'brier': float(brier_iso),
        'ece': float(ece_iso),
        'log_loss': float(logloss_iso)
    }

    # Temperature scaling
    print("\nFitting Temperature scaling...")
    temp = evaluation.temperature_scaling(y_val.reshape(-1), logits_val.reshape(-1))
    probs_test_temp = evaluation.apply_temperature_scaling(logits_test, temp).reshape(-1)

    brier_temp = evaluation.compute_brier_score(y_test_flat, probs_test_temp)
    ece_temp = evaluation.compute_ece(y_test_flat, probs_test_temp)
    logloss_temp = evaluation.compute_log_loss(y_test_flat, probs_test_temp)

    print(f"  Temperature: {temp:.4f}")
    print(f"  Brier: {brier_temp:.4f}, ECE: {ece_temp:.4f}, LogLoss: {logloss_temp:.4f}")

    results['temperature_scaling'] = {
        'temperature': float(temp),
        'brier': float(brier_temp),
        'ece': float(ece_temp),
        'log_loss': float(logloss_temp)
    }

    # Save results
    output_file = output_dir / "calibration_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {output_file}")

    # Generate reliability diagrams
    print("\nGenerating reliability diagrams...")

    fig = evaluation.plot_reliability_diagram(
        y_test_flat, probs_test_uncal,
        output_path=output_dir / "reliability_diagram_uncalibrated.png",
        title="Reliability Diagram (Uncalibrated)"
    )

    fig = evaluation.plot_reliability_diagram(
        y_test_flat, probs_test_platt,
        output_path=output_dir / "reliability_diagram_platt.png",
        title="Reliability Diagram (Platt Scaling)"
    )

    fig = evaluation.plot_reliability_diagram(
        y_test_flat, probs_test_iso,
        output_path=output_dir / "reliability_diagram_isotonic.png",
        title="Reliability Diagram (Isotonic)"
    )

    fig = evaluation.plot_reliability_diagram(
        y_test_flat, probs_test_temp,
        output_path=output_dir / "reliability_diagram_temperature.png",
        title="Reliability Diagram (Temperature Scaling)"
    )

    return results


def generate_molchan_diagram(y_true: np.ndarray,
                            y_score: np.ndarray,
                            output_dir: Path):
    """Generate Molchan diagram."""
    print("\n" + "="*80)
    print("GENERATING MOLCHAN DIAGRAM")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = evaluation.plot_molchan_diagram(
        y_true.reshape(-1),
        y_score.reshape(-1),
        output_path=output_dir / "molchan_diagram.png",
        title="Molchan Diagram - ConvLSTM Model"
    )

    print("Done!")


def evaluate_baselines(y_train: np.ndarray,
                      x_test: np.ndarray,
                      y_test: np.ndarray,
                      output_dir: Path):
    """
    Evaluate baseline models including kernel smoothing.
    """
    print("\n" + "="*80)
    print("BASELINE MODELS EVALUATION")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = baselines.compare_baselines(
        y_train, x_test, y_test,
        sigma_values=[0.5, 1.0, 1.5, 2.0]
    )

    # Save results
    output_dict = {}
    for name, result in baseline_results.items():
        output_dict[name] = {
            'roc_auc': float(result['roc_auc']),
            'pr_auc': float(result['pr_auc']),
            'brier': float(result['brier'])
        }
        if 'sigma' in result:
            output_dict[name]['sigma'] = float(result['sigma'])

    output_file = output_dir / "baseline_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"\nSaved baseline comparison to {output_file}")

    # Save predictions for best kernel density model
    best_kde = max([k for k in baseline_results.keys() if k.startswith('kernel_density')],
                  key=lambda k: baseline_results[k]['roc_auc'])

    print(f"\nBest kernel density model: {best_kde}")
    np.save(output_dir / f"{best_kde}_preds.npy", baseline_results[best_kde]['preds'])

    return baseline_results


def analyze_catalog_completeness(catalog_path: Path,
                                output_dir: Path):
    """
    Analyze catalog completeness (Mc, b-value, temporal evolution).
    """
    print("\n" + "="*80)
    print("CATALOG COMPLETENESS ANALYSIS")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog (assuming GeoJSON or CSV format)
    print(f"\nLoading catalog from {catalog_path}")

    # Try different formats
    if catalog_path.suffix == '.json':
        import json
        with open(catalog_path) as f:
            data = json.load(f)
        # Extract magnitudes and times from GeoJSON
        magnitudes = np.array([f['properties']['mag'] for f in data['features']])
        times = np.array([f['properties']['time'] for f in data['features']], dtype='datetime64[ms]')

    elif catalog_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(catalog_path)
        magnitudes = df['magnitude'].values
        times = pd.to_datetime(df['time']).values

    else:
        print(f"Error: Unsupported catalog format {catalog_path.suffix}")
        return None

    print(f"  Total events: {len(magnitudes)}")
    print(f"  Magnitude range: {magnitudes.min():.2f} - {magnitudes.max():.2f}")

    # Estimate Mc
    print("\nEstimating Mc using maximum curvature method...")
    mc_maxc, max_count = completeness.estimate_mc_maxc(magnitudes)
    print(f"  Mc (max curvature): {mc_maxc:.2f}")

    print("\nEstimating Mc using goodness-of-fit test...")
    mc_gft = completeness.estimate_mc_gft(magnitudes)
    print(f"  Mc (GFT): {mc_gft:.2f}")

    # Use more conservative estimate
    mc_final = max(mc_maxc, mc_gft)
    print(f"\n  Using Mc = {mc_final:.2f} for analysis")

    # Fit Gutenberg-Richter
    print("\nFitting Gutenberg-Richter relation...")
    gr_mle = completeness.fit_gutenberg_richter(magnitudes, mag_min=mc_final, method='mle')
    print(f"  MLE: b = {gr_mle['b_value']:.3f} ± {gr_mle['b_std']:.3f}, "
          f"a = {gr_mle['a_value']:.3f}, N = {gr_mle['n_events']}")

    gr_lsq = completeness.fit_gutenberg_richter(magnitudes, mag_min=mc_final, method='lsq')
    print(f"  LSQ: b = {gr_lsq['b_value']:.3f} ± {gr_lsq['b_std']:.3f}, "
          f"a = {gr_lsq['a_value']:.3f}, N = {gr_lsq['n_events']}")

    # Plot magnitude distribution
    print("\nGenerating magnitude distribution plots...")
    fig = completeness.plot_magnitude_distribution(
        magnitudes,
        mc=mc_final,
        gr_fit=gr_mle,
        output_path=output_dir / "magnitude_distribution.png",
        title="Magnitude Distribution with G-R Fit"
    )

    # Temporal analysis
    print("\nPerforming temporal completeness analysis...")
    temporal_result = completeness.temporal_completeness_analysis(
        magnitudes, times,
        time_bins=26,  # One bin per year for 26-year catalog
        mag_threshold=4.0
    )

    fig = completeness.plot_temporal_completeness(
        temporal_result,
        output_path=output_dir / "temporal_completeness.png",
        title="Temporal Completeness (2000-2025)"
    )

    # Save results
    results = {
        'mc_maxc': float(mc_maxc),
        'mc_gft': float(mc_gft),
        'mc_final': float(mc_final),
        'gr_mle': gr_mle,
        'gr_lsq': gr_lsq,
        'total_events': int(len(magnitudes)),
        'events_above_mc': int(np.sum(magnitudes >= mc_final)),
        'magnitude_range': [float(magnitudes.min()), float(magnitudes.max())]
    }

    output_file = output_dir / "completeness_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved completeness analysis to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for earthquake forecasting models"
    )

    # Input files
    parser.add_argument("--model-preds", type=Path, required=True,
                       help="Model predictions (.npy file, [N, 1, H, W])")
    parser.add_argument("--test-labels", type=Path, required=True,
                       help="Test labels (.npy file, [N, 1, H, W])")
    parser.add_argument("--val-labels", type=Path,
                       help="Validation labels for calibration")
    parser.add_argument("--model-logits", type=Path,
                       help="Model logits for calibration (.npy file)")
    parser.add_argument("--val-logits", type=Path,
                       help="Validation logits for calibration")

    # Baseline comparisons
    parser.add_argument("--historical-rate-preds", type=Path,
                       help="Historical rate baseline predictions")
    parser.add_argument("--train-labels", type=Path,
                       help="Training labels for baseline fitting")
    parser.add_argument("--test-features", type=Path,
                       help="Test features [N, L, C, H, W] for baselines")

    # Catalog analysis
    parser.add_argument("--catalog", type=Path,
                       help="Earthquake catalog file (.json or .csv)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"),
                       help="Output directory for all results")

    # Analysis flags
    parser.add_argument("--skip-bootstrap", action="store_true",
                       help="Skip bootstrap CI computation (slow)")
    parser.add_argument("--skip-calibration", action="store_true",
                       help="Skip calibration comparison")
    parser.add_argument("--skip-baselines", action="store_true",
                       help="Skip baseline evaluation")
    parser.add_argument("--skip-completeness", action="store_true",
                       help="Skip catalog completeness analysis")

    # Parameters
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Number of bootstrap samples")
    parser.add_argument("--n-permutations", type=int, default=1000,
                       help="Number of permutations for significance testing")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model predictions and test labels
    model_preds, test_labels = load_predictions_and_labels(
        args.model_preds, args.test_labels
    )

    y_test_flat = test_labels.reshape(-1)
    model_preds_flat = model_preds.reshape(-1)

    # 1. Bootstrap confidence intervals
    if not args.skip_bootstrap:
        evaluate_with_bootstrap(
            y_test_flat,
            model_preds_flat,
            args.output_dir / "bootstrap",
            n_bootstrap=args.n_bootstrap
        )

    # 2. Generate Molchan diagram
    generate_molchan_diagram(
        test_labels,
        model_preds,
        args.output_dir / "molchan"
    )

    # 3. Calibration methods comparison
    if not args.skip_calibration and args.val_labels and args.model_logits and args.val_logits:
        val_labels = np.load(args.val_labels)
        test_logits = np.load(args.model_logits)
        val_logits = np.load(args.val_logits)

        evaluate_calibration_methods(
            val_labels, val_logits,
            test_labels, test_logits,
            args.output_dir / "calibration"
        )

    # 4. Baseline comparison
    if not args.skip_baselines and args.train_labels and args.test_features:
        train_labels = np.load(args.train_labels)
        test_features = np.load(args.test_features)

        evaluate_baselines(
            train_labels,
            test_features,
            test_labels,
            args.output_dir / "baselines"
        )

    # 5. Statistical comparison with baseline
    if args.historical_rate_preds:
        hist_preds = np.load(args.historical_rate_preds)

        compare_with_baseline(
            y_test_flat,
            model_preds_flat,
            hist_preds.reshape(-1),
            "historical_rate",
            args.output_dir / "statistical_tests",
            n_permutations=args.n_permutations
        )

    # 6. Catalog completeness analysis
    if not args.skip_completeness and args.catalog:
        analyze_catalog_completeness(
            args.catalog,
            args.output_dir / "completeness"
        )

    # Generate Top-K curves
    print("\n" + "="*80)
    print("GENERATING TOP-K ALARM CURVES")
    print("="*80)

    topk_metrics = evaluation.compute_topk_metrics(
        test_labels,
        model_preds,
        alarm_fractions=np.linspace(0.01, 0.5, 50)
    )

    fig = evaluation.plot_topk_curves(
        topk_metrics,
        output_path=args.output_dir / "topk_curves.png",
        title="Top-K Alarm Performance"
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
