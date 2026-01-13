"""
Comprehensive evaluation metrics and visualization tools for earthquake forecasting.

This module provides:
- Standard metrics (ROC-AUC, PR-AUC, Brier score, ECE, etc.)
- Bootstrap confidence intervals
- Probability calibration methods (Platt, Isotonic, Temperature)
- Visualization tools (reliability diagrams, Molchan diagrams)
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


# ============================================================================
# Basic Metrics
# ============================================================================

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC score using rank-based method."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)

    pos = int(y_true.sum())
    neg = int(y_true.size - pos)

    if pos == 0 or neg == 0:
        return np.nan

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)

    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)

    return float(auc)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall AUC (average precision)."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)

    pos = int(y_true.sum())
    if pos == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    # Use midpoint rule for integration
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))

    return ap


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    return float(np.mean((y_prob - y_true) ** 2))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log loss (cross-entropy)."""
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float64).reshape(-1), eps, 1 - eps)

    ll = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(ll)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for binning predictions

    Returns:
        ECE value
    """
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_prob)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_skill_score(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              y_ref_prob: np.ndarray) -> float:
    """
    Compute Brier Skill Score relative to a reference forecast.

    BSS = 1 - BS_model / BS_reference
    """
    bs_model = compute_brier_score(y_true, y_prob)
    bs_ref = compute_brier_score(y_true, y_ref_prob)

    if bs_ref == 0:
        return np.nan

    return 1.0 - (bs_model / bs_ref)


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_metric(y_true: np.ndarray,
                    y_score: np.ndarray,
                    metric_fn,
                    n_bootstrap: int = 1000,
                    confidence_level: float = 0.95,
                    seed: int = 42) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_score: Predicted scores
        metric_fn: Function that takes (y_true, y_score) and returns a scalar
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        try:
            score = metric_fn(y_true_boot, y_score_boot)
            if not np.isnan(score):
                bootstrap_scores.append(score)
        except:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return {
        'mean': float(np.mean(bootstrap_scores)),
        'std': float(np.std(bootstrap_scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_bootstrap': len(bootstrap_scores)
    }


def permutation_test(y_true: np.ndarray,
                    y_score1: np.ndarray,
                    y_score2: np.ndarray,
                    metric_fn,
                    n_permutations: int = 1000,
                    seed: int = 42) -> Dict[str, float]:
    """
    Test if metric_fn(y_true, y_score1) > metric_fn(y_true, y_score2) significantly.

    Returns:
        Dictionary with 'observed_diff', 'p_value', 'significant'
    """
    score1 = metric_fn(y_true, y_score1)
    score2 = metric_fn(y_true, y_score2)
    observed_diff = score1 - score2

    rng = np.random.RandomState(seed)
    n = len(y_true)

    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly swap scores
        swap = rng.binomial(1, 0.5, size=n).astype(bool)
        y_perm1 = np.where(swap, y_score2, y_score1)
        y_perm2 = np.where(swap, y_score1, y_score2)

        try:
            s1 = metric_fn(y_true, y_perm1)
            s2 = metric_fn(y_true, y_perm2)
            if not (np.isnan(s1) or np.isnan(s2)):
                perm_diffs.append(s1 - s2)
        except:
            continue

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

    return {
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01
    }


# ============================================================================
# Probability Calibration Methods
# ============================================================================

def platt_scaling(y_val: np.ndarray,
                 logits_val: np.ndarray) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters: p_cal = sigmoid(a * logit(p_raw) + b)

    Args:
        y_val: Validation set true labels
        logits_val: Validation set raw logits (before sigmoid)

    Returns:
        (a, b) parameters
    """
    from scipy.special import logit as sp_logit

    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits_val.reshape(-1)))

    # Clip to avoid inf in logit
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logit_probs = sp_logit(probs)

    y = y_val.reshape(-1).astype(np.float64)

    def loss(params):
        a, b = params
        z = a * logit_probs + b
        p_cal = 1 / (1 + np.exp(-z))
        p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))

    result = optimize.minimize(loss, x0=[1.0, 0.0], method='BFGS')
    a, b = result.x

    return float(a), float(b)


def apply_platt_scaling(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Platt scaling transformation."""
    from scipy.special import logit as sp_logit

    probs = 1 / (1 + np.exp(-logits.reshape(-1)))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logit_probs = sp_logit(probs)

    z = a * logit_probs + b
    p_cal = 1 / (1 + np.exp(-z))

    return p_cal.reshape(logits.shape)


def isotonic_calibration(y_val: np.ndarray,
                        probs_val: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression for calibration.

    Args:
        y_val: Validation set true labels
        probs_val: Validation set predicted probabilities

    Returns:
        Fitted IsotonicRegression model
    """
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(probs_val.reshape(-1), y_val.reshape(-1))
    return iso_reg


def temperature_scaling(y_val: np.ndarray,
                       logits_val: np.ndarray) -> float:
    """
    Fit temperature scaling parameter: p_cal = sigmoid(logits / T)

    Args:
        y_val: Validation set true labels
        logits_val: Validation set raw logits

    Returns:
        Temperature T
    """
    y = y_val.reshape(-1).astype(np.float64)
    logits = logits_val.reshape(-1)

    def loss(T):
        T = T[0]
        z = logits / T
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = optimize.minimize(loss, x0=[1.0], method='BFGS', bounds=[(0.1, 10.0)])
    return float(result.x[0])


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling."""
    z = logits / temperature
    p = 1 / (1 + np.exp(-z))
    return p


# ============================================================================
# Visualization
# ============================================================================

def plot_reliability_diagram(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             n_bins: int = 10,
                             output_path: Optional[str] = None,
                             title: str = "Reliability Diagram") -> plt.Figure:
    """
    Plot reliability diagram showing calibration quality.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    bin_means = []
    bin_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_freqs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_freqs.append(0)
            bin_counts.append(0)

    bin_means = np.array(bin_means)
    bin_freqs = np.array(bin_freqs)
    bin_counts = np.array(bin_counts)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Actual calibration
    ax.plot(bin_means, bin_freqs, 'o-', markersize=8, linewidth=2,
            label=f'Model (ECE={compute_ece(y_true, y_prob, n_bins):.3f})')

    # Bin counts as bar chart on secondary axis
    ax2 = ax.twinx()
    ax2.bar(bin_means, bin_counts, width=0.08, alpha=0.3, color='gray',
            label='Count per bin')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend(loc='upper left')

    ax.set_xlabel('Mean predicted probability', fontsize=14)
    ax.set_ylabel('Empirical frequency', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reliability diagram to {output_path}")

    return fig


def plot_molchan_diagram(y_true: np.ndarray,
                        y_score: np.ndarray,
                        output_path: Optional[str] = None,
                        title: str = "Molchan Diagram") -> plt.Figure:
    """
    Plot Molchan diagram (miss rate vs. alarm rate).

    In earthquake forecasting, Molchan diagrams show the trade-off between
    miss rate (fraction of events not predicted) and alarm rate (fraction of
    space-time occupied by alarms).

    Args:
        y_true: True binary labels (1 if earthquake occurred)
        y_score: Predicted scores (higher = more likely)
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)

    # Sort by score (descending)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n_total = len(y_true)
    n_pos = int(y_true.sum())

    if n_pos == 0:
        print("Warning: No positive samples, cannot plot Molchan diagram")
        return None

    # Compute cumulative sums
    n_alarmed = np.arange(1, n_total + 1)
    n_captured = np.cumsum(y_sorted)

    # Alarm rate: fraction of space-time alarmed
    alarm_rate = n_alarmed / n_total

    # Miss rate: fraction of events NOT captured
    miss_rate = 1 - (n_captured / n_pos)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Random baseline
    ax.plot([0, 1], [1, 0], 'k--', label='Random', linewidth=2)

    # Model performance
    # Add (0, 1) at start to complete curve
    alarm_rate_full = np.concatenate([[0], alarm_rate])
    miss_rate_full = np.concatenate([[1], miss_rate])
    ax.plot(alarm_rate_full, miss_rate_full, 'b-', linewidth=2, label='Model')

    # Optimal point annotation (e.g., 10% alarm rate)
    idx_10pct = np.argmin(np.abs(alarm_rate - 0.10))
    ax.plot(alarm_rate[idx_10pct], miss_rate[idx_10pct], 'ro', markersize=10,
            label=f'10% alarm: {100*(1-miss_rate[idx_10pct]):.1f}% capture')

    ax.set_xlabel('Alarm rate (fraction of space-time)', fontsize=14)
    ax.set_ylabel('Miss rate (fraction of events missed)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Molchan diagram to {output_path}")

    return fig


# ============================================================================
# Top-K Alarm Evaluation
# ============================================================================

def compute_topk_metrics(y_true: np.ndarray,
                        y_score: np.ndarray,
                        alarm_fractions: np.ndarray = np.linspace(0.01, 0.5, 50)) -> Dict[str, np.ndarray]:
    """
    Compute top-K alarm metrics for various alarm area fractions.

    Args:
        y_true: True labels [N, H, W] or [N*H*W]
        y_score: Predicted scores [N, H, W] or [N*H*W]
        alarm_fractions: Array of alarm area fractions to evaluate

    Returns:
        Dictionary with 'alarm_fractions', 'hit_rates', 'precisions', 'f1_scores', 'miss_rates'
    """
    y_true_flat = y_true.reshape(-1)
    y_score_flat = y_score.reshape(-1)

    n_total = len(y_true_flat)
    n_pos = int(y_true_flat.sum())

    hit_rates = []
    precisions = []
    f1_scores = []
    miss_rates = []

    for alpha in alarm_fractions:
        k = max(1, int(alpha * n_total))

        # Top-K predictions
        top_k_indices = np.argpartition(-y_score_flat, k-1)[:k]

        # Compute metrics
        n_captured = int(y_true_flat[top_k_indices].sum())
        hit_rate = n_captured / n_pos if n_pos > 0 else 0
        precision = n_captured / k
        f1 = 2 * precision * hit_rate / (precision + hit_rate) if (precision + hit_rate) > 0 else 0
        miss_rate = 1 - hit_rate

        hit_rates.append(hit_rate)
        precisions.append(precision)
        f1_scores.append(f1)
        miss_rates.append(miss_rate)

    return {
        'alarm_fractions': alarm_fractions,
        'hit_rates': np.array(hit_rates),
        'precisions': np.array(precisions),
        'f1_scores': np.array(f1_scores),
        'miss_rates': np.array(miss_rates)
    }


def plot_topk_curves(topk_metrics: Dict[str, np.ndarray],
                    output_path: Optional[str] = None,
                    title: str = "Top-K Alarm Performance") -> plt.Figure:
    """Plot top-K alarm performance curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    alpha = topk_metrics['alarm_fractions'] * 100  # Convert to percentage

    # Hit rate
    axes[0, 0].plot(alpha, topk_metrics['hit_rates'] * 100, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Alarm area (%)', fontsize=12)
    axes[0, 0].set_ylabel('Hit rate (%)', fontsize=12)
    axes[0, 0].set_title('Hit Rate vs Alarm Area', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Precision
    axes[0, 1].plot(alpha, topk_metrics['precisions'] * 100, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Alarm area (%)', fontsize=12)
    axes[0, 1].set_ylabel('Precision (%)', fontsize=12)
    axes[0, 1].set_title('Precision vs Alarm Area', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # F1 score
    axes[1, 0].plot(alpha, topk_metrics['f1_scores'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Alarm area (%)', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('F1 Score vs Alarm Area', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Optimal point
    opt_idx = np.argmax(topk_metrics['f1_scores'])
    axes[1, 0].plot(alpha[opt_idx], topk_metrics['f1_scores'][opt_idx], 'ro', markersize=10,
                   label=f'Optimal: {alpha[opt_idx]:.1f}% alarm')
    axes[1, 0].legend(fontsize=10)

    # Miss rate
    axes[1, 1].plot(alpha, topk_metrics['miss_rates'] * 100, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Alarm area (%)', fontsize=12)
    axes[1, 1].set_ylabel('Miss rate (%)', fontsize=12)
    axes[1, 1].set_title('Miss Rate vs Alarm Area', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved top-K curves to {output_path}")

    return fig
