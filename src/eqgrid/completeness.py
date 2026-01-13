"""
Catalog completeness analysis tools.

This module provides tools for analyzing earthquake catalog completeness:
- Magnitude of completeness (Mc) estimation
- Gutenberg-Richter law fitting
- b-value computation
- Temporal completeness analysis
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import optimize, stats


def compute_magnitude_bins(magnitudes: np.ndarray,
                           bin_width: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude frequency distribution.

    Args:
        magnitudes: Array of earthquake magnitudes
        bin_width: Bin width for magnitude bins

    Returns:
        (bin_centers, counts) tuple
    """
    mag_min = np.floor(magnitudes.min() / bin_width) * bin_width
    mag_max = np.ceil(magnitudes.max() / bin_width) * bin_width

    bins = np.arange(mag_min, mag_max + bin_width, bin_width)
    counts, _ = np.histogram(magnitudes, bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, counts


def fit_gutenberg_richter(magnitudes: np.ndarray,
                         mag_min: float,
                         mag_max: Optional[float] = None,
                         method: str = 'mle') -> Dict[str, float]:
    """
    Fit Gutenberg-Richter relationship and compute b-value.

    The Gutenberg-Richter law: log10(N) = a - b*M
    where N is the number of earthquakes with magnitude ≥ M

    Args:
        magnitudes: Array of earthquake magnitudes
        mag_min: Minimum magnitude for fitting (Mc estimate)
        mag_max: Maximum magnitude for fitting (optional)
        method: Fitting method, 'mle' (maximum likelihood) or 'lsq' (least squares)

    Returns:
        Dictionary with 'b_value', 'a_value', 'b_std', 'mc_used', 'n_events'
    """
    # Filter magnitudes
    mask = magnitudes >= mag_min
    if mag_max is not None:
        mask &= magnitudes <= mag_max

    mags_filtered = magnitudes[mask]
    n_events = len(mags_filtered)

    if n_events < 10:
        return {
            'b_value': np.nan,
            'a_value': np.nan,
            'b_std': np.nan,
            'mc_used': mag_min,
            'n_events': n_events,
            'method': method
        }

    if method == 'mle':
        # Maximum likelihood estimator (Aki 1965)
        mean_mag = mags_filtered.mean()
        b_value = 1.0 / (np.log(10) * (mean_mag - (mag_min - 0.05)))  # 0.05 is bin correction

        # Standard deviation (Shi & Bolt 1982)
        b_std = b_value / np.sqrt(n_events)

    elif method == 'lsq':
        # Least squares fit to log10(N) = a - b*M
        bin_centers, counts = compute_magnitude_bins(mags_filtered, bin_width=0.1)

        # Cumulative distribution
        cumulative = np.array([np.sum(mags_filtered >= m) for m in bin_centers])

        # Only use bins with at least 5 events for stable fit
        valid = cumulative >= 5

        if valid.sum() < 3:
            return {
                'b_value': np.nan,
                'a_value': np.nan,
                'b_std': np.nan,
                'mc_used': mag_min,
                'n_events': n_events,
                'method': method
            }

        log_N = np.log10(cumulative[valid])
        M = bin_centers[valid]

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(M, log_N)

        b_value = -slope
        a_value = intercept
        b_std = std_err

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute a-value (only for MLE, as LSQ already computes it)
    if method == 'mle':
        a_value = np.log10(n_events) + b_value * mag_min

    return {
        'b_value': float(b_value),
        'a_value': float(a_value),
        'b_std': float(b_std),
        'mc_used': float(mag_min),
        'n_events': int(n_events),
        'method': method
    }


def estimate_mc_maxc(magnitudes: np.ndarray,
                    bin_width: float = 0.1) -> Tuple[float, float]:
    """
    Estimate Mc using maximum curvature method (Wiemer & Wyss 2000).

    Args:
        magnitudes: Array of earthquake magnitudes
        bin_width: Bin width for magnitude binning

    Returns:
        (Mc, max_count) tuple
    """
    bin_centers, counts = compute_magnitude_bins(magnitudes, bin_width=bin_width)

    # Find magnitude with maximum count
    max_idx = np.argmax(counts)
    mc = bin_centers[max_idx]
    max_count = counts[max_idx]

    return float(mc), float(max_count)


def estimate_mc_gft(magnitudes: np.ndarray,
                   bin_width: float = 0.1,
                   tolerance: float = 0.1) -> float:
    """
    Estimate Mc using Goodness-of-Fit Test (Wiemer & Wyss 2000).

    Tests multiple Mc values and selects the one that best fits the G-R law.

    Args:
        magnitudes: Array of earthquake magnitudes
        bin_width: Bin width for magnitude binning
        tolerance: Tolerance for goodness-of-fit (typically 0.1)

    Returns:
        Estimated Mc
    """
    mc_candidates = np.arange(magnitudes.min(), magnitudes.max() - 0.5, bin_width)

    best_mc = mc_candidates[0]
    best_residual = np.inf

    for mc_test in mc_candidates:
        # Fit G-R for this Mc
        result = fit_gutenberg_richter(magnitudes, mag_min=mc_test, method='lsq')

        if np.isnan(result['b_value']):
            continue

        # Compute residuals
        mags_filtered = magnitudes[magnitudes >= mc_test]
        bin_centers, counts = compute_magnitude_bins(mags_filtered, bin_width=bin_width)

        cumulative = np.array([np.sum(mags_filtered >= m) for m in bin_centers])
        predicted = 10 ** (result['a_value'] - result['b_value'] * bin_centers)

        # Only use bins with sufficient events
        valid = cumulative >= 5
        if valid.sum() < 3:
            continue

        residual = np.abs(np.log10(cumulative[valid]) - np.log10(predicted[valid])).mean()

        if residual < best_residual:
            best_residual = residual
            best_mc = mc_test

        # Early termination if residual is small enough
        if residual < tolerance:
            break

    return float(best_mc)


def temporal_completeness_analysis(magnitudes: np.ndarray,
                                  times: np.ndarray,
                                  time_bins: int = 50,
                                  mag_threshold: float = 4.0) -> Dict:
    """
    Analyze temporal evolution of catalog completeness.

    Args:
        magnitudes: Array of earthquake magnitudes
        times: Array of earthquake times (as timestamps or datetime64)
        time_bins: Number of time bins for analysis
        mag_threshold: Magnitude threshold to analyze

    Returns:
        Dictionary with temporal statistics
    """
    # Convert times to numeric (years since start)
    if times.dtype.kind == 'M':  # datetime64
        times_numeric = (times - times.min()) / np.timedelta64(1, 'Y')
    else:
        times_numeric = times

    # Create time bins
    time_edges = np.linspace(times_numeric.min(), times_numeric.max(), time_bins + 1)
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2

    mc_evolution = []
    b_value_evolution = []
    event_counts = []

    for i in range(time_bins):
        mask = (times_numeric >= time_edges[i]) & (times_numeric < time_edges[i + 1])
        mags_bin = magnitudes[mask]

        event_counts.append(len(mags_bin))

        if len(mags_bin) >= 50:  # Minimum events for reliable estimate
            mc = estimate_mc_maxc(mags_bin)[0]
            gr_result = fit_gutenberg_richter(mags_bin, mag_min=mc, method='mle')

            mc_evolution.append(mc)
            b_value_evolution.append(gr_result['b_value'])
        else:
            mc_evolution.append(np.nan)
            b_value_evolution.append(np.nan)

    # Count events above threshold over time
    above_threshold = []
    for i in range(time_bins):
        mask = (times_numeric >= time_edges[i]) & (times_numeric < time_edges[i + 1])
        above_threshold.append(np.sum(magnitudes[mask] >= mag_threshold))

    return {
        'time_centers': time_centers,
        'time_edges': time_edges,
        'mc_evolution': np.array(mc_evolution),
        'b_value_evolution': np.array(b_value_evolution),
        'event_counts': np.array(event_counts),
        'above_threshold_counts': np.array(above_threshold),
        'threshold': mag_threshold
    }


def plot_magnitude_distribution(magnitudes: np.ndarray,
                                mc: Optional[float] = None,
                                gr_fit: Optional[Dict] = None,
                                output_path: Optional[str] = None,
                                title: str = "Magnitude Distribution") -> plt.Figure:
    """
    Plot magnitude frequency distribution with optional G-R fit.

    Args:
        magnitudes: Array of earthquake magnitudes
        mc: Magnitude of completeness (draws vertical line if provided)
        gr_fit: G-R fit result from fit_gutenberg_richter (draws fit line if provided)
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    bin_centers, counts = compute_magnitude_bins(magnitudes, bin_width=0.1)

    # Cumulative distribution
    cumulative = np.array([np.sum(magnitudes >= m) for m in bin_centers])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Incremental distribution
    ax1.bar(bin_centers, counts, width=0.09, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Number of events', fontsize=12)
    ax1.set_title('Incremental Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)

    if mc is not None:
        ax1.axvline(mc, color='red', linestyle='--', linewidth=2, label=f'Mc = {mc:.1f}')
        ax1.legend(fontsize=10)

    # Cumulative distribution (log scale)
    ax2.semilogy(bin_centers, cumulative, 'bo-', markersize=6, linewidth=2, label='Observed')

    if gr_fit is not None and not np.isnan(gr_fit['b_value']):
        # Plot G-R fit
        mag_range = bin_centers[bin_centers >= gr_fit['mc_used']]
        predicted = 10 ** (gr_fit['a_value'] - gr_fit['b_value'] * mag_range)

        ax2.semilogy(mag_range, predicted, 'r--', linewidth=2,
                    label=f"G-R fit: b={gr_fit['b_value']:.2f}±{gr_fit['b_std']:.2f}")

    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Cumulative number N(≥M)', fontsize=12)
    ax2.set_title('Gutenberg-Richter Relation', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    if mc is not None:
        ax2.axvline(mc, color='red', linestyle='--', linewidth=2)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved magnitude distribution plot to {output_path}")

    return fig


def plot_temporal_completeness(temporal_result: Dict,
                               output_path: Optional[str] = None,
                               title: str = "Temporal Completeness Analysis") -> plt.Figure:
    """
    Plot temporal evolution of catalog completeness.

    Args:
        temporal_result: Result from temporal_completeness_analysis()
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = temporal_result['time_centers']

    # Mc evolution
    axes[0].plot(time, temporal_result['mc_evolution'], 'o-', linewidth=2, markersize=5)
    axes[0].set_ylabel('Mc', fontsize=12)
    axes[0].set_title('Magnitude of Completeness', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(temporal_result['threshold'], color='red', linestyle='--',
                   label=f"M≥{temporal_result['threshold']}")
    axes[0].legend(fontsize=10)

    # b-value evolution
    axes[1].plot(time, temporal_result['b_value_evolution'], 'o-',
                linewidth=2, markersize=5, color='green')
    axes[1].set_ylabel('b-value', fontsize=12)
    axes[1].set_title('b-value Evolution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(1.0, color='gray', linestyle=':', label='Typical b=1.0')
    axes[1].legend(fontsize=10)

    # Event counts
    axes[2].bar(time, temporal_result['event_counts'], width=time[1]-time[0],
               alpha=0.5, label='All events')
    axes[2].plot(time, temporal_result['above_threshold_counts'], 'o-',
                linewidth=2, markersize=5, color='red',
                label=f"M≥{temporal_result['threshold']}")
    axes[2].set_xlabel('Time (years since start)', fontsize=12)
    axes[2].set_ylabel('Event count', fontsize=12)
    axes[2].set_title('Seismicity Rate', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal completeness plot to {output_path}")

    return fig
