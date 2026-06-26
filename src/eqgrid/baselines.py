"""
Baseline models for earthquake forecasting.

This module provides simple baseline models for comparison:
- Historical rate (time-invariant climatology)
- Persistence (last month's activity)
- Kernel density smoothing
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter


class HistoricalRateBaseline:
    """
    Historical rate baseline: predict constant probability equal to training set frequency.

    This is the climatological baseline used in the paper.
    """

    def __init__(self):
        self.rates = None  # [H, W] array of historical rates per cell

    def fit(self, y_train: np.ndarray) -> None:
        """
        Fit the model on training data.

        Args:
            y_train: Training labels [N, 1, H, W] or [N, H, W]
        """
        if y_train.ndim == 4 and y_train.shape[1] == 1:
            y_train = y_train[:, 0, :, :]  # Remove channel dimension

        # Compute mean rate per cell
        self.rates = y_train.mean(axis=0)  # [H, W]

    def predict(self, n_samples: int) -> np.ndarray:
        """
        Generate predictions (constant for all time steps).

        Args:
            n_samples: Number of samples to generate

        Returns:
            Predictions [N, 1, H, W]
        """
        if self.rates is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Tile the historical rates for all samples
        preds = np.tile(self.rates[np.newaxis, np.newaxis, :, :], (n_samples, 1, 1, 1))
        return preds.astype(np.float32)


class PersistenceBaseline:
    """
    Persistence baseline: use last month's normalized earthquake count as forecast.

    This captures short-term temporal patterns.
    """

    def __init__(self):
        pass

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions from input features.

        Args:
            x_test: Input features [N, L, C, H, W]
                   where L is lookback length, C is number of channels

        Returns:
            Predictions [N, 1, H, W]
        """
        if x_test.ndim != 5:
            raise ValueError(f"Expected 5D input [N, L, C, H, W], got {x_test.ndim}D")

        # Use the first channel (normalized frequency) from the last time step
        last_freq = x_test[:, -1, 0, :, :]  # [N, H, W]

        # Clip to [0, 1] range (in case of normalization artifacts)
        preds = np.clip(last_freq, 0, 1)

        # Add channel dimension
        preds = preds[:, np.newaxis, :, :]  # [N, 1, H, W]

        return preds.astype(np.float32)


class KernelDensityBaseline:
    """
    Kernel density (smoothed seismicity) baseline.

    Applies Gaussian smoothing to historical earthquake locations to create
    a smooth spatial probability map. This is a standard baseline in
    earthquake forecasting.

    Reference:
        Helmstetter et al. "High-resolution time-independent grid-based forecast
        for M≥5 earthquakes in California" (Seismological Research Letters 2007)
    """

    def __init__(self, sigma: float = 1.0, temporal_weight_halflife: Optional[float] = None):
        """
        Args:
            sigma: Standard deviation for Gaussian smoothing in grid cells.
                  sigma=1.0 means smoothing over ~1 cell radius.
            temporal_weight_halflife: If provided, apply exponential temporal weighting
                                     with this half-life (in number of time steps).
                                     More recent events get higher weight.
        """
        self.sigma = sigma
        self.temporal_weight_halflife = temporal_weight_halflife
        self.smoothed_rates = None

    def fit(self, y_train: np.ndarray) -> None:
        """
        Fit the model on training data.

        Args:
            y_train: Training labels [N, 1, H, W] or [N, H, W]
        """
        if y_train.ndim == 4 and y_train.shape[1] == 1:
            y_train = y_train[:, 0, :, :]  # [N, H, W]

        n_samples = y_train.shape[0]

        # Apply temporal weighting if specified
        if self.temporal_weight_halflife is not None:
            # Compute weights (exponential decay, most recent = 1.0)
            decay_factor = np.log(2) / self.temporal_weight_halflife
            time_indices = np.arange(n_samples)
            weights = np.exp(-decay_factor * (n_samples - 1 - time_indices))
            weights = weights[:, np.newaxis, np.newaxis]  # [N, 1, 1]

            # Weighted average
            weighted_sum = (y_train * weights).sum(axis=0)
            weight_sum = weights.sum()
            rates = weighted_sum / weight_sum
        else:
            # Simple average
            rates = y_train.mean(axis=0)  # [H, W]

        # Apply Gaussian smoothing
        self.smoothed_rates = gaussian_filter(rates, sigma=self.sigma, mode='constant', cval=0.0)

        # Normalize to [0, 1] range
        if self.smoothed_rates.max() > 0:
            self.smoothed_rates = self.smoothed_rates / self.smoothed_rates.max()

    def predict(self, n_samples: int) -> np.ndarray:
        """
        Generate predictions (constant for all time steps).

        Args:
            n_samples: Number of samples to generate

        Returns:
            Predictions [N, 1, H, W]
        """
        if self.smoothed_rates is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Tile the smoothed rates for all samples
        preds = np.tile(self.smoothed_rates[np.newaxis, np.newaxis, :, :],
                       (n_samples, 1, 1, 1))
        return preds.astype(np.float32)


class AdaptiveKernelDensityBaseline:
    """
    Adaptive kernel density baseline that uses recent seismicity.

    Similar to kernel density, but updates the forecast based on recent activity
    within a sliding window.
    """

    def __init__(self, sigma: float = 1.0, lookback_months: int = 12):
        """
        Args:
            sigma: Standard deviation for Gaussian smoothing
            lookback_months: Number of recent months to use for forecast
        """
        self.sigma = sigma
        self.lookback_months = lookback_months
        self.train_data = None

    def fit(self, y_train: np.ndarray) -> None:
        """
        Store training data for later use.

        Args:
            y_train: Training labels [N, 1, H, W] or [N, H, W]
        """
        if y_train.ndim == 4 and y_train.shape[1] == 1:
            y_train = y_train[:, 0, :, :]

        self.train_data = y_train

    def predict(self, x_test: np.ndarray, y_history: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions using recent activity.

        Args:
            x_test: Input features [N, L, C, H, W]
            y_history: Optional historical labels for test period.
                      If provided, use actual history. Otherwise, use x_test features.

        Returns:
            Predictions [N, 1, H, W]
        """
        if x_test.ndim != 5:
            raise ValueError(f"Expected 5D input [N, L, C, H, W], got {x_test.ndim}D")

        n_samples, lookback, n_channels, h, w = x_test.shape

        preds = np.zeros((n_samples, 1, h, w), dtype=np.float32)

        for i in range(n_samples):
            # Extract recent activity from features
            # Use channel 0 (normalized frequency) from all lookback steps
            recent_activity = x_test[i, :, 0, :, :]  # [L, H, W]

            # Sum over time and smooth
            activity_sum = recent_activity.sum(axis=0)  # [H, W]
            smoothed = gaussian_filter(activity_sum, sigma=self.sigma, mode='constant', cval=0.0)

            # Normalize
            if smoothed.max() > 0:
                smoothed = smoothed / smoothed.max()

            preds[i, 0, :, :] = smoothed

        return preds


class ETASProxyBaseline:
    """
    ETAS-inspired Decaying Aftershock Rate (DAR) baseline.

    Approximates the ETAS model (Ogata 1988) within a monthly gridded framework by
    combining a smoothed spatial background rate with an Omori-Utsu power-law decay
    of recent monthly event counts:

        P̂(t,i,j) = α × KDE(i,j) + (1-α)/Z × Σ_{Δ=1}^{L} X_{t-Δ,0,i,j} × Δ^{-p}

    where Z = Σ_{Δ=1}^{L} Δ^{-p}, X_{t-Δ,0,i,j} is the normalized event count at
    lag Δ months (channel 0 of the input tensor), p is the Omori-Utsu decay exponent,
    and α is the background weight. The KDE background (Gaussian-smoothed SPR) ensures
    everywhere-positive probabilities and finite log loss.

    Parameters p and α are tuned on validation PR-AUC.

    Note: Full ETAS maximum-likelihood fitting (Ogata 1988) requires sub-day inter-event
    times. At monthly gridded resolution, the integrated monthly count is the natural
    approximation; this baseline captures the core ETAS temporal mechanism (aftershock
    decay following Omori's law) within the same data framework as the ML models.

    References:
        Ogata, Y. (1988). Statistical models for earthquake occurrences and residual
            analysis for point processes. JASA, 83(401), 9–27.
        Utsu, T., Ogata, Y., & Matsu'ura, R. S. (1995). The centenary of the Omori
            formula for a decay law of aftershock activity. J. Phys. Earth, 43, 1–33.
    """

    def __init__(self, p: float = 1.0, alpha: float = 0.3, sigma: float = 0.5):
        """
        Args:
            p: Omori-Utsu decay exponent (weight for lag Δ = Δ^{-p}).
               Typical range: 0.8–1.2. Tune on validation.
            alpha: Weight for smoothed background rate (0 = pure triggered, 1 = pure background).
            sigma: Gaussian smoothing bandwidth for background rate (grid cells).
        """
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        self.background_rates = None  # [H, W], smoothed SPR

    def fit(self, y_train: np.ndarray) -> None:
        """
        Fit background rate from training labels.

        Args:
            y_train: Training labels [N, 1, H, W] or [N, H, W]
        """
        if y_train.ndim == 4 and y_train.shape[1] == 1:
            y_train = y_train[:, 0, :, :]

        raw_rates = y_train.mean(axis=0)  # [H, W]
        smoothed = gaussian_filter(raw_rates, sigma=self.sigma, mode='constant', cval=0.0)
        # Small floor ensures finite log loss everywhere (no zero-probability cells)
        self.background_rates = np.maximum(smoothed, 1e-6)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using Omori-weighted recent activity + smoothed background.

        Args:
            x_test: Input features [N, L, C, H, W]
                   Channel 0 (normalized event count) is used for the triggered component.

        Returns:
            Predictions [N, 1, H, W]
        """
        if x_test.ndim != 5:
            raise ValueError(f"Expected 5D input [N, L, C, H, W], got {x_test.ndim}D")
        if self.background_rates is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples, lookback, _, h, w = x_test.shape

        # Omori-Utsu weights: w(Δ) = Δ^{-p} for Δ = 1, 2, ..., L
        lags = np.arange(1, lookback + 1, dtype=np.float64)
        weights = lags ** (-self.p)    # [L]
        Z = weights.sum()

        # Channel 0: normalized event count in [0, 1]
        # x_test[:, L-1] = most recent month (lag Δ=1)
        # x_test[:, L-Δ] = month at lag Δ
        # Reverse so index 0 = most recent (lag 1), index L-1 = oldest (lag L)
        counts = x_test[:, :, 0, :, :]                # [N, L, H, W]
        counts_reversed = counts[:, ::-1, :, :]        # [N, L, H, W], index=lag-1

        # Weighted sum over lookback: triggered[n, h, w] = Σ_Δ counts[n, Δ-1] × Δ^{-p}
        triggered = np.einsum('nlhw,l->nhw', counts_reversed, weights) / Z  # [N, H, W]

        # Blend background and triggered components
        preds = (self.alpha * self.background_rates[np.newaxis, :, :]
                 + (1.0 - self.alpha) * triggered)     # [N, H, W]

        preds = np.clip(preds, 1e-7, 1.0)

        return preds[:, np.newaxis, :, :].astype(np.float32)  # [N, 1, H, W]


def tune_etas_proxy(y_train: np.ndarray,
                    x_val: np.ndarray,
                    y_val: np.ndarray,
                    p_grid: Optional[list] = None,
                    alpha_grid: Optional[list] = None) -> dict:
    """
    Grid-search optimal (p, alpha) for ETASProxyBaseline on validation PR-AUC.

    Args:
        y_train: Training labels [N, 1, H, W]
        x_val: Validation features [N, L, C, H, W]
        y_val: Validation labels [N, 1, H, W]
        p_grid: Omori exponent values to try (default: [0.8, 0.9, 1.0, 1.1, 1.2])
        alpha_grid: Background weight values to try (default: [0.1, 0.2, 0.3, 0.5, 0.7])

    Returns:
        dict with 'best_p', 'best_alpha', 'best_pr_auc', 'grid_results'
    """
    from .evaluation import compute_pr_auc

    if p_grid is None:
        p_grid = [0.8, 0.9, 1.0, 1.1, 1.2]
    if alpha_grid is None:
        alpha_grid = [0.1, 0.2, 0.3, 0.5, 0.7]

    best_pr = -1.0
    best_p = 1.0
    best_alpha = 0.3
    grid_results = []

    y_val_flat = y_val.reshape(-1)

    for p in p_grid:
        for alpha in alpha_grid:
            model = ETASProxyBaseline(p=p, alpha=alpha)
            model.fit(y_train)
            preds = model.predict(x_val)
            pr = compute_pr_auc(y_val_flat, preds.reshape(-1))
            grid_results.append({'p': p, 'alpha': alpha, 'pr_auc': pr})
            if pr > best_pr:
                best_pr = pr
                best_p = p
                best_alpha = alpha

    return {
        'best_p': best_p,
        'best_alpha': best_alpha,
        'best_pr_auc': best_pr,
        'grid_results': grid_results,
    }


def compare_baselines(y_train: np.ndarray,
                     x_test: np.ndarray,
                     y_test: np.ndarray,
                     sigma_values: list[float] = [0.5, 1.0, 2.0]) -> dict:
    """
    Compare multiple baseline models and return predictions.

    Args:
        y_train: Training labels [N, 1, H, W]
        x_test: Test features [N, L, C, H, W]
        y_test: Test labels [N, 1, H, W]
        sigma_values: List of sigma values to try for kernel smoothing

    Returns:
        Dictionary with baseline names as keys and predictions as values
    """
    from .evaluation import compute_roc_auc, compute_pr_auc, compute_brier_score

    n_test = x_test.shape[0]

    results = {}

    # Historical rate
    print("Fitting historical rate baseline...")
    hist_model = HistoricalRateBaseline()
    hist_model.fit(y_train)
    hist_preds = hist_model.predict(n_test)

    hist_roc = compute_roc_auc(y_test.reshape(-1), hist_preds.reshape(-1))
    hist_pr = compute_pr_auc(y_test.reshape(-1), hist_preds.reshape(-1))
    hist_brier = compute_brier_score(y_test.reshape(-1), hist_preds.reshape(-1))

    results['historical_rate'] = {
        'preds': hist_preds,
        'roc_auc': hist_roc,
        'pr_auc': hist_pr,
        'brier': hist_brier
    }

    print(f"  ROC-AUC: {hist_roc:.3f}, PR-AUC: {hist_pr:.3f}, Brier: {hist_brier:.3f}")

    # Persistence
    print("Evaluating persistence baseline...")
    pers_model = PersistenceBaseline()
    pers_preds = pers_model.predict(x_test)

    pers_roc = compute_roc_auc(y_test.reshape(-1), pers_preds.reshape(-1))
    pers_pr = compute_pr_auc(y_test.reshape(-1), pers_preds.reshape(-1))
    pers_brier = compute_brier_score(y_test.reshape(-1), pers_preds.reshape(-1))

    results['persistence'] = {
        'preds': pers_preds,
        'roc_auc': pers_roc,
        'pr_auc': pers_pr,
        'brier': pers_brier
    }

    print(f"  ROC-AUC: {pers_roc:.3f}, PR-AUC: {pers_pr:.3f}, Brier: {pers_brier:.3f}")

    # Kernel density with multiple sigma values
    for sigma in sigma_values:
        print(f"Fitting kernel density baseline (sigma={sigma})...")
        kde_model = KernelDensityBaseline(sigma=sigma)
        kde_model.fit(y_train)
        kde_preds = kde_model.predict(n_test)

        kde_roc = compute_roc_auc(y_test.reshape(-1), kde_preds.reshape(-1))
        kde_pr = compute_pr_auc(y_test.reshape(-1), kde_preds.reshape(-1))
        kde_brier = compute_brier_score(y_test.reshape(-1), kde_preds.reshape(-1))

        results[f'kernel_density_sigma{sigma}'] = {
            'preds': kde_preds,
            'roc_auc': kde_roc,
            'pr_auc': kde_pr,
            'brier': kde_brier,
            'sigma': sigma
        }

        print(f"  ROC-AUC: {kde_roc:.3f}, PR-AUC: {kde_pr:.3f}, Brier: {kde_brier:.3f}")

    return results
