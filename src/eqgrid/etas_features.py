"""ETAS parameter computation for enhanced feature channels.

This module computes ETAS (Epidemic Type Aftershock Sequence) model parameters
as additional features for earthquake forecasting.

ETAS model parameters:
- mu (μ): Background seismicity rate
- K: Aftershock productivity
- c: Omori time constant
- p: Omori decay exponent
- alpha (α): Triggering efficiency
- d: Spatial characteristic distance
- q: Spatial decay exponent
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.ndimage import gaussian_filter


def compute_omori_params(
    times: np.ndarray,
    mainshock_time: float,
    magnitudes: np.ndarray,
    mc: float = 0.0,
) -> dict:
    """Compute Omori-Utsu parameters from aftershock sequence.

    Args:
        times: Array of event times (days since mainshock)
        mainshock_time: Mainshock time
        magnitudes: Array of event magnitudes
        mc: Magnitude of completeness

    Returns:
        dict with 'c', 'p', 'K' parameters
    """
    # Filter aftershocks (events after mainshock)
    aftershock_mask = times > mainshock_time
    aftershock_times = times[aftershock_mask]
    aftershock_mags = magnitudes[aftershock_mask]

    if len(aftershock_times) < 5:
        return {'c': 0.01, 'p': 1.0, 'K': 0.0}

    # Time since mainshock
    t = aftershock_times - mainshock_time

    # Fit Omori law: n(t) = K / (t + c)^p
    # Use log-transform for linear fitting
    # log(n) = log(K) - p * log(t + c)

    # Bin the data
    t_max = t.max()
    if t_max < 1.0:
        return {'c': 0.01, 'p': 1.0, 'K': 0.0}

    # Create time bins
    n_bins = min(int(t_max) + 1, 100)
    bin_edges = np.linspace(0, t_max, n_bins + 1)
    bin_counts, _ = np.histogram(t, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Only use bins with events
    valid = bin_counts > 0
    if valid.sum() < 3:
        return {'c': 0.01, 'p': 1.0, 'K': 0.0}

    t_valid = bin_centers[valid]
    n_valid = bin_counts[valid]

    # Fit log(n) = log(K) - p * log(t + c)
    # First estimate c from early time behavior
    c_estimate = max(t_valid[n_valid > 0].min() * 0.1, 0.001)

    try:
        # Try different c values and find best fit
        def omori_residual(params, t_data, n_data):
            c, p, log_K = params
            if c <= 0 or p <= 0:
                return 1e10
            predicted = log_K - p * np.log(t_data + c)
            actual = np.log(n_data + 1e-10)
            return np.sum((predicted - actual) ** 2)

        result = optimize.minimize(
            omori_residual,
            x0=[c_estimate, 1.0, np.log(n_valid.max() + 1)],
            args=(t_valid, n_valid),
            method='L-BFGS-B',
            bounds=[(0.001, 10.0), (0.5, 3.0), (-10, 10)]
        )

        c_fit, p_fit, log_K_fit = result.x
        K_fit = np.exp(log_K_fit)

        return {
            'c': float(c_fit),
            'p': float(p_fit),
            'K': float(K_fit)
        }
    except Exception:
        return {'c': 0.01, 'p': 1.0, 'K': 0.0}


def compute_background_rate(
    events_df: pd.DataFrame,
    grid_lat: int,
    grid_lon: int,
    sigma: float = 1.0,
) -> float:
    """Compute background seismicity rate (mu) for a grid cell.

    Args:
        events_df: DataFrame with event data for the cell
        grid_lat: Grid latitude index
        grid_lon: Grid longitude index
        sigma: Gaussian smoothing sigma
    Returns:
        Background rate estimate
    """
    if len(events_df) < 5:
        return 0.0

    # Simple estimate: average events per time unit
    # In practice, this would use a declustering algorithm
    return len(events_df) / 100.0  # Normalized


def compute_triggering_efficiency(
    events_df: pd.DataFrame,
    mc: float = 0.0,
) -> float:
    """Compute triggering efficiency (alpha) from magnitude distribution.

    In ETAS: lambda(t) = mu + K * sum(10^(alpha * (M_i - Mc)) / (t - t_i + c)^p)

    Args:
        events_df: DataFrame with event data
        mc: Magnitude of completeness

    Returns:
        Alpha estimate (triggering efficiency)
    """
    if len(events_df) < 10:
        return 0.5

    mags = events_df["mag"].values
    mags = mags[mags >= mc]

    if len(mags) < 10:
        return 0.5

    # Estimate alpha from productivity law: N ~ 10^(alpha * (M - Mc))
    # Fit log(productivity) vs magnitude
    # For simplicity, use b-value relationship
    mean_mag = mags.mean()
    # alpha is typically around 0.5-1.0
    alpha = min(max(mean_mag / 10.0, 0.3), 1.5)

    return float(alpha)


def compute_spatial_decay_params(
    events_df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
) -> dict:
    """Compute spatial decay parameters (d, q).

    In ETAS: g(r) = 1 / (r^2 + d^2)^q

    Args:
        events_df: DataFrame with event data
        center_lat: Center latitude
        center_lon: Center longitude

    Returns:
        dict with 'd' and 'q'
    """
    if len(events_df) < 5:
        return {'d': 1.0, 'q': 1.0}

    # Compute distances from center
    lats = events_df["lat"].values
    lons = events_df["lon"].values

    # Simple Euclidean distance in degrees
    distances = np.sqrt((lats - center_lat) ** 2 + (lons - center_lon) ** 2)

    if len(distances) < 5:
        return {'d': 1.0, 'q': 1.0}

    # Estimate d (characteristic distance)
    d = np.percentile(distances, 50)  # Median distance

    # Estimate q (decay exponent)
    # Fit log(density) vs log(distance)
    # For simplicity, use typical value
    q = 1.0

    return {
        'd': float(d),
        'q': float(q)
    }


def compute_etas_features(
    events: pd.DataFrame,
    grid: 'GridMeta',
    bins: np.ndarray,
) -> dict:
    """Compute ETAS parameters for each grid cell and time bin.

    Args:
        events: DataFrame with columns [time, lat, lon, mag, depth_km, lon_idx, lat_idx, cell_id, time_bin]
        grid: GridMeta object
        bins: Array of time bins

    Returns:
        dict with ETAS feature arrays:
            - mu: [T, H, W] background rate
            - K: [T, H, W] aftershock productivity
            - c: [T, H, W] Omori time constant
            - p: [T, H, W] Omori decay exponent
            - alpha: [T, H, W] triggering efficiency
            - d: [T, H, W] spatial characteristic distance
            - q: [T, H, W] spatial decay exponent
    """
    T = len(bins)
    H, W = grid.n_lat, grid.n_lon

    # Initialize arrays
    mu = np.zeros((T, H, W), dtype=np.float32)
    K = np.zeros((T, H, W), dtype=np.float32)
    c = np.zeros((T, H, W), dtype=np.float32)
    p = np.zeros((T, H, W), dtype=np.float32)
    alpha = np.zeros((T, H, W), dtype=np.float32)
    d = np.zeros((T, H, W), dtype=np.float32)
    q = np.zeros((T, H, W), dtype=np.float32)

    # Group events by time bin and cell
    if events.empty:
        return {
            'mu': mu, 'K': K, 'c': c, 'p': p,
            'alpha': alpha, 'd': d, 'q': q
        }

    events = events.copy()
    events["time_bin"] = pd.to_datetime(events["time_bin"], utc=True)
    bin_index = pd.DatetimeIndex(pd.to_datetime(bins, utc=True))

    for t in range(T):
        # Get events in this time bin
        mask = events["time_bin"] == bin_index[t]
        if not mask.any():
            continue

        bin_events = events[mask]

        for cell_id in bin_events["cell_id"].unique():
            cell_mask = bin_events["cell_id"] == cell_id
            cell_events = bin_events[cell_mask]

            # Get grid indices
            lat_idx = int(cell_events["lat_idx"].iloc[0])
            lon_idx = int(cell_events["lon_idx"].iloc[0])

            if not (0 <= lat_idx < H and 0 <= lon_idx < W):
                continue

            # Compute ETAS parameters
            # Background rate (mu)
            mu[t, lat_idx, lon_idx] = compute_background_rate(
                cell_events, lat_idx, lon_idx
            )

            # Triggering efficiency (alpha)
            alpha[t, lat_idx, lon_idx] = compute_triggering_efficiency(
                cell_events, mc=3.0
            )

            # Omori parameters (c, p, K)
            if len(cell_events) >= 5:
                times = cell_events["time"].astype(np.int64) / 1e9 / 86400  # Convert to days
                mags = cell_events["mag"].values
                omori = compute_omori_params(
                    times.values, times.min(), mags, mc=3.0
                )
                c[t, lat_idx, lon_idx] = omori['c']
                p[t, lat_idx, lon_idx] = omori['p']
                K[t, lat_idx, lon_idx] = omori['K']

            # Spatial decay parameters (d, q)
            center_lat = cell_events["lat"].mean()
            center_lon = cell_events["lon"].mean()
            spatial = compute_spatial_decay_params(
                cell_events, center_lat, center_lon
            )
            d[t, lat_idx, lon_idx] = spatial['d']
            q[t, lat_idx, lon_idx] = spatial['q']

    return {
        'mu': mu, 'K': K, 'c': c, 'p': p,
        'alpha': alpha, 'd': d, 'q': q
    }
