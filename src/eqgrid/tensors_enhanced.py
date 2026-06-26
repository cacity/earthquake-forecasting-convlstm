"""Enhanced tensor building with seismic and spatiotemporal features.

This module extends the original tensors.py with additional features (10 channels total):
- Original 3 channels: count_norm, max_mag_norm, log_energy
- Seismic features: b-value, mean_mag, mag_std, mean_depth
- Temporal features: recency_weight, rate_change
- Spatial features: spatial_density
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cli_utils import BBox
from .tensors import GridMeta, TensorBuildConfig, add_grid_index, add_time_bin, build_bins, build_grid_meta, save_grid_meta


def _fit_b_value(magnitudes: np.ndarray, mc: float = 0.0) -> float:
    """Compute b-value using maximum likelihood estimator (Aki 1965)."""
    mags = magnitudes[magnitudes >= mc]
    if len(mags) < 5:
        return 1.0
    mean_mag = mags.mean()
    b_value = np.log10(np.e) / (mean_mag - (mc - 0.05))
    return float(np.clip(b_value, 0.3, 2.5))


def _compute_spatial_density(count: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compute spatial density using neighborhood averaging."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(count, size=kernel_size, mode='constant')


def build_tensors_enhanced(
    events: pd.DataFrame,
    grid: GridMeta,
    bins: np.ndarray,
    config: TensorBuildConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build enhanced tensors with seismic and spatiotemporal features.

    Returns:
        X: [T, 10, H, W] enhanced feature tensor (10 channels)
        Y: [T, 1, H, W] binary label tensor
    """
    from .tensors import build_tensors as _build_tensors_base

    # First get base features (3 channels)
    X_base, Y = _build_tensors_base(events, grid, bins, config)

    if events.empty:
        return X_base, Y

    # Build additional features
    T = len(bins)
    H, W = grid.n_lat, grid.n_lon

    # Seismic features
    b_value = np.zeros((T, H, W), dtype=np.float32)
    mean_mag = np.zeros((T, H, W), dtype=np.float32)
    mag_std = np.zeros((T, H, W), dtype=np.float32)
    mean_depth = np.zeros((T, H, W), dtype=np.float32)

    # Temporal features
    recency = np.zeros((T, H, W), dtype=np.float32)
    rate_change = np.zeros((T, H, W), dtype=np.float32)

    # Spatial features
    spatial_density = np.zeros((T, H, W), dtype=np.float32)

    # Group events by time bin and cell
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

            mags = cell_events["mag"].values.astype(float)

            # Seismic features
            b_value[t, lat_idx, lon_idx] = _fit_b_value(mags)
            mean_mag[t, lat_idx, lon_idx] = mags.mean()
            if len(mags) > 1:
                mag_std[t, lat_idx, lon_idx] = mags.std()

            # Mean depth
            if "depth_km" in cell_events.columns:
                depths = cell_events["depth_km"].dropna().values
                if len(depths) > 0:
                    mean_depth[t, lat_idx, lon_idx] = depths.mean()

    # Compute recency-weighted activity
    for t in range(T):
        for dt in range(min(t + 1, 12)):
            past_t = t - dt
            weight = np.exp(-dt / 3.0)
            recency[t] += X_base[past_t, 0] * weight

    # Compute spatial density
    for t in range(T):
        spatial_density[t] = _compute_spatial_density(X_base[t, 0])

    # Compute rate of change
    for t in range(1, T):
        rate_change[t] = X_base[t, 0] - X_base[t - 1, 0]
    if T > 1:
        rate_change[0] = rate_change[1]

    # Normalize features
    b_value_norm = (b_value - 0.3) / 2.2
    mean_mag_norm = mean_mag / 10.0
    mag_std_norm = np.clip(mag_std / 5.0, 0, 1)
    mean_depth_norm = np.clip(mean_depth / 100.0, 0, 1)
    recency_norm = np.clip(recency, 0, 1)

    # X_base[:, 0] is clipped to [0, 1], so a 3x3 neighborhood average is
    # already bounded. Keeping the analytic scale avoids full-period leakage.
    spatial_density_norm = np.clip(spatial_density, 0, 1)

    # Difference of two [0, 1] count maps is analytically bounded in [-1, 1].
    # Do not scale by the full-period standard deviation.
    rate_change_norm = np.clip(rate_change, -1, 1)

    # Stack all features (10 channels total)
    X = np.stack([
        # Original 3 channels
        X_base[:, 0],           # 0: count_norm
        X_base[:, 1],           # 1: max_mag_norm
        X_base[:, 2],           # 2: log_energy

        # Seismic features (4 channels)
        b_value_norm,           # 3: b-value
        mean_mag_norm,          # 4: mean magnitude
        mag_std_norm,           # 5: magnitude std
        mean_depth_norm,        # 6: mean depth

        # Temporal features (2 channels)
        recency_norm,           # 7: recency-weighted activity
        rate_change_norm,       # 8: rate of change

        # Spatial features (1 channel)
        spatial_density_norm,   # 9: spatial density
    ], axis=1).astype(np.float32)

    return X, Y
