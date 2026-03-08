from __future__ import annotations

import numpy as np


def compute_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.isfinite(ndwi) & (ndwi > threshold)


def compute_water_mask_refined(
    ndwi: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    threshold: float = 0.0,
    nir_to_green_ratio_max: float = 1.15,
    ndvi_max: float = 0.15,
) -> np.ndarray:
    """Water detection for MVP inference.

    Uses NDWI as the primary detector, then applies a simple spectral gate
    (NIR-to-green ratio + NDVI cap) to reduce land false-positives.
    """
    adaptive_threshold = threshold
    finite_ndwi = ndwi[np.isfinite(ndwi)]
    if finite_ndwi.size >= 1024:
        adaptive_threshold = max(threshold, _otsu_threshold(np.clip(finite_ndwi, -1.0, 1.0)))

    base = compute_water_mask(ndwi=ndwi, threshold=adaptive_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(b3) & (b3 != 0), b8 / b3, np.inf)
        ndvi = np.where(
            np.isfinite(b8) & np.isfinite(b4) & ((b8 + b4) != 0),
            (b8 - b4) / (b8 + b4),
            np.inf,
        )
    spectral_gate = np.isfinite(ratio) & (ratio <= nir_to_green_ratio_max)
    ndvi_gate = np.isfinite(ndvi) & (ndvi <= ndvi_max)
    return base & spectral_gate & ndvi_gate


def _otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    hist, bin_edges = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0

    prob = hist / total
    omega = np.cumsum(prob)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b = np.full_like(denom, -1.0)
    valid = denom > 0
    sigma_b[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    idx = int(np.nanargmax(sigma_b))
    return float(centers[idx])

