from __future__ import annotations

import numpy as np


def compute_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.isfinite(ndwi) & (ndwi > threshold)


def compute_water_mask_refined(
    ndwi: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    b11: np.ndarray | None = None,
    b12: np.ndarray | None = None,
    threshold: float = 0.0,
    nir_to_green_ratio_max: float = 1.15,
    ndvi_max: float = 0.15,
) -> np.ndarray:
    """Water detection for MVP inference.

    Uses NDWI as the primary detector, then applies spectral gates
    (NIR-to-green ratio + NDVI cap). If SWIR bands are available, a
    conservative MNDWI/AWEI-like refinement is applied to reduce urban
    and bare-land false positives.
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
    mask = base & spectral_gate & ndvi_gate

    if b11 is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi = np.where(
                np.isfinite(b3) & np.isfinite(b11) & ((b3 + b11) != 0),
                (b3 - b11) / (b3 + b11),
                -np.inf,
            )
        swir_gate = np.isfinite(mndwi) & (mndwi > -0.05)
        mask = mask & swir_gate

    if b11 is not None and b12 is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            # AWEI-like water enhancement for shadow/urban suppression.
            awei = 4.0 * (b3 - b11) - (0.25 * b8 + 2.75 * b12)
        awei_gate = np.isfinite(awei) & (awei > -0.02)
        mask = mask & awei_gate

    return _majority_filter(mask.astype(bool), kernel_size=3, min_neighbors=4)


def _majority_filter(mask: np.ndarray, kernel_size: int = 3, min_neighbors: int = 5) -> np.ndarray:
    if kernel_size != 3:
        raise ValueError("Only 3x3 kernel is supported in MVP majority filter.")
    if mask.size == 0:
        return mask
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return mask

    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
    neighbors = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return mask & (neighbors >= min_neighbors)


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

