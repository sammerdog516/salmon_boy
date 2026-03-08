from __future__ import annotations

import numpy as np


def compute_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.isfinite(ndwi) & (ndwi > threshold)


def compute_water_mask_refined(
    ndwi: np.ndarray,
    b3: np.ndarray,
    b8: np.ndarray,
    threshold: float = 0.0,
    nir_to_green_ratio_max: float = 1.15,
) -> np.ndarray:
    """Water detection for MVP inference.

    Uses NDWI as the primary detector, then applies a simple spectral gate
    (NIR-to-green ratio) to reduce land false-positives.
    """
    base = compute_water_mask(ndwi=ndwi, threshold=threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(b3) & (b3 != 0), b8 / b3, np.inf)
    spectral_gate = np.isfinite(ratio) & (ratio <= nir_to_green_ratio_max)
    return base & spectral_gate

