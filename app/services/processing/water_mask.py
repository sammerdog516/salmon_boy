from __future__ import annotations

import numpy as np


def compute_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.isfinite(ndwi) & (ndwi > threshold)

