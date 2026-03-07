from __future__ import annotations

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def chlorophyll_index(b5: np.ndarray, b4: np.ndarray) -> np.ndarray:
    # Proxy for algal bloom risk using red-edge and red reflectance.
    return safe_divide(b5 - b4, b5 + b4)


def chlorophyll_blue_green_proxy(
    b3: np.ndarray, b2: np.ndarray | None = None
) -> np.ndarray | None:
    if b2 is None:
        return None
    return safe_divide(b3, b2)


def turbidity_index(b4: np.ndarray, b3: np.ndarray) -> np.ndarray:
    return safe_divide(b4, b3)


def ndwi_index(b3: np.ndarray, b8: np.ndarray) -> np.ndarray:
    return safe_divide(b3 - b8, b3 + b8)

