from __future__ import annotations

import numpy as np


def binary_risk_label(
    risk_norm: np.ndarray,
    threshold: float = 0.65,
    water_mask: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(risk_norm)
    label = valid & (risk_norm >= threshold)
    if water_mask is not None:
        label = label & water_mask.astype(bool)
    return label.astype(np.uint8)


def multiclass_risk_label(
    risk_norm: np.ndarray,
    thresholds: dict[str, float],
    water_mask: np.ndarray | None = None,
) -> np.ndarray:
    classes = np.zeros(risk_norm.shape, dtype=np.uint8)
    valid = np.isfinite(risk_norm)
    if water_mask is not None:
        valid = valid & water_mask.astype(bool)
    yellow = thresholds.get("yellow", 0.30)
    red = thresholds.get("red", 0.65)
    infrared = thresholds.get("infrared", 0.85)

    classes[valid & (risk_norm >= yellow)] = 1
    classes[valid & (risk_norm >= red)] = 2
    classes[valid & (risk_norm >= infrared)] = 3
    return classes

