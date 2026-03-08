from __future__ import annotations

import numpy as np

from app.services.training.weak_labels import binary_risk_label


def test_binary_risk_label_uses_threshold_and_water_mask() -> None:
    risk = np.array([[0.2, 0.7], [0.9, np.nan]], dtype=np.float32)
    water = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    label = binary_risk_label(risk_norm=risk, threshold=0.65, water_mask=water)
    expected = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    assert np.array_equal(label, expected)

