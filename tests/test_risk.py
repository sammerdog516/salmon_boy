from __future__ import annotations

import numpy as np

from app.services.processing.risk import normalize_risk


def test_normalize_risk_to_unit_interval() -> None:
    raw = np.array([[2.0, 4.0], [8.0, 10.0]], dtype=np.float32)
    valid = np.array([[True, True], [False, True]])
    normalized = normalize_risk(raw, valid)

    assert np.nanmin(normalized) >= 0.0
    assert np.nanmax(normalized) <= 1.0
    assert np.isnan(normalized[1, 0])

