from __future__ import annotations

import numpy as np

from app.services.processing.indices import chlorophyll_index, ndwi_index, turbidity_index


def test_indices_expected_shapes_and_values() -> None:
    b3 = np.array([[0.4, 0.5]], dtype=np.float32)
    b4 = np.array([[0.3, 0.4]], dtype=np.float32)
    b5 = np.array([[0.6, 0.7]], dtype=np.float32)
    b8 = np.array([[0.2, 0.3]], dtype=np.float32)

    chlorophyll = chlorophyll_index(b5, b4)
    turbidity = turbidity_index(b4, b3)
    ndwi = ndwi_index(b3, b8)

    assert chlorophyll.shape == b3.shape
    assert turbidity.shape == b3.shape
    assert ndwi.shape == b3.shape
    assert np.all(np.isfinite(chlorophyll))
    assert np.all(np.isfinite(turbidity))
    assert np.all(ndwi > 0)

