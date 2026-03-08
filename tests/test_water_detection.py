from __future__ import annotations

import numpy as np

from app.services.processing.water_mask import compute_water_mask, compute_water_mask_refined


def test_refined_water_mask_applies_nir_to_green_gate() -> None:
    ndwi = np.array([[0.2, 0.2], [0.2, -0.1]], dtype=np.float32)
    b3 = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    b8 = np.array([[0.15, 0.35], [0.18, 0.3]], dtype=np.float32)

    base = compute_water_mask(ndwi=ndwi, threshold=0.0)
    refined = compute_water_mask_refined(
        ndwi=ndwi,
        b3=b3,
        b8=b8,
        threshold=0.0,
        nir_to_green_ratio_max=1.15,
    )

    # Base NDWI marks first 3 as water, but high NIR/green ratio should remove one.
    assert int(base.sum()) == 3
    assert int(refined.sum()) == 2
    assert not bool(refined[0, 1])
