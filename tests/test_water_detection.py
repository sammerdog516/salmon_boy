from __future__ import annotations

import numpy as np

from app.services.processing.water_mask import compute_water_mask, compute_water_mask_refined


def test_refined_water_mask_applies_nir_to_green_gate() -> None:
    ndwi = np.array([[0.2, 0.2], [0.2, -0.1]], dtype=np.float32)
    b3 = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    b4 = np.array([[0.15, 0.15], [0.15, 0.15]], dtype=np.float32)
    b8 = np.array([[0.15, 0.35], [0.18, 0.3]], dtype=np.float32)

    base = compute_water_mask(ndwi=ndwi, threshold=0.0)
    refined = compute_water_mask_refined(
        ndwi=ndwi,
        b3=b3,
        b4=b4,
        b8=b8,
        threshold=0.0,
        nir_to_green_ratio_max=1.15,
    )

    # Base NDWI marks first 3 as water, but high NIR/green ratio should remove one.
    assert int(base.sum()) == 3
    assert int(refined.sum()) == 2
    assert not bool(refined[0, 1])


def test_refined_water_mask_uses_swir_when_available() -> None:
    ndwi = np.array(
        [
            [0.25, 0.22, 0.20],
            [0.24, 0.23, 0.21],
            [0.26, 0.22, 0.20],
        ],
        dtype=np.float32,
    )
    b3 = np.full((3, 3), 0.20, dtype=np.float32)
    b4 = np.full((3, 3), 0.12, dtype=np.float32)
    b8 = np.full((3, 3), 0.10, dtype=np.float32)
    # Center cell mimics bright urban/swir response and should be filtered.
    b11 = np.full((3, 3), 0.07, dtype=np.float32)
    b12 = np.full((3, 3), 0.06, dtype=np.float32)
    b11[1, 1] = 0.24
    b12[1, 1] = 0.22

    without_swir = compute_water_mask_refined(
        ndwi=ndwi,
        b3=b3,
        b4=b4,
        b8=b8,
        threshold=0.0,
        nir_to_green_ratio_max=1.15,
    )
    with_swir = compute_water_mask_refined(
        ndwi=ndwi,
        b3=b3,
        b4=b4,
        b8=b8,
        b11=b11,
        b12=b12,
        threshold=0.0,
        nir_to_green_ratio_max=1.15,
    )

    assert bool(without_swir[1, 1])
    assert not bool(with_swir[1, 1])
