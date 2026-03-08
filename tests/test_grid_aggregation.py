from __future__ import annotations

import numpy as np
from affine import Affine

from app.services.processing.grid import aggregate_raster_to_grid_geojson


def test_grid_aggregation_keeps_non_water_blocks_with_zero_risk() -> None:
    risk = np.array(
        [
            [0.9, 0.8, np.nan, np.nan],
            [0.7, 0.6, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    chlorophyll = np.nan_to_num(risk, nan=0.0)
    turbidity = np.nan_to_num(risk, nan=0.0)
    water_mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    grid = aggregate_raster_to_grid_geojson(
        risk=risk,
        chlorophyll=chlorophyll,
        turbidity=turbidity,
        water_mask=water_mask,
        transform=Affine.identity(),
        crs="EPSG:4326",
        scene_id="scene-test",
        thresholds={"blue": 0.0, "yellow": 0.30, "red": 0.65, "infrared": 0.85},
        block_size=2,
    )

    assert grid["type"] == "FeatureCollection"
    assert len(grid["features"]) == 4

    # Bottom-right block is pure non-water -> retained with explicit zero risk.
    target = grid["features"][-1]["properties"]
    assert target["water_detected"] is False
    assert target["water_fraction"] == 0.0
    assert target["risk_score"] == 0.0
