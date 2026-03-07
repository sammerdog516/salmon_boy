from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon, mapping

from app.services.processing.risk import risk_category
from app.utils.geospatial import project_geometry


def _block_polygon(
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    transform: Any,
) -> Polygon:
    ul_x, ul_y = transform * (col_start, row_start)
    ur_x, ur_y = transform * (col_end, row_start)
    lr_x, lr_y = transform * (col_end, row_end)
    ll_x, ll_y = transform * (col_start, row_end)
    return Polygon([(ul_x, ul_y), (ur_x, ur_y), (lr_x, lr_y), (ll_x, ll_y), (ul_x, ul_y)])


def aggregate_raster_to_grid_geojson(
    risk: np.ndarray,
    chlorophyll: np.ndarray,
    turbidity: np.ndarray,
    water_mask: np.ndarray,
    transform: Any,
    crs: str,
    scene_id: str,
    thresholds: dict[str, float],
    block_size: int = 32,
    path_id: str | None = None,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    rows, cols = risk.shape

    for row_start in range(0, rows, block_size):
        row_end = min(rows, row_start + block_size)
        for col_start in range(0, cols, block_size):
            col_end = min(cols, col_start + block_size)
            block_risk = risk[row_start:row_end, col_start:col_end]
            block_chl = chlorophyll[row_start:row_end, col_start:col_end]
            block_turb = turbidity[row_start:row_end, col_start:col_end]
            block_water = water_mask[row_start:row_end, col_start:col_end]

            valid = np.isfinite(block_risk)
            if not np.any(valid):
                continue

            score = float(np.nanmean(block_risk[valid]))
            polygon = _block_polygon(row_start, row_end, col_start, col_end, transform)
            polygon_wgs84 = project_geometry(polygon, src_crs=crs, dst_crs="EPSG:4326")

            properties = {
                "scene_id": scene_id,
                "risk_score": score,
                "risk_category": risk_category(score, thresholds),
                "chlorophyll_index_mean": float(np.nanmean(block_chl[block_water]))
                if np.any(block_water)
                else 0.0,
                "turbidity_index_mean": float(np.nanmean(block_turb[block_water]))
                if np.any(block_water)
                else 0.0,
                "water_fraction": float(np.count_nonzero(block_water)) / float(block_water.size),
                "pixel_count": int(block_risk.size),
                "water_pixel_count": int(np.count_nonzero(block_water)),
            }
            if path_id is not None:
                properties["path_id"] = path_id

            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(polygon_wgs84),
                    "properties": properties,
                }
            )

    return {"type": "FeatureCollection", "features": features}

