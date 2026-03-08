from __future__ import annotations

from statistics import mean
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.utils.geospatial import buffer_geometry_meters


def _as_path_geometry(path_features: list[dict[str, Any]]) -> BaseGeometry:
    geometries = [shape(feature["geometry"]) for feature in path_features if feature.get("geometry")]
    if not geometries:
        raise ValueError("No valid migration path geometries were found.")
    if len(geometries) == 1:
        return geometries[0]
    return unary_union(geometries)


def summarize_grid_near_paths(
    grid_feature_collection: dict[str, Any],
    path_features: list[dict[str, Any]],
    buffer_meters: float,
    selected_path_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_geometry = _as_path_geometry(path_features)
    path_buffer = buffer_geometry_meters(path_geometry, buffer_meters=buffer_meters)

    selected_features = []
    risk_scores = []
    for feature in grid_feature_collection.get("features", []):
        geometry = shape(feature.get("geometry"))
        if not geometry.intersects(path_buffer):
            continue
        feature_copy = {
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": dict(feature.get("properties", {})),
        }
        if selected_path_id:
            feature_copy["properties"]["path_id"] = selected_path_id
        selected_features.append(feature_copy)
        score = feature_copy["properties"].get("risk_score")
        if isinstance(score, (int, float)):
            risk_scores.append(float(score))

    summary = {
        "path_id": selected_path_id,
        "intersecting_cell_count": len(selected_features),
        "risk_mean": float(mean(risk_scores)) if risk_scores else 0.0,
        "risk_max": float(max(risk_scores)) if risk_scores else 0.0,
        "risk_min": float(min(risk_scores)) if risk_scores else 0.0,
        "buffer_meters": buffer_meters,
    }

    filtered_fc = {
        "type": "FeatureCollection",
        "features": selected_features,
    }
    return filtered_fc, summary

