from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MigrationPathService:
    def __init__(self, migration_geojson_path: Path) -> None:
        self.migration_geojson_path = migration_geojson_path

    def load_feature_collection(self) -> dict[str, Any]:
        if not self.migration_geojson_path.exists():
            return {"type": "FeatureCollection", "features": []}
        payload = json.loads(self.migration_geojson_path.read_text(encoding="utf-8"))
        if payload.get("type") != "FeatureCollection":
            raise ValueError("Migration paths file must be a GeoJSON FeatureCollection.")
        features = payload.get("features", [])
        normalized = []
        for index, feature in enumerate(features):
            if not isinstance(feature, dict):
                continue
            feature = feature.copy()
            properties = feature.get("properties") or {}
            path_id = (
                str(feature.get("id"))
                if feature.get("id") is not None
                else str(properties.get("path_id") or properties.get("id") or f"path-{index+1}")
            )
            feature["id"] = path_id
            properties["path_id"] = path_id
            properties["name"] = properties.get("name") or path_id
            feature["properties"] = properties
            normalized.append(feature)
        return {"type": "FeatureCollection", "features": normalized}

    def list_paths(self) -> list[dict[str, str]]:
        feature_collection = self.load_feature_collection()
        output = []
        for feature in feature_collection["features"]:
            properties = feature.get("properties", {})
            output.append(
                {
                    "path_id": properties.get("path_id", feature.get("id")),
                    "name": properties.get("name", feature.get("id")),
                    "feature_type": feature.get("geometry", {}).get("type", "Unknown"),
                }
            )
        return output

    def get_path_feature(self, path_id: str) -> dict[str, Any] | None:
        for feature in self.load_feature_collection()["features"]:
            if str(feature.get("id")) == str(path_id):
                return feature
            if str(feature.get("properties", {}).get("path_id")) == str(path_id):
                return feature
        return None

