from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import rasterio

from app.core.config import Settings
from app.core.constants import REQUIRED_SENTINEL_BANDS
from app.services.migration.loader import MigrationPathService
from app.services.migration.summarizer import summarize_grid_near_paths
from app.services.processing.grid import aggregate_raster_to_grid_geojson
from app.services.processing.indices import chlorophyll_index, ndwi_index, turbidity_index
from app.services.processing.raster import RasterBundle, load_and_align_bands
from app.services.processing.risk import score_risk, summarize_risk, temperature_proxy_stub
from app.services.processing.water_mask import compute_water_mask
from app.services.storage.metadata_store import MetadataStore


class ProcessingService:
    def __init__(
        self,
        settings: Settings,
        metadata_store: MetadataStore,
        migration_service: MigrationPathService,
    ) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self.migration_service = migration_service

    def process_scene_by_id(
        self,
        scene_id: str,
        aoi_bbox: list[float] | None = None,
        aoi_crs: str = "EPSG:4326",
        include_grid: bool = True,
        grid_block_size: int | None = None,
        migration_path_id: str | None = None,
        migration_buffer_meters: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        scene = self.metadata_store.get_scene(scene_id)
        if scene is None:
            raise ValueError(f"Scene not found: {scene_id}")
        assets = scene.get("assets")
        if not isinstance(assets, dict):
            raise ValueError(f"Scene {scene_id} has no usable asset mapping.")

        return self.process_assets(
            scene_id=scene_id,
            assets={str(k): str(v) for k, v in assets.items()},
            aoi_bbox=aoi_bbox,
            aoi_crs=aoi_crs,
            include_grid=include_grid,
            grid_block_size=grid_block_size,
            migration_path_id=migration_path_id,
            migration_buffer_meters=migration_buffer_meters,
            persist=persist,
        )

    def process_assets(
        self,
        scene_id: str,
        assets: dict[str, str],
        aoi_bbox: list[float] | None = None,
        aoi_crs: str = "EPSG:4326",
        include_grid: bool = True,
        grid_block_size: int | None = None,
        migration_path_id: str | None = None,
        migration_buffer_meters: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        raster_bundle = load_and_align_bands(
            assets=assets,
            required_bands=REQUIRED_SENTINEL_BANDS,
            aoi_bbox=aoi_bbox,
            aoi_crs=aoi_crs,
        )
        features = self._compute_features(raster_bundle)

        risk_raw, risk_normalized = score_risk(
            chlorophyll=features["chlorophyll"],
            turbidity=features["turbidity"],
            temperature=features["temperature_proxy"],
            water_mask=features["water_mask"],
        )
        summary = summarize_risk(
            risk=risk_normalized,
            chlorophyll=features["chlorophyll"],
            turbidity=features["turbidity"],
            water_mask=features["water_mask"],
        )
        summary["ndwi_mean"] = (
            float(np.nanmean(features["ndwi"][features["water_mask"]]))
            if np.any(features["water_mask"])
            else 0.0
        )

        block_size = grid_block_size or self.settings.default_grid_block_size
        grid = (
            aggregate_raster_to_grid_geojson(
                risk=risk_normalized,
                chlorophyll=features["chlorophyll"],
                turbidity=features["turbidity"],
                water_mask=features["water_mask"],
                transform=raster_bundle.transform,
                crs=raster_bundle.crs,
                scene_id=scene_id,
                thresholds=self.settings.heatmap_thresholds,
                block_size=block_size,
            )
            if include_grid
            else None
        )

        path_summary = None
        if grid and migration_path_id:
            path_feature = self.migration_service.get_path_feature(migration_path_id)
            if path_feature is None:
                raise ValueError(f"Migration path not found: {migration_path_id}")
            buffer_meters = migration_buffer_meters or self.settings.default_migration_buffer_meters
            grid, path_summary = summarize_grid_near_paths(
                grid_feature_collection=grid,
                path_features=[path_feature],
                buffer_meters=buffer_meters,
                selected_path_id=migration_path_id,
            )

        processed_scene_id = f"proc-{uuid4().hex[:12]}"
        artifact_paths: dict[str, str] = {}
        if persist:
            artifact_paths = self._persist_artifacts(
                processed_scene_id=processed_scene_id,
                scene_id=scene_id,
                raster_bundle=raster_bundle,
                risk_normalized=risk_normalized,
                risk_raw=risk_raw,
                summary=summary,
                grid=grid,
            )
            self.metadata_store.save_processed_scene(
                processed_scene_id,
                {
                    "processed_scene_id": processed_scene_id,
                    "scene_id": scene_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "summary": summary,
                    "artifact_paths": artifact_paths,
                    "has_grid": grid is not None,
                    "path_summary": path_summary,
                },
            )

        return {
            "processed_scene_id": processed_scene_id,
            "scene_id": scene_id,
            "summary": summary,
            "artifact_paths": artifact_paths,
            "grid": grid,
            "path_summary": path_summary,
        }

    def get_processed_scene(self, processed_scene_id: str) -> dict[str, Any] | None:
        return self.metadata_store.get_processed_scene(processed_scene_id)

    def list_processed_scenes(self) -> list[dict[str, Any]]:
        return self.metadata_store.list_processed_scenes()

    def load_grid_artifact(self, processed_scene_id: str) -> dict[str, Any]:
        record = self.get_processed_scene(processed_scene_id)
        if record is None:
            raise ValueError(f"Processed scene not found: {processed_scene_id}")
        grid_path = record.get("artifact_paths", {}).get("grid_geojson")
        if not grid_path:
            raise ValueError(f"No grid artifact available for {processed_scene_id}")
        payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
        if payload.get("type") != "FeatureCollection":
            raise ValueError("Grid artifact is not valid GeoJSON.")
        return payload

    def _compute_features(self, bundle: RasterBundle) -> dict[str, np.ndarray]:
        b3 = bundle.arrays["B3"]
        b4 = bundle.arrays["B4"]
        b5 = bundle.arrays["B5"]
        b8 = bundle.arrays["B8"]

        chlorophyll = chlorophyll_index(b5=b5, b4=b4)
        turbidity = turbidity_index(b4=b4, b3=b3)
        ndwi = ndwi_index(b3=b3, b8=b8)
        water_mask = compute_water_mask(ndwi=ndwi, threshold=self.settings.ndwi_water_threshold)
        temperature = temperature_proxy_stub(chlorophyll)

        return {
            "chlorophyll": chlorophyll,
            "turbidity": turbidity,
            "ndwi": ndwi,
            "water_mask": water_mask,
            "temperature_proxy": temperature,
        }

    def _persist_artifacts(
        self,
        processed_scene_id: str,
        scene_id: str,
        raster_bundle: RasterBundle,
        risk_normalized: np.ndarray,
        risk_raw: np.ndarray,
        summary: dict[str, Any],
        grid: dict[str, Any] | None,
    ) -> dict[str, str]:
        output_dir = self.settings.resolve_path(self.settings.artifacts_dir) / "processed" / processed_scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        risk_path = output_dir / "risk_normalized.tif"
        self._write_risk_geotiff(risk_path, risk_normalized, raster_bundle)

        risk_raw_path = output_dir / "risk_raw.tif"
        self._write_risk_geotiff(risk_raw_path, risk_raw, raster_bundle)

        summary_path = output_dir / "summary.json"
        summary_payload = {
            "processed_scene_id": processed_scene_id,
            "scene_id": scene_id,
            "summary": summary,
            "thresholds": self.settings.heatmap_thresholds,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        artifacts = {
            "risk_normalized_tif": str(risk_path),
            "risk_raw_tif": str(risk_raw_path),
            "summary_json": str(summary_path),
        }
        if grid is not None:
            grid_path = output_dir / "grid.geojson"
            grid_path.write_text(json.dumps(grid, indent=2), encoding="utf-8")
            artifacts["grid_geojson"] = str(grid_path)
        return artifacts

    def _write_risk_geotiff(
        self,
        destination: Path,
        array: np.ndarray,
        raster_bundle: RasterBundle,
    ) -> None:
        nodata_value = -9999.0
        writable = np.where(np.isfinite(array), array, nodata_value).astype(np.float32)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            height=raster_bundle.height,
            width=raster_bundle.width,
            count=1,
            dtype="float32",
            crs=raster_bundle.crs,
            transform=raster_bundle.transform,
            nodata=nodata_value,
        ) as dst:
            dst.write(writable, 1)
