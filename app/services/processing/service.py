from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from shapely.geometry import shape

from app.core.config import Settings
from app.core.constants import OPTIONAL_SENTINEL_BANDS, REQUIRED_SENTINEL_BANDS
from app.services.migration.loader import MigrationPathService
from app.services.migration.summarizer import summarize_grid_near_paths
from app.services.processing.grid import aggregate_raster_to_grid_geojson
from app.services.processing.indices import chlorophyll_index, ndwi_index, turbidity_index
from app.services.processing.raster import RasterBundle, load_and_align_bands
from app.services.processing.risk import score_risk, summarize_risk, temperature_proxy_stub
from app.services.processing.water_mask import compute_water_mask_refined
from app.services.storage.cache_manager import CacheManager
from app.services.storage.metadata_store import MetadataStore
from app.utils.geospatial import buffer_geometry_meters


class ProcessingService:
    def __init__(
        self,
        settings: Settings,
        cache_manager: CacheManager,
        metadata_store: MetadataStore,
        migration_service: MigrationPathService,
    ) -> None:
        self.settings = settings
        self.cache_manager = cache_manager
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
            scene_metadata=scene,
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
        scene_metadata: dict[str, Any] | None = None,
        aoi_bbox: list[float] | None = None,
        aoi_crs: str = "EPSG:4326",
        include_grid: bool = True,
        grid_block_size: int | None = None,
        migration_path_id: str | None = None,
        migration_buffer_meters: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        block_size = grid_block_size or self.settings.default_grid_block_size
        effective_bbox, effective_bbox_crs = self._resolve_processing_bbox(
            scene_metadata=scene_metadata,
            aoi_bbox=aoi_bbox,
            aoi_crs=aoi_crs,
            migration_path_id=migration_path_id,
            migration_buffer_meters=migration_buffer_meters,
        )
        cache_key = self._build_cache_key(
            scene_id=scene_id,
            scene_metadata=scene_metadata,
            assets=assets,
            bbox=effective_bbox,
            grid_block_size=block_size,
        )

        if persist:
            cached = self.cache_manager.load_derived_cache(
                cache_key=cache_key,
                include_grid=include_grid,
            )
            if cached is not None:
                return self._response_from_cached_derived(
                    cache_key=cache_key,
                    scene_id=scene_id,
                    cached=cached,
                    include_grid=include_grid,
                    migration_path_id=migration_path_id,
                    migration_buffer_meters=migration_buffer_meters,
                    persist=persist,
                )

        processing_assets = dict(assets)
        clipped_cache_hit = False
        if effective_bbox is not None and effective_bbox_crs is not None:
            optional_bands = tuple(
                band for band in OPTIONAL_SENTINEL_BANDS if band in processing_assets
            )
            cached_clipped = self.cache_manager.get_cached_clipped_assets(
                cache_key=cache_key,
                required_bands=REQUIRED_SENTINEL_BANDS,
                optional_bands=optional_bands,
            )
            if cached_clipped is None:
                processing_assets = self.cache_manager.cache_clipped_assets(
                    cache_key=cache_key,
                    source_assets=processing_assets,
                    bbox=effective_bbox,
                    aoi_crs=effective_bbox_crs,
                    required_bands=REQUIRED_SENTINEL_BANDS,
                    optional_bands=optional_bands,
                )
            else:
                processing_assets = cached_clipped
                clipped_cache_hit = True

        raster_bundle = load_and_align_bands(
            assets=processing_assets,
            required_bands=REQUIRED_SENTINEL_BANDS,
            aoi_bbox=None if effective_bbox is not None else aoi_bbox,
            aoi_crs=effective_bbox_crs or aoi_crs,
        )
        features = self._compute_features(raster_bundle)

        _, risk_normalized = score_risk(
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
            artifact_paths = self.cache_manager.save_derived_cache(
                cache_key=cache_key,
                scene_id=scene_id,
                chlorophyll=features["chlorophyll"],
                turbidity=features["turbidity"],
                ndwi=features["ndwi"],
                risk_normalized=risk_normalized,
                summary=summary,
                thresholds=self.settings.heatmap_thresholds,
                grid=grid,
            )
            artifact_paths["cache_key"] = cache_key
            if effective_bbox is not None:
                artifact_paths["clipped_cache_dir"] = str(self.cache_manager.clipped_dir / cache_key)
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
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "clip_cache_hit": clipped_cache_hit,
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
        water_mask = compute_water_mask_refined(
            ndwi=ndwi,
            b3=b3,
            b8=b8,
            threshold=self.settings.ndwi_water_threshold,
            nir_to_green_ratio_max=self.settings.water_nir_to_green_ratio_max,
        )
        temperature = temperature_proxy_stub(chlorophyll)

        return {
            "chlorophyll": chlorophyll,
            "turbidity": turbidity,
            "ndwi": ndwi,
            "water_mask": water_mask,
            "temperature_proxy": temperature,
        }

    def _response_from_cached_derived(
        self,
        cache_key: str,
        scene_id: str,
        cached: dict[str, Any],
        include_grid: bool,
        migration_path_id: str | None,
        migration_buffer_meters: float | None,
        persist: bool,
    ) -> dict[str, Any]:
        grid = cached["grid"] if include_grid else None
        summary = cached["summary"]

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
        artifact_paths = dict(cached["artifact_paths"])
        artifact_paths["cache_key"] = cache_key
        if persist:
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
                    "cache_key": cache_key,
                    "cache_hit": True,
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

    def _resolve_processing_bbox(
        self,
        scene_metadata: dict[str, Any] | None,
        aoi_bbox: list[float] | None,
        aoi_crs: str,
        migration_path_id: str | None,
        migration_buffer_meters: float | None,
    ) -> tuple[list[float] | None, str | None]:
        if aoi_bbox is not None:
            if len(aoi_bbox) != 4:
                raise ValueError("aoi_bbox must be [minx, miny, maxx, maxy].")
            return [float(v) for v in aoi_bbox], aoi_crs

        if migration_path_id:
            feature = self.migration_service.get_path_feature(migration_path_id)
            if feature is None:
                raise ValueError(f"Migration path not found: {migration_path_id}")
            geometry = shape(feature["geometry"])
            buffered = buffer_geometry_meters(
                geometry_wgs84=geometry,
                buffer_meters=(
                    migration_buffer_meters or self.settings.default_migration_buffer_meters
                ),
            )
            minx, miny, maxx, maxy = buffered.bounds
            return [float(minx), float(miny), float(maxx), float(maxy)], "EPSG:4326"

        if scene_metadata:
            bbox = scene_metadata.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(v) for v in bbox], "EPSG:4326"
        return None, None

    def _build_cache_key(
        self,
        scene_id: str,
        scene_metadata: dict[str, Any] | None,
        assets: dict[str, str],
        bbox: list[float] | None,
        grid_block_size: int,
    ) -> str:
        dataset = self.settings.cache_default_dataset
        if scene_metadata and scene_metadata.get("provider") == "sentinel":
            dataset = "sentinel2"

        date_source = None
        if scene_metadata:
            date_source = scene_metadata.get("acquired_date") or scene_metadata.get("created_at")
        if isinstance(date_source, str) and "T" in date_source:
            date_source = date_source.split("T")[0]
        if not date_source:
            date_source = datetime.now(UTC).date().isoformat()

        # Cache suffix bumps when grid/water detection logic changes.
        resolution_label = f"native-g{grid_block_size}-wf2"
        if bbox is None:
            stable_assets = json.dumps(
                {k: assets[k] for k in sorted(assets.keys())},
                sort_keys=True,
                separators=(",", ":"),
            )
            assets_hash = hashlib.sha1(stable_assets.encode("utf-8")).hexdigest()[:6]
            resolution_label = f"{resolution_label}-{assets_hash}"

        return self.cache_manager.build_cache_key(
            dataset=dataset,
            date_str=str(date_source),
            bbox=bbox,
            resolution=resolution_label,
        )
