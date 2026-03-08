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
from app.services.processing.raster import load_and_align_bands
from app.services.processing.risk import normalize_risk, score_risk, summarize_risk
from app.services.processing.water_detector import detect_water_mask
from app.services.storage.cache_manager import CacheManager
from app.services.storage.metadata_store import MetadataStore
from app.utils.geospatial import buffer_geometry_meters


class ModelInferenceService:
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

    def predict_scene(
        self,
        scene_id: str,
        model_checkpoint: str | None = None,
        model_id: str | None = None,
        aoi_bbox: list[float] | None = None,
        aoi_crs: str = "EPSG:4326",
        include_grid: bool = True,
        grid_block_size: int | None = None,
        migration_path_id: str | None = None,
        migration_buffer_meters: float | None = None,
        inference_tile_size: int | None = None,
        inference_batch_size: int | None = None,
        device: str | None = None,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        scene = self.metadata_store.get_scene(scene_id)
        if scene is None:
            raise ValueError(f"Scene not found: {scene_id}")
        assets = scene.get("assets")
        if not isinstance(assets, dict):
            raise ValueError(f"Scene {scene_id} has no usable asset mapping.")

        # If source assets are gone (common in local/dev), serve latest cached prediction
        # for this scene so frontend demo does not hard-fail.
        missing_assets = [
            (str(band), str(path))
            for band, path in assets.items()
            if not Path(str(path)).exists()
        ]
        if missing_assets:
            if not force_recompute:
                cached_prediction = self._recover_latest_prediction_for_scene(
                    scene_id=scene_id,
                    include_grid=include_grid,
                    migration_path_id=migration_path_id,
                    migration_buffer_meters=migration_buffer_meters,
                )
                if cached_prediction is not None:
                    return cached_prediction
            missing_text = ", ".join(f"{band}={path}" for band, path in missing_assets[:4])
            raise FileNotFoundError(
                "Scene assets are missing on disk and no cached prediction is available. "
                f"Missing assets: {missing_text}. Re-ingest scene assets or restore files."
            )

        resolved_checkpoint = self._resolve_model_checkpoint(model_checkpoint)
        resolved_model_id = model_id or self._model_id_from_checkpoint(resolved_checkpoint)
        block_size = grid_block_size or self.settings.default_grid_block_size
        effective_bbox, effective_bbox_crs = self._resolve_processing_bbox(
            scene_metadata=scene,
            aoi_bbox=aoi_bbox,
            aoi_crs=aoi_crs,
            migration_path_id=migration_path_id,
            migration_buffer_meters=migration_buffer_meters,
        )
        cache_key = self._build_model_cache_key(
            scene_metadata=scene,
            assets={str(k): str(v) for k, v in assets.items()},
            bbox=effective_bbox,
            grid_block_size=block_size,
            model_id=resolved_model_id,
        )

        if not force_recompute:
            cached = self.cache_manager.load_prediction_cache(
                cache_key=cache_key,
                include_grid=include_grid,
            )
            if cached is not None:
                return self._prediction_from_cache(
                    scene_id=scene_id,
                    cache_key=cache_key,
                    cached=cached,
                    model_id=resolved_model_id,
                    migration_path_id=migration_path_id,
                    migration_buffer_meters=migration_buffer_meters,
                    include_grid=include_grid,
                    model_checkpoint=resolved_checkpoint,
                    clip_cache_hit=False,
                )

        processing_assets = {str(k): str(v) for k, v in assets.items()}
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
        b3 = raster_bundle.arrays["B3"]
        b4 = raster_bundle.arrays["B4"]
        b5 = raster_bundle.arrays["B5"]
        b8 = raster_bundle.arrays["B8"]
        stack = np.stack([b3, b4, b5, b8], axis=0).astype(np.float32)

        ndwi = ndwi_index(b3=b3, b8=b8)
        water_result = detect_water_mask(
            bundle=raster_bundle,
            ndwi=ndwi,
            b3=b3,
            b4=b4,
            b8=b8,
            threshold=self.settings.ndwi_water_threshold,
            nir_to_green_ratio_max=self.settings.water_nir_to_green_ratio_max,
            ndvi_max=self.settings.water_ndvi_max,
            mode=self.settings.water_detector_mode,
            pretrained_repo_id=self.settings.pretrained_water_model_repo_id,
            hf_token=self.settings.huggingface_token,
        )
        water_mask = water_result.mask
        chlorophyll = chlorophyll_index(b5=b5, b4=b4)
        turbidity = turbidity_index(b4=b4, b3=b3)
        _, rule_risk_norm = score_risk(
            chlorophyll=chlorophyll,
            turbidity=turbidity,
            water_mask=water_mask,
        )
        probability = self._predict_probability(
            stack=stack,
            checkpoint_path=resolved_checkpoint,
            tile_size=inference_tile_size or self.settings.default_inference_tile_size,
            batch_size=inference_batch_size or self.settings.default_inference_batch_size,
            device=device,
        )
        risk_normalized, fusion_meta = self._fuse_model_and_rule_risk(
            model_probability=probability,
            rule_risk_norm=rule_risk_norm,
            water_mask=water_mask,
        )
        risk_normalized[~water_mask] = np.nan

        summary = summarize_risk(
            risk=risk_normalized,
            chlorophyll=chlorophyll,
            turbidity=turbidity,
            water_mask=water_mask,
        )
        summary["model_id"] = resolved_model_id
        summary["ndwi_mean"] = float(np.nanmean(ndwi[water_mask])) if np.any(water_mask) else 0.0
        summary["water_detection_method"] = water_result.method
        summary["water_detection_details"] = dict(water_result.details)
        summary["model_probability_mean"] = fusion_meta["model_probability_mean"]
        summary["model_probability_std"] = fusion_meta["model_probability_std"]
        summary["risk_fusion_mode"] = fusion_meta["risk_fusion_mode"]

        grid = (
            aggregate_raster_to_grid_geojson(
                risk=risk_normalized,
                chlorophyll=chlorophyll,
                turbidity=turbidity,
                water_mask=water_mask,
                transform=raster_bundle.transform,
                crs=raster_bundle.crs,
                scene_id=scene_id,
                thresholds=self.settings.heatmap_thresholds,
                block_size=block_size,
                path_id=migration_path_id,
                min_water_fraction_for_risk=self.settings.min_tile_water_fraction_for_risk,
            )
            if include_grid
            else None
        )

        path_summary = None
        if grid and migration_path_id:
            path_feature = self.migration_service.get_path_feature(migration_path_id)
            if path_feature is None:
                raise ValueError(f"Migration path not found: {migration_path_id}")
            path_buffer = migration_buffer_meters or self.settings.default_migration_buffer_meters
            grid, path_summary = summarize_grid_near_paths(
                grid_feature_collection=grid,
                path_features=[path_feature],
                buffer_meters=path_buffer,
                selected_path_id=migration_path_id,
            )

        artifact_paths = self.cache_manager.save_prediction_cache(
            cache_key=cache_key,
            scene_id=scene_id,
            model_id=resolved_model_id,
            risk_probability=probability,
            risk_normalized=risk_normalized,
            water_mask=water_mask,
            summary=summary,
            thresholds=self.settings.heatmap_thresholds,
            grid=grid,
        )
        artifact_paths["cache_key"] = cache_key
        if effective_bbox is not None:
            artifact_paths["clipped_cache_dir"] = str(self.cache_manager.clipped_dir / cache_key)

        prediction_id = f"pred-{uuid4().hex[:12]}"
        self.metadata_store.save_prediction(
            prediction_id,
            {
                "prediction_id": prediction_id,
                "scene_id": scene_id,
                "model_id": resolved_model_id,
                "model_checkpoint": str(resolved_checkpoint),
                "created_at": datetime.now(UTC).isoformat(),
                "summary": summary,
                "artifact_paths": artifact_paths,
                "cache_key": cache_key,
                "cache_hit": False,
                "clip_cache_hit": clipped_cache_hit,
                "path_summary": path_summary,
            },
        )
        return {
            "prediction_id": prediction_id,
            "scene_id": scene_id,
            "model_id": resolved_model_id,
            "cache_key": cache_key,
            "summary": summary,
            "artifact_paths": artifact_paths,
            "grid": grid,
            "path_summary": path_summary,
            "cache_hit": False,
        }

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return self.metadata_store.get_prediction(prediction_id)

    def list_predictions(self, scene_id: str | None = None) -> list[dict[str, Any]]:
        predictions = self.metadata_store.list_predictions()
        if scene_id is None:
            return predictions
        return [item for item in predictions if str(item.get("scene_id")) == str(scene_id)]

    def load_prediction_grid(self, prediction_id: str) -> dict[str, Any]:
        record = self.get_prediction(prediction_id)
        if record is None:
            raise ValueError(f"Prediction not found: {prediction_id}")
        grid_path = record.get("artifact_paths", {}).get("grid_geojson")
        if not grid_path:
            raise ValueError(f"No grid artifact available for prediction: {prediction_id}")
        payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
        if payload.get("type") != "FeatureCollection":
            raise ValueError("Prediction grid artifact is not valid GeoJSON.")
        return payload

    def _prediction_from_cache(
        self,
        scene_id: str,
        cache_key: str,
        cached: dict[str, Any],
        model_id: str,
        migration_path_id: str | None,
        migration_buffer_meters: float | None,
        include_grid: bool,
        model_checkpoint: Path,
        clip_cache_hit: bool,
    ) -> dict[str, Any]:
        grid = cached["grid"] if include_grid else None
        path_summary = None
        if grid and migration_path_id:
            path_feature = self.migration_service.get_path_feature(migration_path_id)
            if path_feature is None:
                raise ValueError(f"Migration path not found: {migration_path_id}")
            path_buffer = migration_buffer_meters or self.settings.default_migration_buffer_meters
            grid, path_summary = summarize_grid_near_paths(
                grid_feature_collection=grid,
                path_features=[path_feature],
                buffer_meters=path_buffer,
                selected_path_id=migration_path_id,
            )

        prediction_id = f"pred-{uuid4().hex[:12]}"
        artifact_paths = dict(cached["artifact_paths"])
        artifact_paths["cache_key"] = cache_key
        self.metadata_store.save_prediction(
            prediction_id,
            {
                "prediction_id": prediction_id,
                "scene_id": scene_id,
                "model_id": model_id,
                "model_checkpoint": str(model_checkpoint),
                "created_at": datetime.now(UTC).isoformat(),
                "summary": cached["summary"],
                "artifact_paths": artifact_paths,
                "cache_key": cache_key,
                "cache_hit": True,
                "clip_cache_hit": clip_cache_hit,
                "path_summary": path_summary,
            },
        )
        return {
            "prediction_id": prediction_id,
            "scene_id": scene_id,
            "model_id": model_id,
            "cache_key": cache_key,
            "summary": cached["summary"],
            "artifact_paths": artifact_paths,
            "grid": grid,
            "path_summary": path_summary,
            "cache_hit": True,
        }

    def _recover_latest_prediction_for_scene(
        self,
        scene_id: str,
        include_grid: bool,
        migration_path_id: str | None,
        migration_buffer_meters: float | None,
    ) -> dict[str, Any] | None:
        predictions = self.list_predictions(scene_id=scene_id)
        if not predictions:
            return None

        for record in predictions:
            artifact_paths = record.get("artifact_paths", {})
            if not isinstance(artifact_paths, dict):
                continue
            grid_path = artifact_paths.get("grid_geojson")
            if include_grid and (not grid_path or not Path(str(grid_path)).exists()):
                continue

            grid = None
            if include_grid:
                grid = self.load_prediction_grid(str(record.get("prediction_id")))

            path_summary = None
            if grid and migration_path_id:
                path_feature = self.migration_service.get_path_feature(migration_path_id)
                if path_feature is None:
                    raise ValueError(f"Migration path not found: {migration_path_id}")
                path_buffer = migration_buffer_meters or self.settings.default_migration_buffer_meters
                grid, path_summary = summarize_grid_near_paths(
                    grid_feature_collection=grid,
                    path_features=[path_feature],
                    buffer_meters=path_buffer,
                    selected_path_id=migration_path_id,
                )

            return {
                "prediction_id": str(record.get("prediction_id")),
                "scene_id": str(record.get("scene_id") or scene_id),
                "model_id": str(record.get("model_id") or "cached"),
                "cache_key": str(record.get("cache_key") or ""),
                "summary": record.get("summary", {}),
                "artifact_paths": artifact_paths,
                "grid": grid,
                "path_summary": path_summary,
                "cache_hit": True,
            }
        return None

    def _resolve_model_checkpoint(self, value: str | None) -> Path:
        if value:
            candidate = Path(value)
            resolved = candidate if candidate.is_absolute() else self.settings.project_root / candidate
            resolved = resolved.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
            return resolved

        default_checkpoint = (
            self.settings.resolve_path(self.settings.model_artifacts_dir)
            / "weakrisk_baseline"
            / "best.pt"
        )
        if not default_checkpoint.exists():
            raise FileNotFoundError(
                "Default model checkpoint not found. Train first with scripts/train.py "
                "or pass --model-checkpoint / model_checkpoint explicitly."
            )
        return default_checkpoint

    def _model_id_from_checkpoint(self, checkpoint_path: Path) -> str:
        stem = checkpoint_path.stem
        parent = checkpoint_path.parent.name
        safe_parent = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in parent)
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        return f"{safe_parent}:{safe_stem}"

    def _build_model_cache_key(
        self,
        scene_metadata: dict[str, Any],
        assets: dict[str, str],
        bbox: list[float] | None,
        grid_block_size: int,
        model_id: str,
    ) -> str:
        dataset = self.settings.cache_default_dataset
        if scene_metadata.get("provider") == "sentinel":
            dataset = "sentinel2"
        date_source = scene_metadata.get("acquired_date") or scene_metadata.get("created_at")
        if isinstance(date_source, str) and "T" in date_source:
            date_source = date_source.split("T")[0]
        if not date_source:
            date_source = datetime.now(UTC).date().isoformat()

        model_hash = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
        # Include pipeline tag to avoid stale cache reuse across fusion/detection updates.
        water_detector_tag = "".join(
            ch for ch in str(self.settings.water_detector_mode).lower() if ch.isalnum()
        ) or "auto"
        resolution = f"native-g{grid_block_size}-m{model_hash}-wf4-{water_detector_tag}"
        if bbox is None:
            stable_assets = json.dumps(
                {k: assets[k] for k in sorted(assets.keys())},
                sort_keys=True,
                separators=(",", ":"),
            )
            assets_hash = hashlib.sha1(stable_assets.encode("utf-8")).hexdigest()[:6]
            resolution = f"{resolution}-{assets_hash}"
        return self.cache_manager.build_cache_key(
            dataset=dataset,
            date_str=str(date_source),
            bbox=bbox,
            resolution=resolution,
        )

    def _resolve_processing_bbox(
        self,
        scene_metadata: dict[str, Any],
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

        bbox = scene_metadata.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(v) for v in bbox], "EPSG:4326"
        return None, None

    def _predict_probability(
        self,
        stack: np.ndarray,
        checkpoint_path: Path,
        tile_size: int,
        batch_size: int,
        device: str | None,
    ) -> np.ndarray:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for model inference. Install with `python -m pip install torch`."
            ) from exc

        if stack.ndim != 3 or stack.shape[0] != 4:
            raise ValueError(f"Expected stack shape (4,H,W), got {stack.shape}")

        model = self._build_model(nn)
        runtime_device = self._resolve_device(torch=torch, raw_device=device)
        checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(runtime_device)
        model.eval()

        _, height, width = stack.shape
        sum_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)

        patches: list[np.ndarray] = []
        windows: list[tuple[int, int, int, int]] = []
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                row_end = min(height, row + tile_size)
                col_end = min(width, col + tile_size)
                h = row_end - row
                w = col_end - col
                patch = np.zeros((4, tile_size, tile_size), dtype=np.float32)
                patch[:, :h, :w] = stack[:, row:row_end, col:col_end]
                patches.append(patch)
                windows.append((row, row_end, col, col_end))

        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                end = min(len(patches), start + batch_size)
                batch = np.stack(patches[start:end], axis=0)
                x = torch.from_numpy(batch).to(runtime_device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0, :, :]
                for idx, prob in enumerate(probs):
                    row, row_end, col, col_end = windows[start + idx]
                    h = row_end - row
                    w = col_end - col
                    sum_map[row:row_end, col:col_end] += prob[:h, :w]
                    count_map[row:row_end, col:col_end] += 1.0

        probability = np.zeros((height, width), dtype=np.float32)
        np.divide(
            sum_map,
            np.maximum(count_map, 1.0),
            out=probability,
        )
        return np.clip(probability, 0.0, 1.0).astype(np.float32)

    def _fuse_model_and_rule_risk(
        self,
        model_probability: np.ndarray,
        rule_risk_norm: np.ndarray,
        water_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        fused = np.full(model_probability.shape, np.nan, dtype=np.float32)
        valid = water_mask & np.isfinite(model_probability)
        if not np.any(valid):
            return fused, {
                "risk_fusion_mode": "no_valid_water_pixels",
                "model_probability_mean": 0.0,
                "model_probability_std": 0.0,
            }

        values = model_probability[valid]
        model_mean = float(np.nanmean(values))
        model_std = float(np.nanstd(values))
        model_norm = normalize_risk(model_probability, valid)

        # Weak labels can produce low-contrast model outputs; if spread is too small,
        # prefer rule-based risk so frontend does not flatten into one color band.
        if model_std < 0.05:
            fused[valid] = rule_risk_norm[valid]
            mode = "rule_fallback_low_model_variance"
        else:
            blended = 0.70 * model_norm[valid] + 0.30 * rule_risk_norm[valid]
            fused[valid] = np.clip(blended, 0.0, 1.0)
            mode = "model_rule_blend"

        return fused.astype(np.float32), {
            "risk_fusion_mode": mode,
            "model_probability_mean": model_mean,
            "model_probability_std": model_std,
        }

    def _build_model(self, nn: Any) -> Any:
        class _TinySegNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(4, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=1),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _TinySegNet()

    def _resolve_device(self, torch: Any, raw_device: str | None) -> Any:
        if raw_device and raw_device != "auto":
            return torch.device(raw_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
