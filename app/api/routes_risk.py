from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_services
from app.models.schemas import (
    RiskPredictRequest,
    RiskPredictResponse,
    RiskScoreRequest,
    RiskScoreResponse,
    RiskTilesResponse,
)
from app.services.container import AppServices
from app.services.migration.summarizer import summarize_grid_near_paths
from app.utils.bands import normalize_band_name

router = APIRouter(tags=["risk"])


def _resolve_assets(raw_assets: dict[str, str], project_root: Path) -> dict[str, str]:
    assets: dict[str, str] = {}
    for band_raw, path_raw in raw_assets.items():
        band = normalize_band_name(band_raw)
        candidate = Path(path_raw)
        resolved = candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Asset path not found for {band}: {resolved}")
        assets[band] = str(resolved)
    return assets


@router.post("/risk/score", response_model=RiskScoreResponse)
def score_risk(
    request: RiskScoreRequest,
    services: AppServices = Depends(get_services),
) -> RiskScoreResponse:
    try:
        if request.processed_scene_id:
            record = services.processing_service.get_processed_scene(request.processed_scene_id)
            if record is None:
                raise ValueError(f"Processed scene not found: {request.processed_scene_id}")
            grid = None
            path_summary = None
            if request.migration_path_id:
                grid = services.processing_service.load_grid_artifact(request.processed_scene_id)
                path_feature = services.migration_service.get_path_feature(request.migration_path_id)
                if path_feature is None:
                    raise ValueError(f"Migration path not found: {request.migration_path_id}")
                grid, path_summary = summarize_grid_near_paths(
                    grid_feature_collection=grid,
                    path_features=[path_feature],
                    buffer_meters=services.settings.default_migration_buffer_meters,
                    selected_path_id=request.migration_path_id,
                )
            return RiskScoreResponse(
                scene_id=record["scene_id"],
                processed_scene_id=request.processed_scene_id,
                summary=record.get("summary", {}),
                grid=grid,
                path_summary=path_summary,
            )

        if request.scene_id:
            result = services.processing_service.process_scene_by_id(
                scene_id=request.scene_id,
                include_grid=True,
                grid_block_size=request.grid_block_size,
                migration_path_id=request.migration_path_id,
                persist=True,
            )
            return RiskScoreResponse(
                scene_id=result["scene_id"],
                processed_scene_id=result["processed_scene_id"],
                summary=result["summary"],
                grid=result["grid"],
                path_summary=result["path_summary"],
            )

        assets = _resolve_assets(request.assets or {}, services.settings.project_root)
        ad_hoc_scene_id = f"adhoc-{uuid4().hex[:10]}"
        result = services.processing_service.process_assets(
            scene_id=ad_hoc_scene_id,
            assets=assets,
            include_grid=True,
            grid_block_size=request.grid_block_size,
            migration_path_id=request.migration_path_id,
            persist=True,
        )
        return RiskScoreResponse(
            scene_id=result["scene_id"],
            processed_scene_id=result["processed_scene_id"],
            summary=result["summary"],
            grid=result["grid"],
            path_summary=result["path_summary"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/risk/predict", response_model=RiskPredictResponse)
def predict_risk(
    request: RiskPredictRequest,
    services: AppServices = Depends(get_services),
) -> RiskPredictResponse:
    try:
        result = services.inference_service.predict_scene(
            scene_id=request.scene_id,
            model_checkpoint=request.model_checkpoint,
            model_id=request.model_id,
            aoi_bbox=request.aoi_bbox,
            aoi_crs=request.aoi_crs,
            include_grid=request.include_grid,
            grid_block_size=request.grid_block_size,
            migration_path_id=request.migration_path_id,
            migration_buffer_meters=request.migration_buffer_meters,
            inference_tile_size=request.inference_tile_size,
            inference_batch_size=request.inference_batch_size,
            device=request.device,
            force_recompute=request.force_recompute,
        )
        return RiskPredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/risk/tiles", response_model=RiskTilesResponse)
def risk_tiles(
    processed_scene_id: str | None = Query(default=None),
    prediction_id: str | None = Query(default=None),
    source: str = Query(default="rule", pattern="^(rule|model)$"),
    scene_id: str | None = Query(default=None),
    migration_path_id: str | None = Query(default=None),
    buffer_meters: float | None = Query(default=None, ge=0.0),
    services: AppServices = Depends(get_services),
) -> RiskTilesResponse:
    try:
        if source == "model" or prediction_id is not None:
            selected_prediction_id = prediction_id
            if selected_prediction_id is None:
                predictions = services.inference_service.list_predictions(scene_id=scene_id)
                if not predictions:
                    raise ValueError(
                        "No model predictions found. Run /risk/predict first or provide prediction_id."
                    )
                selected_prediction_id = str(predictions[0]["prediction_id"])

            record = services.inference_service.get_prediction(selected_prediction_id)
            if record is None:
                raise ValueError(f"Prediction not found: {selected_prediction_id}")
            grid = services.inference_service.load_prediction_grid(selected_prediction_id)
            path_summary = None
            if migration_path_id:
                path_feature = services.migration_service.get_path_feature(migration_path_id)
                if path_feature is None:
                    raise ValueError(f"Migration path not found: {migration_path_id}")
                grid, path_summary = summarize_grid_near_paths(
                    grid_feature_collection=grid,
                    path_features=[path_feature],
                    buffer_meters=buffer_meters
                    or services.settings.default_migration_buffer_meters,
                    selected_path_id=migration_path_id,
                )
            return RiskTilesResponse(
                scene_id=record["scene_id"],
                processed_scene_id=None,
                prediction_id=selected_prediction_id,
                model_id=record.get("model_id"),
                source="model",
                thresholds=services.settings.heatmap_thresholds,
                features=grid,
                path_summary=path_summary,
            )

        selected_processed_id = processed_scene_id
        if selected_processed_id is None:
            processed = services.processing_service.list_processed_scenes()
            if not processed:
                raise ValueError("No processed scenes found. Run /process/scene first.")
            selected_processed_id = str(processed[0]["processed_scene_id"])

        record = services.processing_service.get_processed_scene(selected_processed_id)
        if record is None:
            raise ValueError(f"Processed scene not found: {selected_processed_id}")

        grid = services.processing_service.load_grid_artifact(selected_processed_id)
        path_summary = None
        if migration_path_id:
            path_feature = services.migration_service.get_path_feature(migration_path_id)
            if path_feature is None:
                raise ValueError(f"Migration path not found: {migration_path_id}")
            grid, path_summary = summarize_grid_near_paths(
                grid_feature_collection=grid,
                path_features=[path_feature],
                buffer_meters=buffer_meters or services.settings.default_migration_buffer_meters,
                selected_path_id=migration_path_id,
            )

        return RiskTilesResponse(
            scene_id=record["scene_id"],
            processed_scene_id=selected_processed_id,
            prediction_id=None,
            model_id=None,
            source="rule",
            thresholds=services.settings.heatmap_thresholds,
            features=grid,
            path_summary=path_summary,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

