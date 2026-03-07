from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_services
from app.models.schemas import ProcessSceneRequest, ProcessSceneResponse
from app.services.container import AppServices

router = APIRouter(tags=["processing"])


@router.post("/process/scene", response_model=ProcessSceneResponse)
def process_scene(
    request: ProcessSceneRequest,
    services: AppServices = Depends(get_services),
) -> ProcessSceneResponse:
    try:
        result = services.processing_service.process_scene_by_id(
            scene_id=request.scene_id,
            aoi_bbox=request.aoi_bbox,
            aoi_crs=request.aoi_crs,
            include_grid=request.include_grid,
            grid_block_size=request.grid_block_size,
            migration_path_id=request.migration_path_id,
            migration_buffer_meters=request.migration_buffer_meters,
            persist=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProcessSceneResponse(**result)

