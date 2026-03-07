from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.api.dependencies import get_services
from app.models.schemas import (
    PrithviTrainRequest,
    PrithviTrainResponse,
    TrainStatusResponse,
)
from app.services.container import AppServices

router = APIRouter(tags=["training"])


@router.post("/train/prithvi", response_model=PrithviTrainResponse)
def train_prithvi(
    request: PrithviTrainRequest,
    background_tasks: BackgroundTasks,
    services: AppServices = Depends(get_services),
) -> PrithviTrainResponse:
    try:
        response = services.training_service.start_training_job(request)
        if not request.dry_run:
            background_tasks.add_task(services.training_service.mark_completed, response["job_id"])
        return PrithviTrainResponse(**response)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/train/status", response_model=TrainStatusResponse)
def training_status(
    job_id: str | None = Query(default=None),
    services: AppServices = Depends(get_services),
) -> TrainStatusResponse:
    jobs = services.training_service.get_status(job_id=job_id)
    return TrainStatusResponse(jobs=jobs, count=len(jobs))

