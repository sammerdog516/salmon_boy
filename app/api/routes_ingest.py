from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_services
from app.models.schemas import IngestSentinelRequest, IngestSentinelResponse
from app.services.container import AppServices

router = APIRouter(tags=["ingestion"])


@router.post("/ingest/sentinel", response_model=IngestSentinelResponse)
def ingest_sentinel_scene(
    request: IngestSentinelRequest,
    services: AppServices = Depends(get_services),
) -> IngestSentinelResponse:
    try:
        response = services.ingestion_service.ingest_scene(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestSentinelResponse(**response)

