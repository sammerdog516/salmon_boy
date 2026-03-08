from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services
from app.models.schemas import MigrationPathItem, MigrationPathsResponse
from app.services.container import AppServices

router = APIRouter(tags=["migration"])


@router.get("/migration-paths", response_model=MigrationPathsResponse)
def list_migration_paths(
    services: AppServices = Depends(get_services),
) -> MigrationPathsResponse:
    paths = [MigrationPathItem(**path) for path in services.migration_service.list_paths()]
    return MigrationPathsResponse(count=len(paths), paths=paths)


@router.get("/migration-paths/geojson")
def migration_paths_geojson(
    services: AppServices = Depends(get_services),
) -> dict[str, Any]:
    return services.migration_service.load_feature_collection()

