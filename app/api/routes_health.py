from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services
from app.models.schemas import HealthResponse
from app.services.container import AppServices

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(services: AppServices = Depends(get_services)) -> HealthResponse:
    settings = services.settings
    artifacts_dir = settings.resolve_path(settings.artifacts_dir)
    cache_root = settings.resolve_path(settings.cache_dir)
    migration_path = settings.resolve_path(settings.migration_paths_file)

    readiness = {
        "artifacts_dir": artifacts_dir.exists(),
        "cache_root": cache_root.exists(),
        "cache_metadata_dir": (cache_root / "metadata").exists(),
        "cache_clipped_dir": (cache_root / "clipped").exists(),
        "cache_derived_dir": (cache_root / "derived").exists(),
        "cache_tiles_dir": (cache_root / "tiles").exists(),
        "model_artifacts_dir": settings.resolve_path(settings.model_artifacts_dir).exists(),
        "migration_paths_file": migration_path.exists(),
        "scene_registry": settings.resolve_path(settings.scene_registry_path).exists(),
        "processed_registry": settings.resolve_path(settings.processed_registry_path).exists(),
        "prediction_registry": settings.resolve_path(settings.prediction_registry_path).exists(),
    }
    status = "ok" if all(readiness.values()) else "degraded"
    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        readiness=readiness,
    )
