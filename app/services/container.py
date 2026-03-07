from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.services.ingestion.local_provider import LocalSceneProvider
from app.services.ingestion.sentinel_provider import SentinelSceneProvider
from app.services.ingestion.service import IngestionService
from app.services.migration.loader import MigrationPathService
from app.services.processing.service import ProcessingService
from app.services.storage.metadata_store import MetadataStore
from app.services.training.prithvi import PrithviTrainingService


@dataclass
class AppServices:
    settings: Settings
    metadata_store: MetadataStore
    migration_service: MigrationPathService
    ingestion_service: IngestionService
    processing_service: ProcessingService
    training_service: PrithviTrainingService


def build_services(settings: Settings) -> AppServices:
    metadata_store = MetadataStore(
        scene_registry_path=settings.resolve_path(settings.scene_registry_path),
        processed_registry_path=settings.resolve_path(settings.processed_registry_path),
        training_registry_path=settings.resolve_path(settings.training_registry_path),
    )
    migration_service = MigrationPathService(
        migration_geojson_path=settings.resolve_path(settings.migration_paths_file)
    )
    ingestion_service = IngestionService(
        metadata_store=metadata_store,
        local_provider=LocalSceneProvider(project_root=settings.project_root),
        sentinel_provider=SentinelSceneProvider(
            api_url=settings.sentinel_api_url, api_key=settings.sentinel_api_key
        ),
    )
    processing_service = ProcessingService(
        settings=settings,
        metadata_store=metadata_store,
        migration_service=migration_service,
    )
    training_service = PrithviTrainingService(
        settings=settings,
        metadata_store=metadata_store,
    )
    return AppServices(
        settings=settings,
        metadata_store=metadata_store,
        migration_service=migration_service,
        ingestion_service=ingestion_service,
        processing_service=processing_service,
        training_service=training_service,
    )

