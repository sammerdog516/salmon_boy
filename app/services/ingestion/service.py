from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.models.schemas import IngestSentinelRequest, ProviderType
from app.services.ingestion.local_provider import LocalSceneProvider
from app.services.ingestion.sentinel_provider import SentinelSceneProvider
from app.services.storage.metadata_store import MetadataStore


class IngestionService:
    def __init__(
        self,
        metadata_store: MetadataStore,
        local_provider: LocalSceneProvider,
        sentinel_provider: SentinelSceneProvider,
    ) -> None:
        self.metadata_store = metadata_store
        self.local_provider = local_provider
        self.sentinel_provider = sentinel_provider

    def ingest_scene(self, request: IngestSentinelRequest) -> dict[str, object]:
        if request.provider == ProviderType.local:
            result = self.local_provider.ingest(request)
        elif request.provider == ProviderType.sentinel:
            result = self.sentinel_provider.ingest(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

        scene_id = f"scene-{uuid4().hex[:12]}"
        payload = {
            "scene_id": scene_id,
            "scene_name": result.scene_name,
            "provider": request.provider.value,
            "assets": result.assets,
            "bands": result.discovered_bands,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "ingested",
        }
        self.metadata_store.save_scene(scene_id, payload)
        return {
            "scene_id": scene_id,
            "provider": request.provider,
            "scene_name": result.scene_name,
            "assets": result.assets,
            "discovered_bands": result.discovered_bands,
            "message": result.provider_message,
        }

