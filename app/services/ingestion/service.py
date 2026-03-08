from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.models.schemas import IngestSentinelRequest, ProviderType
from app.services.ingestion.local_provider import LocalSceneProvider
from app.services.ingestion.sentinel_provider import SentinelSceneProvider
from app.services.storage.cache_manager import CacheManager
from app.services.storage.metadata_store import MetadataStore


class IngestionService:
    def __init__(
        self,
        metadata_store: MetadataStore,
        cache_manager: CacheManager,
        local_provider: LocalSceneProvider,
        sentinel_provider: SentinelSceneProvider,
    ) -> None:
        self.metadata_store = metadata_store
        self.cache_manager = cache_manager
        self.local_provider = local_provider
        self.sentinel_provider = sentinel_provider

    def ingest_scene(self, request: IngestSentinelRequest) -> dict[str, object]:
        request_payload = request.model_dump(mode="json", exclude_none=True)
        request_hash = self.cache_manager.compute_request_hash(request_payload)
        metadata_hit = self.cache_manager.get_metadata_entry(request_hash)
        if metadata_hit:
            cached_scene_id = metadata_hit.get("scene_id")
            if isinstance(cached_scene_id, str):
                scene = self.metadata_store.get_scene(cached_scene_id)
                if scene:
                    return {
                        "scene_id": scene["scene_id"],
                        "provider": ProviderType(scene["provider"]),
                        "scene_name": scene.get("scene_name", scene["scene_id"]),
                        "assets": scene.get("assets", {}),
                        "discovered_bands": scene.get("bands", []),
                        "message": "Scene metadata cache hit.",
                    }

        if request.provider == ProviderType.local:
            result = self.local_provider.ingest(request)
        elif request.provider == ProviderType.sentinel:
            result = self.sentinel_provider.ingest(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

        scene_id = f"scene-{uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        bbox = request.sentinel.bbox if request.sentinel else None
        acquired_date = None
        cloud_cover = None
        if request.sentinel:
            acquired_date = request.sentinel.date_start or request.sentinel.date_end
            cloud_cover = request.sentinel.cloud_cover_max

        payload = {
            "scene_id": scene_id,
            "scene_name": result.scene_name,
            "provider": request.provider.value,
            "assets": result.assets,
            "bands": result.discovered_bands,
            "bbox": bbox,
            "acquired_date": acquired_date,
            "cloud_cover": cloud_cover,
            "request_hash": request_hash,
            "source_urls": list(result.assets.values()),
            "created_at": now,
            "status": "ingested",
        }
        self.metadata_store.save_scene(scene_id, payload)
        self.cache_manager.save_metadata_entry(
            request_hash=request_hash,
            entry={
                "scene_id": scene_id,
                "bbox": bbox,
                "date": acquired_date,
                "cloud_cover": cloud_cover,
                "source_urls": list(result.assets.values()),
                "provider": request.provider.value,
                "dataset": "sentinel2",
                "request_hash": request_hash,
            },
        )
        return {
            "scene_id": scene_id,
            "provider": request.provider,
            "scene_name": result.scene_name,
            "assets": result.assets,
            "discovered_bands": result.discovered_bands,
            "message": result.provider_message,
        }
