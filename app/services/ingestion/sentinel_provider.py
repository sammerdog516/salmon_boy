from __future__ import annotations

from app.models.schemas import IngestSentinelRequest
from app.services.ingestion.base import IngestionResult


class SentinelSceneProvider:
    def __init__(self, api_url: str, api_key: str | None = None) -> None:
        self.api_url = api_url
        self.api_key = api_key

    def ingest(self, request: IngestSentinelRequest) -> IngestionResult:
        # TODO: implement authenticated Sentinel provider query and asset discovery.
        # MVP intentionally keeps this stub so local ingestion remains the reliable demo path.
        raise NotImplementedError(
            "Remote Sentinel ingestion is scaffolded but not implemented in this MVP. "
            "Use provider='local' with local GeoTIFF assets."
        )

