from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.schemas import IngestSentinelRequest


@dataclass
class IngestionResult:
    scene_name: str
    assets: dict[str, str]
    discovered_bands: list[str]
    provider_message: str


class SceneIngestionProvider(Protocol):
    def ingest(self, request: IngestSentinelRequest) -> IngestionResult:
        ...

