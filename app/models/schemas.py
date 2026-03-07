from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProviderType(str, Enum):
    local = "local"
    sentinel = "sentinel"


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    readiness: dict[str, bool]


class LocalIngestPayload(BaseModel):
    scene_name: str | None = Field(default=None, description="Human-readable scene name.")
    scene_dir: str | None = Field(
        default=None, description="Directory containing scene bands (GeoTIFF files)."
    )
    assets: dict[str, str] | None = Field(
        default=None, description="Band key to local file path mapping."
    )

    @model_validator(mode="after")
    def validate_source(self) -> "LocalIngestPayload":
        if not self.scene_dir and not self.assets:
            raise ValueError("Provide either `scene_dir` or `assets` for local ingestion.")
        return self


class SentinelIngestPayload(BaseModel):
    collection: str = "sentinel-2-l2a"
    tile_id: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    bbox: list[float] | None = Field(default=None, description="[minx, miny, maxx, maxy]")
    cloud_cover_max: float | None = Field(default=20.0, ge=0.0, le=100.0)


class IngestSentinelRequest(BaseModel):
    provider: ProviderType = ProviderType.local
    local: LocalIngestPayload | None = None
    sentinel: SentinelIngestPayload | None = None

    @model_validator(mode="after")
    def validate_provider_payload(self) -> "IngestSentinelRequest":
        if self.provider == ProviderType.local and self.local is None:
            raise ValueError("`local` payload is required when provider='local'.")
        if self.provider == ProviderType.sentinel and self.sentinel is None:
            raise ValueError("`sentinel` payload is required when provider='sentinel'.")
        return self


class IngestSentinelResponse(BaseModel):
    scene_id: str
    provider: ProviderType
    scene_name: str
    assets: dict[str, str]
    discovered_bands: list[str]
    message: str


class ProcessSceneRequest(BaseModel):
    scene_id: str
    aoi_bbox: list[float] | None = Field(
        default=None,
        description="Optional AOI bbox [minx, miny, maxx, maxy] in `aoi_crs`.",
    )
    aoi_crs: str = "EPSG:4326"
    include_grid: bool = True
    grid_block_size: int | None = Field(default=None, ge=4, le=256)
    migration_path_id: str | None = None
    migration_buffer_meters: float | None = Field(default=None, ge=0.0)


class ProcessSceneResponse(BaseModel):
    processed_scene_id: str
    scene_id: str
    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    grid: dict[str, Any] | None = None
    path_summary: dict[str, Any] | None = None


class RiskScoreRequest(BaseModel):
    processed_scene_id: str | None = None
    scene_id: str | None = None
    assets: dict[str, str] | None = None
    migration_path_id: str | None = None
    grid_block_size: int | None = Field(default=None, ge=4, le=256)

    @model_validator(mode="after")
    def validate_source(self) -> "RiskScoreRequest":
        if not self.processed_scene_id and not self.scene_id and not self.assets:
            raise ValueError(
                "Provide one of `processed_scene_id`, `scene_id`, or `assets`."
            )
        return self


class RiskScoreResponse(BaseModel):
    scene_id: str
    processed_scene_id: str | None = None
    summary: dict[str, Any]
    grid: dict[str, Any] | None = None
    path_summary: dict[str, Any] | None = None


class RiskTilesResponse(BaseModel):
    scene_id: str
    processed_scene_id: str
    thresholds: dict[str, float]
    features: dict[str, Any]
    path_summary: dict[str, Any] | None = None


class MigrationPathItem(BaseModel):
    path_id: str
    name: str
    feature_type: str


class MigrationPathsResponse(BaseModel):
    count: int
    paths: list[MigrationPathItem]


class PrithviTrainRequest(BaseModel):
    dataset_path: str | None = None
    epochs: int = Field(default=1, ge=1, le=1000)
    dry_run: bool = True
    use_embeddings_only: bool = True
    notes: str | None = None


class PrithviTrainResponse(BaseModel):
    job_id: str
    status: str
    model_name: str
    message: str


class TrainStatusResponse(BaseModel):
    jobs: list[dict[str, Any]]
    count: int

