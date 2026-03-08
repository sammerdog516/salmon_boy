from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Salmon Water Risk Backend"
    app_version: str = "0.1.0"
    environment: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000

    project_root: Path = Field(default_factory=_default_project_root)
    artifacts_dir: Path = Path("artifacts")
    cache_dir: Path = Path("artifacts/cache")
    migration_paths_file: Path = Path("data/migration_paths/salmon_paths.geojson")

    scene_registry_path: Path = Path("artifacts/registry/scenes.json")
    processed_registry_path: Path = Path("artifacts/registry/processed_scenes.json")
    training_registry_path: Path = Path("artifacts/registry/training_jobs.json")

    ndwi_water_threshold: float = 0.0
    heatmap_yellow_threshold: float = 0.30
    heatmap_red_threshold: float = 0.65
    heatmap_infrared_threshold: float = 0.85
    default_grid_block_size: int = 32
    default_migration_buffer_meters: float = 250.0
    cache_max_size_gb: float = 10.0
    clipped_cache_max_dimension: int = 2048
    cache_default_dataset: str = "sentinel2"

    default_aoi_bbox: str | None = None

    sentinel_api_key: str | None = None
    sentinel_api_url: str = "https://example-sentinel-provider.local"

    prithvi_model_name: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
    prithvi_enabled: bool = False

    log_level: str = "INFO"

    def resolve_path(self, value: Path | str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def ensure_directories(self) -> None:
        cache_root = self.resolve_path(self.cache_dir)
        directories = {
            self.resolve_path(self.artifacts_dir),
            cache_root,
            cache_root / "metadata",
            cache_root / "clipped",
            cache_root / "derived",
            cache_root / "tiles",
            self.resolve_path(self.scene_registry_path).parent,
            self.resolve_path(self.processed_registry_path).parent,
            self.resolve_path(self.training_registry_path).parent,
            self.resolve_path(self.migration_paths_file).parent,
        }
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def heatmap_thresholds(self) -> dict[str, float]:
        return {
            "blue": 0.0,
            "yellow": self.heatmap_yellow_threshold,
            "red": self.heatmap_red_threshold,
            "infrared": self.heatmap_infrared_threshold,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "artifacts_dir": str(self.resolve_path(self.artifacts_dir)),
            "migration_paths_file": str(self.resolve_path(self.migration_paths_file)),
        }


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
