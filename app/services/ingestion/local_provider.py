from __future__ import annotations

from pathlib import Path

from app.core.constants import OPTIONAL_SENTINEL_BANDS, REQUIRED_SENTINEL_BANDS
from app.models.schemas import IngestSentinelRequest
from app.services.ingestion.base import IngestionResult
from app.utils.bands import normalize_band_name


class LocalSceneProvider:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def ingest(self, request: IngestSentinelRequest) -> IngestionResult:
        if request.local is None:
            raise ValueError("Missing `local` payload for local ingestion.")

        local_payload = request.local
        assets: dict[str, str]
        if local_payload.assets:
            assets = self._normalize_assets(local_payload.assets)
        else:
            scene_dir = self._resolve_path(local_payload.scene_dir or "")
            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
            assets = self._discover_assets(scene_dir)

        self._validate_required_bands(assets)
        scene_name = local_payload.scene_name or "local-scene"
        discovered_bands = sorted(assets.keys())
        return IngestionResult(
            scene_name=scene_name,
            assets=assets,
            discovered_bands=discovered_bands,
            provider_message="Local scene ingested successfully.",
        )

    def _resolve_path(self, path_str: str) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def _normalize_assets(self, assets: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_band, raw_path in assets.items():
            band = normalize_band_name(raw_band)
            path = self._resolve_path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Band asset not found for {band}: {path}")
            normalized[band] = str(path)
        return normalized

    def _discover_assets(self, scene_dir: Path) -> dict[str, str]:
        files = list(scene_dir.glob("*.tif")) + list(scene_dir.glob("*.tiff"))
        if not files:
            raise ValueError(f"No GeoTIFF assets found in scene_dir: {scene_dir}")

        discovered: dict[str, str] = {}
        for file_path in files:
            upper_name = file_path.stem.upper()
            for token in REQUIRED_SENTINEL_BANDS + OPTIONAL_SENTINEL_BANDS:
                if token in upper_name or token.replace("B", "B0") in upper_name:
                    discovered[token] = str(file_path.resolve())
        if not discovered:
            raise ValueError(
                "Unable to infer Sentinel band assets from file names. "
                "Provide explicit `local.assets` mapping instead."
            )
        return discovered

    def _validate_required_bands(self, assets: dict[str, str]) -> None:
        missing = [band for band in REQUIRED_SENTINEL_BANDS if band not in assets]
        if missing:
            raise ValueError(
                "Missing required bands for processing: "
                + ", ".join(missing)
                + ". Required: "
                + ", ".join(REQUIRED_SENTINEL_BANDS)
            )

