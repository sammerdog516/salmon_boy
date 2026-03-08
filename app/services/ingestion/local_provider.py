from __future__ import annotations

from pathlib import Path

import numpy as np

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
        generated_sample = False
        enriched_sample = False
        assets: dict[str, str]
        if local_payload.assets:
            assets = self._normalize_assets(local_payload.assets)
        else:
            raw_scene_dir = local_payload.scene_dir or ""
            scene_dir = self._resolve_path(raw_scene_dir)
            if not scene_dir.exists():
                if self._is_sample_scene_path(raw_scene_dir):
                    self._bootstrap_sample_scene(scene_dir)
                    generated_sample = True
                else:
                    raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
            try:
                assets = self._discover_assets(scene_dir)
            except ValueError as exc:
                if self._is_sample_scene_path(raw_scene_dir):
                    self._bootstrap_sample_scene(scene_dir)
                    generated_sample = True
                    assets = self._discover_assets(scene_dir)
                else:
                    raise exc
            if self._is_sample_scene_path(raw_scene_dir):
                missing_required = [b for b in REQUIRED_SENTINEL_BANDS if b not in assets]
                if missing_required:
                    self._bootstrap_sample_scene(scene_dir)
                    generated_sample = True
                    assets = self._discover_assets(scene_dir)
                assets, enriched_sample = self._ensure_sample_optional_bands(scene_dir, assets)

        self._validate_required_bands(assets)
        scene_name = local_payload.scene_name or "local-scene"
        discovered_bands = sorted(assets.keys())
        provider_message = "Local scene ingested successfully."
        if generated_sample:
            provider_message = (
                "Local sample scene was missing and has been auto-generated, then ingested."
            )
        elif enriched_sample:
            provider_message = (
                "Local sample scene ingested and supplemented with optional bands "
                "(B2/B11/B12) for pretrained water detection."
            )
        return IngestionResult(
            scene_name=scene_name,
            assets=assets,
            discovered_bands=discovered_bands,
            provider_message=provider_message,
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

    def _is_sample_scene_path(self, raw_path: str) -> bool:
        normalized = raw_path.replace("\\", "/").strip().lower()
        if not normalized:
            return False
        return (
            normalized == "data/sample"
            or normalized.endswith("/data/sample")
            or normalized == "sample"
            or normalized.endswith("/sample")
        )

    def _bootstrap_sample_scene(self, scene_dir: Path) -> None:
        scene_dir.mkdir(parents=True, exist_ok=True)
        seed = np.random.default_rng(42)
        height = 512
        width = 512
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]

        # Synthetic river + estuary pattern with clear land/water contrast.
        river = np.exp(-((y + 0.25 * x) ** 2) / 0.08).astype(np.float32)
        estuary = np.exp(-(((x + 0.35) ** 2) + ((y - 0.10) ** 2)) / 0.12).astype(np.float32)
        water_score = np.clip(0.65 * river + 0.75 * estuary, 0.0, 1.0)
        water = water_score > 0.38

        noise = lambda scale: seed.normal(0.0, scale, size=(height, width)).astype(np.float32)

        b3 = np.where(
            water,
            0.20 + 0.12 * water_score + noise(0.004),
            0.10 + 0.03 * (1.0 - y) + noise(0.004),
        )
        b4 = np.where(
            water,
            0.08 + 0.06 * water_score + noise(0.004),
            0.15 + 0.04 * x + noise(0.004),
        )
        b5 = np.where(
            water,
            0.12 + 0.08 * water_score + noise(0.004),
            0.18 + 0.05 * (0.5 * x - 0.2 * y) + noise(0.004),
        )
        b8 = np.where(
            water,
            0.06 + 0.05 * (1.0 - water_score) + noise(0.004),
            0.30 + 0.10 * (0.5 * x + 0.5) + noise(0.004),
        )

        b3 = np.clip(b3, 0.001, 1.0).astype(np.float32)
        b4 = np.clip(b4, 0.001, 1.0).astype(np.float32)
        b5 = np.clip(b5, 0.001, 1.0).astype(np.float32)
        b8 = np.clip(b8, 0.001, 1.0).astype(np.float32)

        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError as exc:
            raise RuntimeError(
                "rasterio is required to auto-generate sample scene files."
            ) from exc

        transform = from_origin(-123.2, 49.4, 0.0005, 0.0005)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": None,
        }
        for band_name, arr in (("B3", b3), ("B4", b4), ("B5", b5), ("B8", b8)):
            destination = scene_dir / f"{band_name}.tif"
            with rasterio.open(destination, "w", **profile) as dst:
                dst.write(arr, 1)

    def _ensure_sample_optional_bands(
        self,
        scene_dir: Path,
        assets: dict[str, str],
    ) -> tuple[dict[str, str], bool]:
        missing_optional = [band for band in ("B2", "B11", "B12") if band not in assets]
        if not missing_optional:
            return assets, False
        if not all(band in assets for band in ("B3", "B4", "B8")):
            return assets, False

        try:
            import rasterio
        except ImportError:
            return assets, False

        b3_path = Path(assets["B3"])
        b4_path = Path(assets["B4"])
        b8_path = Path(assets["B8"])

        with rasterio.open(b3_path) as src_b3, rasterio.open(b4_path) as src_b4, rasterio.open(
            b8_path
        ) as src_b8:
            b3 = src_b3.read(1).astype(np.float32)
            b4 = src_b4.read(1).astype(np.float32)
            b8 = src_b8.read(1).astype(np.float32)
            profile = src_b3.profile.copy()

        # Lightweight synthetic approximations so pretrained water detector can run
        # in demo mode when only B3/B4/B8 are originally available.
        b2 = np.clip(1.10 * b3 - 0.02 * b4, 0.001, 1.0).astype(np.float32)
        b11 = np.clip(0.55 * b8 + 0.20 * b4, 0.001, 1.0).astype(np.float32)
        b12 = np.clip(0.45 * b8 + 0.25 * b4, 0.001, 1.0).astype(np.float32)

        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        band_data = {"B2": b2, "B11": b11, "B12": b12}
        for band in missing_optional:
            destination = scene_dir / f"{band}.tif"
            with rasterio.open(destination, "w", **profile) as dst:
                dst.write(band_data[band], 1)

        return self._discover_assets(scene_dir), True

