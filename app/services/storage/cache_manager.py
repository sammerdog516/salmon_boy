from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling
from rasterio.windows import Window, from_bounds

from app.core.config import Settings
from app.utils.geospatial import transform_bounds


class CacheManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_root = settings.resolve_path(settings.cache_dir)
        self.metadata_dir = self.cache_root / "metadata"
        self.clipped_dir = self.cache_root / "clipped"
        self.derived_dir = self.cache_root / "derived"
        self.tiles_dir = self.cache_root / "tiles"

    def compute_request_hash(self, payload: dict[str, Any]) -> str:
        stable_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(stable_json.encode("utf-8")).hexdigest()

    def build_cache_key(
        self,
        dataset: str,
        date_str: str | None,
        bbox: Iterable[float] | None,
        resolution: str,
    ) -> str:
        bbox_hash = self._bbox_hash(bbox)
        normalized_date = (date_str or "unknown-date").replace(":", "-").replace("/", "-")
        safe_resolution = resolution.replace(".", "p").replace(" ", "")
        return f"{dataset}_{normalized_date}_{bbox_hash}_{safe_resolution}"

    def get_metadata_entry(self, request_hash: str) -> dict[str, Any] | None:
        path = self.metadata_dir / f"{request_hash}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_metadata_entry(self, request_hash: str, entry: dict[str, Any]) -> Path:
        path = self.metadata_dir / f"{request_hash}.json"
        payload = dict(entry)
        payload["request_hash"] = request_hash
        payload["updated_at"] = datetime.now(UTC).isoformat()
        if "created_at" not in payload:
            payload["created_at"] = payload["updated_at"]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def get_cached_clipped_assets(
        self,
        cache_key: str,
        required_bands: tuple[str, ...],
        optional_bands: tuple[str, ...] = (),
    ) -> dict[str, str] | None:
        key_dir = self.clipped_dir / cache_key
        if not key_dir.exists():
            return None

        asset_paths: dict[str, str] = {}
        for band in required_bands:
            path = key_dir / f"{band}.tif"
            if not path.exists():
                return None
            asset_paths[band] = str(path)
        for band in optional_bands:
            path = key_dir / f"{band}.tif"
            if path.exists():
                asset_paths[band] = str(path)
        return asset_paths

    def cache_clipped_assets(
        self,
        cache_key: str,
        source_assets: dict[str, str],
        bbox: list[float],
        aoi_crs: str,
        required_bands: tuple[str, ...],
        optional_bands: tuple[str, ...] = (),
    ) -> dict[str, str]:
        key_dir = self.clipped_dir / cache_key
        key_dir.mkdir(parents=True, exist_ok=True)

        selected = [band for band in required_bands if band in source_assets]
        selected.extend([band for band in optional_bands if band in source_assets])
        for band in selected:
            destination = key_dir / f"{band}.tif"
            if destination.exists():
                continue
            self._clip_and_store_band(
                source_path=Path(source_assets[band]),
                destination=destination,
                bbox=bbox,
                aoi_crs=aoi_crs,
                max_dimension=self.settings.clipped_cache_max_dimension,
            )

        self.enforce_cache_size()
        cached_assets = self.get_cached_clipped_assets(
            cache_key=cache_key,
            required_bands=required_bands,
            optional_bands=optional_bands,
        )
        if cached_assets is None:
            raise ValueError(f"Failed to create clipped cache assets for key: {cache_key}")
        return cached_assets

    def load_derived_cache(
        self,
        cache_key: str,
        include_grid: bool,
    ) -> dict[str, Any] | None:
        npz_path = self.derived_dir / f"{cache_key}.npz"
        summary_path = self.derived_dir / f"{cache_key}.summary.json"
        grid_path = self.tiles_dir / f"{cache_key}.geojson"
        if not npz_path.exists() or not summary_path.exists():
            return None
        if include_grid and not grid_path.exists():
            return None

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        grid_payload = None
        if include_grid and grid_path.exists():
            grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))

        artifacts = {
            "derived_npz": str(npz_path),
            "summary_json": str(summary_path),
        }
        if grid_path.exists():
            artifacts["grid_geojson"] = str(grid_path)

        return {
            "summary": summary_payload.get("summary", {}),
            "grid": grid_payload,
            "artifact_paths": artifacts,
        }

    def save_derived_cache(
        self,
        cache_key: str,
        scene_id: str,
        chlorophyll: np.ndarray,
        turbidity: np.ndarray,
        ndwi: np.ndarray,
        risk_normalized: np.ndarray,
        summary: dict[str, Any],
        thresholds: dict[str, float],
        grid: dict[str, Any] | None,
    ) -> dict[str, str]:
        npz_path = self.derived_dir / f"{cache_key}.npz"
        summary_path = self.derived_dir / f"{cache_key}.summary.json"
        grid_path = self.tiles_dir / f"{cache_key}.geojson"

        np.savez_compressed(
            npz_path,
            chlorophyll=chlorophyll.astype(np.float32),
            turbidity=turbidity.astype(np.float32),
            ndwi=ndwi.astype(np.float32),
            risk_normalized=risk_normalized.astype(np.float32),
        )
        summary_payload = {
            "cache_key": cache_key,
            "scene_id": scene_id,
            "summary": summary,
            "thresholds": thresholds,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        if grid is not None:
            grid_path.write_text(json.dumps(grid, indent=2), encoding="utf-8")

        self.enforce_cache_size()
        artifacts = {
            "derived_npz": str(npz_path),
            "summary_json": str(summary_path),
        }
        if grid is not None:
            artifacts["grid_geojson"] = str(grid_path)
        return artifacts

    def enforce_cache_size(self) -> None:
        max_bytes = int(self.settings.cache_max_size_gb * 1024 * 1024 * 1024)
        candidates: list[Path] = []
        for directory in (self.clipped_dir, self.derived_dir, self.tiles_dir):
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if path.is_file():
                    candidates.append(path)
        total_bytes = sum(path.stat().st_size for path in candidates)
        if total_bytes <= max_bytes:
            return

        candidates.sort(key=lambda p: p.stat().st_mtime)
        for file_path in candidates:
            if total_bytes <= max_bytes:
                break
            size = file_path.stat().st_size
            file_path.unlink(missing_ok=True)
            total_bytes -= size
            self._cleanup_empty_parents(file_path.parent, stop=self.cache_root)

    def _bbox_hash(self, bbox: Iterable[float] | None) -> str:
        if bbox is None:
            return "fullscene"
        values = [round(float(v), 5) for v in bbox]
        stable = json.dumps(values, separators=(",", ":"))
        return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:6]

    def _clip_and_store_band(
        self,
        source_path: Path,
        destination: Path,
        bbox: list[float],
        aoi_crs: str,
        max_dimension: int,
    ) -> None:
        with rasterio.open(source_path) as src:
            src_crs = str(src.crs) if src.crs else "EPSG:4326"
            clip_bounds = transform_bounds(bounds=bbox, src_crs=aoi_crs, dst_crs=src_crs)

            raw_window = from_bounds(*clip_bounds, transform=src.transform)
            window = self._sanitize_window(raw_window, src.width, src.height)
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"Clipping window is empty for asset: {source_path}")

            window_w = max(1, int(window.width))
            window_h = max(1, int(window.height))
            scale = 1.0
            if max_dimension > 0 and max(window_w, window_h) > max_dimension:
                scale = float(max_dimension) / float(max(window_w, window_h))
            out_w = max(1, int(window_w * scale))
            out_h = max(1, int(window_h * scale))

            nodata_value = (
                float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else -9999.0
            )
            masked = src.read(
                1,
                window=window,
                out_shape=(out_h, out_w),
                masked=True,
                resampling=Resampling.bilinear,
            )
            data = masked.filled(nodata_value).astype(np.float32)
            transform = src.window_transform(window)
            if out_w != window_w or out_h != window_h:
                x_scale = float(window.width) / float(out_w)
                y_scale = float(window.height) / float(out_h)
                transform = transform * Affine.scale(x_scale, y_scale)

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=out_h,
                width=out_w,
                count=1,
                dtype="float32",
                crs=src.crs,
                transform=transform,
                nodata=nodata_value,
                compress="LZW",
                tiled=True,
                bigtiff="IF_SAFER",
            )
            with rasterio.open(destination, "w", **profile) as dst:
                dst.write(data, 1)

    def _sanitize_window(self, window: Window, width: int, height: int) -> Window:
        row_off = max(0, int(np.floor(window.row_off)))
        col_off = max(0, int(np.floor(window.col_off)))
        row_end = min(height, int(np.ceil(window.row_off + window.height)))
        col_end = min(width, int(np.ceil(window.col_off + window.width)))
        return Window(
            col_off=col_off,
            row_off=row_off,
            width=max(0, col_end - col_off),
            height=max(0, row_end - row_off),
        )

    def _cleanup_empty_parents(self, path: Path, stop: Path) -> None:
        current = path
        while current != stop and current.exists():
            if any(current.iterdir()):
                break
            current.rmdir()
            current = current.parent
