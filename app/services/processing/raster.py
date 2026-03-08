from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds

from app.utils.geospatial import transform_bounds


@dataclass
class RasterBundle:
    arrays: dict[str, np.ndarray]
    transform: Affine
    crs: str
    width: int
    height: int


def _read_band(path: str | Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    src = rasterio.open(path)
    data = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    return data, src


def _read_or_reproject_to_reference(
    path: str | Path,
    ref_crs: str,
    ref_transform: Affine,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    with rasterio.open(path) as src:
        same_grid = (
            str(src.crs) == str(ref_crs)
            and src.transform.almost_equals(ref_transform)
            and src.height == ref_shape[0]
            and src.width == ref_shape[1]
        )
        if same_grid:
            band = src.read(1).astype(np.float32)
            if src.nodata is not None and np.isfinite(src.nodata):
                band[band == src.nodata] = np.nan
            return band

        destination = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return destination


def _clip_arrays(
    arrays: dict[str, np.ndarray],
    transform: Affine,
    bounds: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], Affine]:
    first = next(iter(arrays.values()))
    window = from_bounds(*bounds, transform=transform)
    row_off = max(0, int(np.floor(window.row_off)))
    col_off = max(0, int(np.floor(window.col_off)))
    row_end = min(first.shape[0], int(np.ceil(window.row_off + window.height)))
    col_end = min(first.shape[1], int(np.ceil(window.col_off + window.width)))

    if row_off >= row_end or col_off >= col_end:
        raise ValueError("AOI clipping produced an empty window.")

    clipped = {
        band: values[row_off:row_end, col_off:col_end]
        for band, values in arrays.items()
    }
    clipped_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )
    clipped_transform = rasterio.windows.transform(clipped_window, transform)
    return clipped, clipped_transform


def load_and_align_bands(
    assets: dict[str, str],
    required_bands: tuple[str, ...],
    aoi_bbox: list[float] | None = None,
    aoi_crs: str = "EPSG:4326",
) -> RasterBundle:
    missing = [band for band in required_bands if band not in assets]
    if missing:
        raise ValueError(f"Missing required band assets: {', '.join(missing)}")

    # Fail fast on stale scene registry paths so API returns a clear 404-style error.
    for band_name, raw_path in assets.items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Asset path not found for {band_name}: {path}. "
                "Re-ingest the scene with valid local files."
            )

    reference_band = required_bands[0]
    ref_path = assets[reference_band]
    with rasterio.open(ref_path) as ref_src:
        ref_crs = str(ref_src.crs) if ref_src.crs else "EPSG:4326"
        ref_transform = ref_src.transform
        ref_shape = (ref_src.height, ref_src.width)

    arrays = {}
    for band in required_bands:
        arrays[band] = _read_or_reproject_to_reference(
            path=assets[band],
            ref_crs=ref_crs,
            ref_transform=ref_transform,
            ref_shape=ref_shape,
        )

    for optional_band in ("B2", "B11", "B12"):
        if optional_band in assets:
            arrays[optional_band] = _read_or_reproject_to_reference(
                path=assets[optional_band],
                ref_crs=ref_crs,
                ref_transform=ref_transform,
                ref_shape=ref_shape,
            )

    if aoi_bbox is not None:
        if len(aoi_bbox) != 4:
            raise ValueError("aoi_bbox must be [minx, miny, maxx, maxy].")
        aoi_bounds_in_ref = transform_bounds(aoi_bbox, src_crs=aoi_crs, dst_crs=ref_crs)
        arrays, ref_transform = _clip_arrays(arrays, ref_transform, aoi_bounds_in_ref)

    sample = next(iter(arrays.values()))
    return RasterBundle(
        arrays=arrays,
        transform=ref_transform,
        crs=str(CRS.from_user_input(ref_crs)),
        width=sample.shape[1],
        height=sample.shape[0],
    )

