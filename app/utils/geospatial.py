from __future__ import annotations

from typing import Iterable

from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


def transform_bounds(
    bounds: Iterable[float], src_crs: str | CRS, dst_crs: str | CRS
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    if str(src_crs) == str(dst_crs):
        return (float(minx), float(miny), float(maxx), float(maxy))
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)
    return (float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2)))


def project_geometry(geometry: BaseGeometry, src_crs: str | CRS, dst_crs: str | CRS) -> BaseGeometry:
    if str(src_crs) == str(dst_crs):
        return geometry
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def buffer_geometry_meters(
    geometry_wgs84: BaseGeometry, buffer_meters: float
) -> BaseGeometry:
    if buffer_meters <= 0:
        return geometry_wgs84
    geometry_3857 = project_geometry(geometry_wgs84, "EPSG:4326", "EPSG:3857")
    buffered_3857 = geometry_3857.buffer(buffer_meters)
    return project_geometry(buffered_3857, "EPSG:3857", "EPSG:4326")

