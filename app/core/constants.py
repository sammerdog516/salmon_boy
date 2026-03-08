from __future__ import annotations

from dataclasses import dataclass


REQUIRED_SENTINEL_BANDS = ("B3", "B4", "B5", "B8")
OPTIONAL_SENTINEL_BANDS = ("B2", "B11", "B12")

DEFAULT_CHLOROPHYLL_WEIGHT = 0.5
DEFAULT_TURBIDITY_WEIGHT = 0.3
DEFAULT_TEMPERATURE_WEIGHT = 0.2


@dataclass(frozen=True)
class HeatmapThresholds:
    blue: float = 0.0
    yellow: float = 0.30
    red: float = 0.65
    infrared: float = 0.85

