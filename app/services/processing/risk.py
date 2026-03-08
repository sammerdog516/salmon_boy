from __future__ import annotations

import numpy as np

from app.core.constants import (
    DEFAULT_CHLOROPHYLL_WEIGHT,
    DEFAULT_TEMPERATURE_WEIGHT,
    DEFAULT_TURBIDITY_WEIGHT,
)


def temperature_proxy_stub(reference: np.ndarray) -> np.ndarray:
    # TODO: Replace with optional thermal provider when available.
    return np.zeros(reference.shape, dtype=np.float32)


def normalize_risk(
    risk_raw: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    normalized = np.full(risk_raw.shape, np.nan, dtype=np.float32)
    valid_values = risk_raw[valid_mask & np.isfinite(risk_raw)]
    if valid_values.size == 0:
        return normalized

    min_value = float(np.nanmin(valid_values))
    max_value = float(np.nanmax(valid_values))
    if max_value == min_value:
        normalized[valid_mask] = 0.0
        return normalized

    normalized[valid_mask] = (risk_raw[valid_mask] - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid_mask] = np.nan
    return normalized.astype(np.float32)


def score_risk(
    chlorophyll: np.ndarray,
    turbidity: np.ndarray,
    water_mask: np.ndarray,
    temperature: np.ndarray | None = None,
    chlorophyll_weight: float = DEFAULT_CHLOROPHYLL_WEIGHT,
    turbidity_weight: float = DEFAULT_TURBIDITY_WEIGHT,
    temperature_weight: float = DEFAULT_TEMPERATURE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    temp = temperature if temperature is not None else temperature_proxy_stub(chlorophyll)
    # Chlorophyll index can be negative (no bloom) — clip to 0 so absence of bloom
    # doesn't cancel out turbidity signal in the weighted sum.
    chl_positive = np.clip(chlorophyll, 0.0, None)
    risk_raw = (
        chlorophyll_weight * chl_positive
        + turbidity_weight * turbidity
        + temperature_weight * temp
    ).astype(np.float32)
    valid_water = water_mask & np.isfinite(chlorophyll) & np.isfinite(turbidity)
    risk_norm = normalize_risk(risk_raw, valid_water)
    return risk_raw, risk_norm


def risk_category(
    score: float,
    thresholds: dict[str, float],
) -> str:
    if score >= thresholds["infrared"]:
        return "infrared"
    if score >= thresholds["red"]:
        return "red"
    if score >= thresholds["yellow"]:
        return "yellow"
    return "blue"


def summarize_risk(
    risk: np.ndarray,
    chlorophyll: np.ndarray,
    turbidity: np.ndarray,
    water_mask: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(risk)
    water_fraction = float(np.count_nonzero(water_mask)) / float(water_mask.size) if water_mask.size else 0.0
    if not np.any(valid):
        return {
            "risk_mean": 0.0,
            "risk_max": 0.0,
            "risk_min": 0.0,
            "chlorophyll_index_mean": 0.0,
            "turbidity_index_mean": 0.0,
            "water_fraction": water_fraction,
            "valid_water_pixels": 0,
        }
    return {
        "risk_mean": float(np.nanmean(risk[valid])),
        "risk_max": float(np.nanmax(risk[valid])),
        "risk_min": float(np.nanmin(risk[valid])),
        "chlorophyll_index_mean": float(np.nanmean(chlorophyll[water_mask])),
        "turbidity_index_mean": float(np.nanmean(turbidity[water_mask])),
        "water_fraction": water_fraction,
        "valid_water_pixels": int(np.count_nonzero(valid)),
    }

