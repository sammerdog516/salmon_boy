from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.processing.raster import RasterBundle
from app.services.processing.water_mask import compute_water_mask_refined


@dataclass
class WaterDetectionResult:
    mask: np.ndarray
    method: str
    details: dict[str, Any]


def detect_water_mask(
    bundle: RasterBundle,
    ndwi: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    threshold: float,
    nir_to_green_ratio_max: float,
    ndvi_max: float,
    mode: str = "auto",
    pretrained_repo_id: str | None = None,
    hf_token: str | None = None,
) -> WaterDetectionResult:
    mode_normalized = (mode or "auto").strip().lower()
    spectral_mask = compute_water_mask_refined(
        ndwi=ndwi,
        b3=b3,
        b4=b4,
        b8=b8,
        threshold=threshold,
        nir_to_green_ratio_max=nir_to_green_ratio_max,
        ndvi_max=ndvi_max,
    )
    if mode_normalized == "spectral":
        return WaterDetectionResult(
            mask=spectral_mask,
            method="spectral_refined",
            details={},
        )

    if mode_normalized not in {"auto", "pretrained"}:
        return WaterDetectionResult(
            mask=spectral_mask,
            method="spectral_refined_invalid_mode_fallback",
            details={"requested_mode": mode},
        )

    required = ("B2", "B3", "B4", "B8", "B11", "B12")
    missing = [band for band in required if band not in bundle.arrays]
    if missing:
        return WaterDetectionResult(
            mask=spectral_mask,
            method="spectral_refined_missing_pretrained_bands",
            details={"missing_bands": missing},
        )

    pretrained, pretrained_error = _detect_water_with_geoai(
        bundle=bundle,
        repo_id=pretrained_repo_id or "geoai4cities/sentinel2-water-segmentation",
        hf_token=hf_token,
    )
    if pretrained is not None:
        return pretrained

    if mode_normalized == "pretrained":
        # Explicit pretrained mode but backend dependency unavailable.
        return WaterDetectionResult(
            mask=spectral_mask,
            method="spectral_refined_pretrained_unavailable",
            details={
                "hint": (
                    "Install geoai-py and dependencies, then set "
                    "WATER_DETECTOR_MODE=pretrained."
                ),
                "error": pretrained_error,
            },
        )
    return WaterDetectionResult(
        mask=spectral_mask,
        method="spectral_refined_pretrained_fallback",
        details={"error": pretrained_error},
    )


def _detect_water_with_geoai(
    bundle: RasterBundle,
    repo_id: str,
    hf_token: str | None,
) -> tuple[WaterDetectionResult | None, str | None]:
    run_dir = Path.cwd() / "artifacts" / "tmp" / "water_model" / f"run-{uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = run_dir / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["TMP"] = str(run_dir)
    os.environ["TEMP"] = str(run_dir)
    os.environ.setdefault("HF_HOME", str(run_dir / "hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(run_dir / "hf_home" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import rasterio
        from geoai.timm_segment import timm_segmentation_from_hub
    except Exception as exc:
        return None, f"geoai_import_failed: {exc}"

    band_order = ("B2", "B3", "B4", "B8", "B11", "B12")
    stack = np.stack([bundle.arrays[band] for band in band_order], axis=0).astype(np.float32)
    scale = 10000.0 if float(np.nanmax(stack)) <= 1.0 else 1.0
    scaled = np.clip(stack * scale, 0.0, 65535.0).astype(np.uint16)

    try:
        input_path = run_dir / "s2_input.tif"
        output_path = run_dir / "water_mask.tif"
        profile = {
            "driver": "GTiff",
            "height": bundle.height,
            "width": bundle.width,
            "count": 6,
            "dtype": "uint16",
            "crs": bundle.crs,
            "transform": bundle.transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": 0,
        }
        try:
            with rasterio.open(input_path, "w", **profile) as dst:
                dst.write(scaled)
        except Exception as exc:
            return None, f"geoai_input_write_failed: {exc}"

        try:
            timm_segmentation_from_hub(
                input_path=str(input_path),
                output_path=str(output_path),
                repo_id=repo_id,
                quiet=True,
                token=hf_token,
            )
        except Exception as exc:
            return None, f"geoai_inference_failed: {exc}"

        if not output_path.exists():
            return None, "geoai_output_missing"

        with rasterio.open(output_path) as src:
            prediction = src.read(1)
    except Exception as exc:
        return None, f"geoai_runtime_failed: {exc}"

    mask = np.isfinite(prediction) & (prediction > 0)
    return (
        WaterDetectionResult(
            mask=mask,
            method="pretrained_geoai",
            details={"repo_id": repo_id},
        ),
        None,
    )
