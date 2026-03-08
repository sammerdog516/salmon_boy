from __future__ import annotations

from app.services.processing.raster import load_and_align_bands


def test_load_and_align_bands_missing_asset_path_raises_file_not_found() -> None:
    missing = "C:/definitely_missing_for_test/B3.tif"
    assets = {
        "B3": missing,
        "B4": missing,
        "B5": missing,
        "B8": missing,
    }
    try:
        load_and_align_bands(assets=assets, required_bands=("B3", "B4", "B5", "B8"))
        raise AssertionError("Expected FileNotFoundError for missing asset paths.")
    except FileNotFoundError as exc:
        assert "Asset path not found for B3" in str(exc)
