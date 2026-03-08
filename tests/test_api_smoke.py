from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def _write_band(path: Path, data: np.ndarray) -> None:
    transform = from_origin(-123.2, 49.3, 0.01, 0.01)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def test_ingest_and_process_scene_smoke(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    b3 = np.full((8, 8), 0.5, dtype=np.float32)
    b4 = np.full((8, 8), 0.3, dtype=np.float32)
    b5 = np.full((8, 8), 0.6, dtype=np.float32)
    b8 = np.full((8, 8), 0.2, dtype=np.float32)

    b3_path = scene_dir / "B3.tif"
    b4_path = scene_dir / "B4.tif"
    b5_path = scene_dir / "B5.tif"
    b8_path = scene_dir / "B8.tif"
    _write_band(b3_path, b3)
    _write_band(b4_path, b4)
    _write_band(b5_path, b5)
    _write_band(b8_path, b8)

    ingest_payload = {
        "provider": "local",
        "local": {
            "scene_name": "smoke-scene",
            "assets": {
                "B3": str(b3_path),
                "B4": str(b4_path),
                "B5": str(b5_path),
                "B8": str(b8_path),
            },
        },
    }
    ingest_response = client.post("/ingest/sentinel", json=ingest_payload)
    assert ingest_response.status_code == 200, ingest_response.text
    scene_id = ingest_response.json()["scene_id"]

    process_response = client.post(
        "/process/scene",
        json={"scene_id": scene_id, "include_grid": True, "grid_block_size": 4},
    )
    assert process_response.status_code == 200, process_response.text
    process_payload = process_response.json()
    assert process_payload["scene_id"] == scene_id
    assert "summary" in process_payload
    assert process_payload["summary"]["valid_water_pixels"] > 0
    assert "artifact_paths" in process_payload
    assert Path(process_payload["artifact_paths"]["derived_npz"]).exists()
    assert Path(process_payload["artifact_paths"]["summary_json"]).exists()

    tiles_response = client.get(
        "/risk/tiles",
        params={"processed_scene_id": process_payload["processed_scene_id"]},
    )
    assert tiles_response.status_code == 200, tiles_response.text
    tiles_payload = tiles_response.json()
    assert tiles_payload["features"]["type"] == "FeatureCollection"
