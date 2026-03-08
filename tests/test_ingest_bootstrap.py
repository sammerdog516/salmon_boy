from __future__ import annotations

from pathlib import Path

from app.core.constants import REQUIRED_SENTINEL_BANDS


def test_local_ingest_bootstraps_missing_sample_scene(client) -> None:
    payload = {
        "provider": "local",
        "local": {
            "scene_name": "auto-sample-scene",
            "scene_dir": "data/sample",
        },
    }
    response = client.post("/ingest/sentinel", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assets = body["assets"]
    for band in REQUIRED_SENTINEL_BANDS:
        assert band in assets
        assert Path(assets[band]).exists()
    assert "auto-generated" in body["message"]


def test_local_ingest_defaults_to_sample_scene_when_local_payload_is_missing(client) -> None:
    response = client.post("/ingest/sentinel", json={"provider": "local"})
    assert response.status_code == 200, response.text
    body = response.json()
    assets = body["assets"]
    for band in REQUIRED_SENTINEL_BANDS:
        assert band in assets
        assert Path(assets[band]).exists()
