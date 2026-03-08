from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    migration_path = tmp_path / "data" / "migration_paths" / "salmon_paths.geojson"
    migration_path.parent.mkdir(parents=True, exist_ok=True)
    migration_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "test-path",
                "properties": {"name": "Test Path"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-123.1, 49.2], [-122.9, 49.25]],
                },
            }
        ],
    }
    migration_path.write_text(json.dumps(migration_payload), encoding="utf-8")

    settings = Settings(
        app_name="Test Backend",
        app_version="test",
        environment="test",
        project_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "artifacts" / "cache",
        scene_registry_path=tmp_path / "artifacts" / "registry" / "scenes.json",
        processed_registry_path=tmp_path / "artifacts" / "registry" / "processed_scenes.json",
        training_registry_path=tmp_path / "artifacts" / "registry" / "training_jobs.json",
        prediction_registry_path=tmp_path / "artifacts" / "registry" / "predictions.json",
        migration_paths_file=migration_path,
        default_grid_block_size=4,
    )
    settings.ensure_directories()
    return settings


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
