from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.services.storage.metadata_store import MetadataStore


def _build_store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(
        scene_registry_path=tmp_path / "registry" / "scenes.json",
        processed_registry_path=tmp_path / "registry" / "processed.json",
        training_registry_path=tmp_path / "registry" / "training.json",
        prediction_registry_path=tmp_path / "registry" / "predictions.json",
    )


def test_prediction_registry_thread_safe(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    total = 200

    def write_one(index: int) -> None:
        prediction_id = f"pred-{index:04d}"
        store.save_prediction(
            prediction_id,
            {
                "prediction_id": prediction_id,
                "scene_id": "scene-1",
                "created_at": f"2026-03-08T00:00:{index:02d}Z",
            },
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write_one, range(total)))

    records = store.list_predictions()
    assert len(records) == total
    assert store.get_prediction("pred-0000") is not None
    assert store.get_prediction(f"pred-{(total - 1):04d}") is not None
