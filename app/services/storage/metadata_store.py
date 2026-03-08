from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetadataStore:
    def __init__(
        self,
        scene_registry_path: Path,
        processed_registry_path: Path,
        training_registry_path: Path,
        prediction_registry_path: Path,
    ) -> None:
        self.scene_registry_path = scene_registry_path
        self.processed_registry_path = processed_registry_path
        self.training_registry_path = training_registry_path
        self.prediction_registry_path = prediction_registry_path
        self._ensure_files()

    def _ensure_files(self) -> None:
        for path in (
            self.scene_registry_path,
            self.processed_registry_path,
            self.training_registry_path,
            self.prediction_registry_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("{}", encoding="utf-8")

    def _read_registry(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _write_registry(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_record(self, path: Path, record_id: str, payload: dict[str, Any]) -> None:
        records = self._read_registry(path)
        records[record_id] = payload
        self._write_registry(path, records)

    def _get_record(self, path: Path, record_id: str) -> dict[str, Any] | None:
        records = self._read_registry(path)
        value = records.get(record_id)
        if isinstance(value, dict):
            return value
        return None

    def _list_records(self, path: Path) -> list[dict[str, Any]]:
        records = self._read_registry(path)
        values = [v for v in records.values() if isinstance(v, dict)]
        values.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return values

    def save_scene(self, scene_id: str, payload: dict[str, Any]) -> None:
        self._save_record(self.scene_registry_path, scene_id, payload)

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        return self._get_record(self.scene_registry_path, scene_id)

    def list_scenes(self) -> list[dict[str, Any]]:
        return self._list_records(self.scene_registry_path)

    def save_processed_scene(self, processed_scene_id: str, payload: dict[str, Any]) -> None:
        self._save_record(self.processed_registry_path, processed_scene_id, payload)

    def get_processed_scene(self, processed_scene_id: str) -> dict[str, Any] | None:
        return self._get_record(self.processed_registry_path, processed_scene_id)

    def list_processed_scenes(self) -> list[dict[str, Any]]:
        return self._list_records(self.processed_registry_path)

    def save_training_job(self, job_id: str, payload: dict[str, Any]) -> None:
        self._save_record(self.training_registry_path, job_id, payload)

    def get_training_job(self, job_id: str) -> dict[str, Any] | None:
        return self._get_record(self.training_registry_path, job_id)

    def list_training_jobs(self) -> list[dict[str, Any]]:
        return self._list_records(self.training_registry_path)

    def save_prediction(self, prediction_id: str, payload: dict[str, Any]) -> None:
        self._save_record(self.prediction_registry_path, prediction_id, payload)

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return self._get_record(self.prediction_registry_path, prediction_id)

    def list_predictions(self) -> list[dict[str, Any]]:
        return self._list_records(self.prediction_registry_path)
