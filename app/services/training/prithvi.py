from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import Settings
from app.models.schemas import PrithviTrainRequest
from app.services.storage.metadata_store import MetadataStore
from app.services.training.dataset import PrithviDatasetBuilder


class PrithviTrainingService:
    def __init__(self, settings: Settings, metadata_store: MetadataStore) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self.dataset_builder = PrithviDatasetBuilder()

    def start_training_job(self, request: PrithviTrainRequest) -> dict[str, str]:
        dataset_info = self.dataset_builder.prepare(request.dataset_path)
        job_id = f"train-{uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        status = "completed" if request.dry_run else "running"
        message = (
            "Dry-run scaffold completed. Rule-based risk scoring remains primary MVP."
            if request.dry_run
            else "Training job scaffold registered. Execution is simulated for MVP."
        )

        payload = {
            "job_id": job_id,
            "status": status,
            "model_name": self.settings.prithvi_model_name,
            "created_at": now,
            "updated_at": now,
            "request": request.model_dump(),
            "dataset_info": dataset_info,
            "message": message,
            "todo": [
                "Add optional huggingface model loader behind feature flag.",
                "Implement embedding extraction pipeline for EO tiles.",
                "Implement real fine-tuning loop with checkpoint persistence.",
            ],
        }
        self.metadata_store.save_training_job(job_id, payload)
        return {
            "job_id": job_id,
            "status": status,
            "model_name": self.settings.prithvi_model_name,
            "message": message,
        }

    def mark_completed(self, job_id: str) -> dict[str, str]:
        record = self.metadata_store.get_training_job(job_id)
        if record is None:
            raise ValueError(f"Training job not found: {job_id}")
        record["status"] = "completed"
        record["updated_at"] = datetime.now(UTC).isoformat()
        record["message"] = "Simulated training completed."
        self.metadata_store.save_training_job(job_id, record)
        return {
            "job_id": job_id,
            "status": "completed",
        }

    def get_status(self, job_id: str | None = None) -> list[dict[str, object]]:
        if job_id:
            record = self.metadata_store.get_training_job(job_id)
            return [record] if record else []
        return self.metadata_store.list_training_jobs()

