from __future__ import annotations

from pathlib import Path


class PrithviDatasetBuilder:
    def prepare(self, dataset_path: str | None) -> dict[str, str]:
        if dataset_path is None:
            return {
                "status": "placeholder",
                "message": "No dataset path provided. Using scaffold-only dry run.",
            }
        resolved = Path(dataset_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path not found: {resolved}")
        # TODO: Convert local Sentinel assets into Prithvi-compatible training shards.
        return {
            "status": "validated",
            "dataset_path": str(resolved),
            "message": "Dataset path exists. Formatting pipeline is scaffolded.",
        }

