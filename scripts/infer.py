from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure `app` imports resolve when running as `python scripts/infer.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.container import build_services


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model inference on a scene and cache frontend-ready outputs."
    )
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--bbox", type=str, default=None, help="minx,miny,maxx,maxy")
    parser.add_argument("--aoi-crs", type=str, default="EPSG:4326")
    parser.add_argument("--grid-block-size", type=int, default=32)
    parser.add_argument("--migration-path-id", type=str, default=None)
    parser.add_argument("--migration-buffer-meters", type=float, default=None)
    parser.add_argument("--inference-tile-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto/cpu/cuda")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def parse_bbox(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'minx,miny,maxx,maxy'.")
    values = [float(v) for v in parts]
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError("Invalid bbox: expected minx < maxx and miny < maxy.")
    return values


def main() -> None:
    args = parse_args()
    settings = get_settings()
    services = build_services(settings)
    bbox = parse_bbox(args.bbox)

    result = services.inference_service.predict_scene(
        scene_id=args.scene_id,
        model_checkpoint=args.model_checkpoint,
        model_id=args.model_id,
        aoi_bbox=bbox,
        aoi_crs=args.aoi_crs,
        include_grid=True,
        grid_block_size=args.grid_block_size,
        migration_path_id=args.migration_path_id,
        migration_buffer_meters=args.migration_buffer_meters,
        inference_tile_size=args.inference_tile_size,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        force_recompute=args.force_recompute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
