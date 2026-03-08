from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.constants import OPTIONAL_SENTINEL_BANDS, REQUIRED_SENTINEL_BANDS
from app.models.schemas import IngestSentinelRequest, LocalIngestPayload, ProviderType
from app.services.container import build_services
from app.services.processing.indices import chlorophyll_index, ndwi_index, turbidity_index
from app.services.processing.raster import load_and_align_bands
from app.services.processing.risk import score_risk, summarize_risk, temperature_proxy_stub
from app.services.processing.water_mask import compute_water_mask
from app.services.training.weak_labels import binary_risk_label, multiclass_risk_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weakly labeled salmon water-risk training sample from Sentinel bands."
    )
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--scene-dir", type=str, default=None)
    parser.add_argument(
        "--assets-json",
        type=str,
        default=None,
        help="Either JSON string or path to JSON file with band->path mapping.",
    )
    parser.add_argument("--scene-name", type=str, default="dataset-scene")
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="minx,miny,maxx,maxy in --aoi-crs (required unless --allow-full-scene).",
    )
    parser.add_argument("--aoi-crs", type=str, default="EPSG:4326")
    parser.add_argument("--allow-full-scene", action="store_true")
    parser.add_argument("--dataset", type=str, default="sentinel2")
    parser.add_argument(
        "--resolution-label",
        type=str,
        default="native",
        help="Used in deterministic cache key.",
    )
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed",
        help="Base directory for scenes/tiles/labels artifacts.",
    )
    parser.add_argument("--risk-threshold", type=float, default=0.65)
    parser.add_argument("--save-multiclass", action="store_true")
    return parser.parse_args()


def parse_bbox(raw_bbox: str | None) -> list[float] | None:
    if raw_bbox is None:
        return None
    parts = [p.strip() for p in raw_bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'minx,miny,maxx,maxy'.")
    bbox = [float(v) for v in parts]
    minx, miny, maxx, maxy = bbox
    if not (minx < maxx and miny < maxy):
        raise ValueError("Invalid bbox: expected minx < maxx and miny < maxy.")
    return bbox


def parse_assets_json(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("assets-json must be an object mapping band names to file paths.")
    return {str(k): str(v) for k, v in payload.items()}


def next_sample_id(scenes_dir: Path) -> str:
    max_index = 0
    for path in scenes_dir.glob("sample_*_meta.json"):
        name = path.name
        token = name.replace("sample_", "", 1).split("_", 1)[0]
        try:
            max_index = max(max_index, int(token))
        except ValueError:
            continue
    return f"sample_{max_index + 1:03d}"


def stable_assets_hash(assets: dict[str, str]) -> str:
    stable = json.dumps(
        {k: assets[k] for k in sorted(assets.keys())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:6]


def scene_date(scene: dict[str, Any]) -> str:
    value = scene.get("acquired_date") or scene.get("created_at")
    if isinstance(value, str):
        return value.split("T")[0]
    return datetime.now(UTC).date().isoformat()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    services = build_services(settings)

    scenes_dir = settings.resolve_path(args.output_root) / "scenes"
    labels_dir = settings.resolve_path(args.output_root) / "labels"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    bbox = parse_bbox(args.bbox)
    input_scene: dict[str, Any] | None = None
    scene_id: str
    assets: dict[str, str]

    if args.scene_id:
        input_scene = services.metadata_store.get_scene(args.scene_id)
        if input_scene is None:
            raise ValueError(f"Scene not found in registry: {args.scene_id}")
        scene_id = str(input_scene["scene_id"])
        assets = {str(k): str(v) for k, v in input_scene.get("assets", {}).items()}
    else:
        local_assets = parse_assets_json(args.assets_json)
        ingest_request = IngestSentinelRequest(
            provider=ProviderType.local,
            local=LocalIngestPayload(
                scene_name=args.scene_name,
                scene_dir=args.scene_dir,
                assets=local_assets,
            ),
        )
        ingested = services.ingestion_service.ingest_scene(ingest_request)
        scene_id = str(ingested["scene_id"])
        input_scene = services.metadata_store.get_scene(scene_id)
        if input_scene is None:
            raise ValueError("Failed to retrieve ingested scene metadata.")
        assets = {str(k): str(v) for k, v in input_scene.get("assets", {}).items()}

    if not assets:
        raise ValueError("No band assets available for dataset build.")

    if bbox is None:
        scene_bbox = input_scene.get("bbox") if input_scene else None
        if isinstance(scene_bbox, list) and len(scene_bbox) == 4:
            bbox = [float(v) for v in scene_bbox]
        elif not args.allow_full_scene:
            raise ValueError(
                "Provide --bbox (minx,miny,maxx,maxy) for clip-first dataset generation, "
                "or pass --allow-full-scene."
            )

    date_str = scene_date(input_scene or {})
    resolution_label = args.resolution_label
    if bbox is None:
        resolution_label = f"{resolution_label}-{stable_assets_hash(assets)}"
    cache_key = services.cache_manager.build_cache_key(
        dataset=args.dataset,
        date_str=date_str,
        bbox=bbox,
        resolution=resolution_label,
    )

    processing_assets = dict(assets)
    clipped_cache_hit = False
    if bbox is not None:
        optional_bands = tuple(
            band for band in OPTIONAL_SENTINEL_BANDS if band in processing_assets
        )
        cached_clipped = services.cache_manager.get_cached_clipped_assets(
            cache_key=cache_key,
            required_bands=REQUIRED_SENTINEL_BANDS,
            optional_bands=optional_bands,
        )
        if cached_clipped is None:
            processing_assets = services.cache_manager.cache_clipped_assets(
                cache_key=cache_key,
                source_assets=processing_assets,
                bbox=bbox,
                aoi_crs=args.aoi_crs,
                required_bands=REQUIRED_SENTINEL_BANDS,
                optional_bands=optional_bands,
            )
        else:
            processing_assets = cached_clipped
            clipped_cache_hit = True

    raster_bundle = load_and_align_bands(
        assets=processing_assets,
        required_bands=REQUIRED_SENTINEL_BANDS,
        aoi_bbox=None if bbox is not None else None,
        aoi_crs=args.aoi_crs,
    )
    b3 = raster_bundle.arrays["B3"]
    b4 = raster_bundle.arrays["B4"]
    b5 = raster_bundle.arrays["B5"]
    b8 = raster_bundle.arrays["B8"]

    chlorophyll = chlorophyll_index(b5=b5, b4=b4)
    turbidity = turbidity_index(b4=b4, b3=b3)
    ndwi = ndwi_index(b3=b3, b8=b8)
    water_mask = compute_water_mask(ndwi=ndwi, threshold=settings.ndwi_water_threshold)
    _, risk_norm = score_risk(
        chlorophyll=chlorophyll,
        turbidity=turbidity,
        water_mask=water_mask,
        temperature=temperature_proxy_stub(chlorophyll),
    )
    binary_label = binary_risk_label(
        risk_norm=risk_norm,
        threshold=args.risk_threshold,
        water_mask=water_mask,
    )

    sample_id = args.sample_id or next_sample_id(scenes_dir)
    bands_stack = np.stack([b3, b4, b5, b8], axis=0).astype(np.float32)
    bands_path = scenes_dir / f"{sample_id}_bands.npz"
    risk_path = scenes_dir / f"{sample_id}_risk.npy"
    water_mask_path = scenes_dir / f"{sample_id}_mask.npy"
    binary_label_path = labels_dir / f"{sample_id}_binary.npy"

    np.savez_compressed(
        bands_path,
        bands=bands_stack,
        band_order=np.array(["B3", "B4", "B5", "B8"]),
    )
    np.save(risk_path, risk_norm.astype(np.float32))
    np.save(water_mask_path, water_mask.astype(np.uint8))
    np.save(binary_label_path, binary_label.astype(np.uint8))

    multiclass_path = None
    if args.save_multiclass:
        multiclass = multiclass_risk_label(
            risk_norm=risk_norm,
            thresholds=settings.heatmap_thresholds,
            water_mask=water_mask,
        )
        multiclass_path = labels_dir / f"{sample_id}_multiclass.npy"
        np.save(multiclass_path, multiclass.astype(np.uint8))

    summary = summarize_risk(
        risk=risk_norm,
        chlorophyll=chlorophyll,
        turbidity=turbidity,
        water_mask=water_mask,
    )
    summary["ndwi_mean"] = float(np.nanmean(ndwi[water_mask])) if np.any(water_mask) else 0.0
    summary["positive_fraction_binary"] = float(binary_label.mean())

    services.cache_manager.save_derived_cache(
        cache_key=cache_key,
        scene_id=scene_id,
        chlorophyll=chlorophyll,
        turbidity=turbidity,
        ndwi=ndwi,
        risk_normalized=risk_norm,
        summary=summary,
        thresholds=settings.heatmap_thresholds,
        grid=None,
    )

    metadata = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "cache_key": cache_key,
        "dataset": args.dataset,
        "date": date_str,
        "bbox": bbox,
        "aoi_crs": args.aoi_crs if bbox is not None else None,
        "shape": {"height": int(risk_norm.shape[0]), "width": int(risk_norm.shape[1])},
        "crs": raster_bundle.crs,
        "transform": list(raster_bundle.transform),
        "paths": {
            "bands_npz": str(bands_path),
            "risk_npy": str(risk_path),
            "water_mask_npy": str(water_mask_path),
            "binary_label_npy": str(binary_label_path),
            "multiclass_label_npy": str(multiclass_path) if multiclass_path else None,
            "clipped_cache_dir": str(services.cache_manager.clipped_dir / cache_key)
            if bbox is not None
            else None,
        },
        "summary": summary,
        "clip_cache_hit": clipped_cache_hit,
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = scenes_dir / f"{sample_id}_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest_path = settings.resolve_path(args.output_root) / "dataset_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

