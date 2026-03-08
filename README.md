# Salmon Water Risk Backend (Hackathon MVP)

FastAPI backend MVP for detecting water-related environmental risk proxies along salmon migratory paths using Sentinel-2 style inputs.

This repository is optimized for an end-to-end local demo:
- local scene ingestion (fully implemented)
- raster preprocessing and spectral index calculations
- water-only risk scoring (`0..1`)
- heatmap-ready GeoJSON grid output
- migration-path-aware risk summaries
- 3-layer deterministic cache with size cap + eviction
- Prithvi model training scaffold (non-blocking)

## What Is Implemented Now

### Real MVP functionality
- Local Sentinel-like ingestion via `POST /ingest/sentinel` (`provider=local`)
- Band validation (`B3`, `B4`, `B5`, `B8`)
- Raster loading with reprojection to a reference grid when needed
- Optional AOI clipping
- Spectral proxies:
  - chlorophyll/algal proxy: `(B5 - B4) / (B5 + B4)`
  - turbidity proxy: `B4 / B3`
  - NDWI water mask: `(B3 - B8) / (B3 + B8)`
- Water detection modes:
  - `spectral`: NDWI + NIR/green + NDVI gates
  - `auto` (default): tries pretrained water model if available, otherwise spectral fallback
  - `pretrained`: explicit pretrained attempt, with spectral fallback metadata if unavailable
- Water-only risk scoring:
  - `risk_raw = 0.5*chlorophyll + 0.3*turbidity + 0.2*temperature_proxy`
  - `temperature_proxy` is currently a stub (`0.0`) with explicit TODO
  - normalized risk in `0..1` on water pixels
- GeoJSON grid aggregation with risk category metadata for frontend heatmaps
- Migration path loading from static GeoJSON + buffered intersection summary
- Artifact persistence (GeoTIFF + JSON + GeoJSON) and local registry tracking
- Cache policy:
  - tiny metadata cache (`cache/metadata`)
  - clipped-band cache (`cache/clipped`)
  - derived risk cache (`cache/derived`, `cache/tiles`)
  - deterministic cache keys and oldest-first eviction

### Scaffolded (future-facing)
- `provider=sentinel` remote ingestion interface exists but is intentionally stubbed
- Prithvi integration endpoints exist, with dataset validation and job tracking scaffold
- No heavyweight model download/fine-tuning required for MVP run

## Architecture

```text
app/
  main.py
  api/
    routes_health.py
    routes_ingest.py
    routes_process.py
    routes_risk.py
    routes_migration.py
    routes_train.py
  core/
    config.py
    constants.py
    logging.py
  models/
    schemas.py
  services/
    container.py
    ingestion/
      base.py
      local_provider.py
      sentinel_provider.py
      service.py
    processing/
      raster.py
      indices.py
      water_mask.py
      risk.py
      grid.py
      service.py
    migration/
      loader.py
      summarizer.py
    training/
      dataset.py
      inference.py
      prithvi.py
      weak_labels.py
    storage/
      metadata_store.py
      cache_manager.py
  utils/
    bands.py
    geospatial.py
scripts/
  build_dataset.py
  tile_dataset.py
  train.py
  infer.py
data/
  raw/
  processed/
  sample/
  migration_paths/
artifacts/
tests/
```

## Cache Strategy (3 Layers)

The backend now uses only these cache layers under `artifacts/cache/`:

1. `metadata/` (tiny JSON records)
- stores `scene_id`, `bbox`, `date`, `cloud_cover`, `source_urls`, `request_hash`
- avoids repeated ingestion/search calls for identical requests

2. `clipped/` (clip-first band cache)
- stores only corridor/AOI-clipped bands
- stores only required Sentinel bands (`B3`, `B4`, `B5`, `B8`) and optional `B2/B11/B12` if present
- writes compressed GeoTIFFs (LZW), optional downsample via `CLIPPED_CACHE_MAX_DIMENSION`
- never stores full scenes in clipped cache

3. `derived/` and `tiles/` (demo-ready outputs)
- `derived/*.npz`: chlorophyll, turbidity, NDWI, normalized risk arrays
- `derived/*.summary.json`: summary and thresholds
- `tiles/*.geojson`: heatmap-ready polygons

### Deterministic Cache Key

Format:

`{dataset}_{date}_{bboxhash}_{resolution}`

Example:

`sentinel2_2026-03-07_a13f92_native-g32`

Model prediction caches append model identity into the resolution fragment, e.g.:

`sentinel2_2026-03-07_a13f92_native-g32-m1a2b3c4`

### Eviction Policy

- configurable max cache size (`CACHE_MAX_SIZE_GB`, default `10`)
- oldest files removed first from clipped/derived/tile layers

## Data Flow

1. Scene ingestion (`/ingest/sentinel`)
2. Scene metadata registration in local JSON registry
3. Band validation and loading
4. Raster alignment/reprojection and optional AOI clipping
5. Chlorophyll proxy computation
6. Turbidity proxy computation
7. NDWI and water mask generation
8. Water-only filtering
9. Raw + normalized risk computation (`0..1`)
10. Grid aggregation to heatmap-ready GeoJSON features
11. Optional migration-path buffered intersection summary
12. API delivery of summaries/artifact references/grid features

## Dataset-First Training Flow

Order used in this repo:

1. `scripts/build_dataset.py`
- input: local scene assets + bbox
- output: scene-level arrays (`4-band`, `risk_norm`, `water_mask`, `binary_label`)
- writes manifests and metadata for repeatability

2. `scripts/tile_dataset.py`
- converts scene arrays into train/val tiles (default `256x256`)
- keeps tiles with sufficient water and a balanced positive/negative mix
- writes `tiles_manifest.jsonl`

3. `scripts/train.py`
- trains a lightweight binary segmentation baseline on tiled weak labels
- target is `label = (risk_norm >= 0.65)` (water-aware)
- `--model prithvi-head` is scaffolded and currently falls back to baseline model

## Risk Thresholds

Configured in environment/config (not hardcoded throughout code):
- `blue`: baseline/low
- `yellow`: `risk >= 0.30`
- `red`: `risk >= 0.65`
- `infrared`: `risk >= 0.85`

## Setup

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For Python 3.13 environments, use:

```bash
pip install -r requirements-py313.txt
```

3. Copy env defaults:

```bash
cp .env.example .env
```

4. Run API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Optional for local training script:

```bash
python -m pip install torch
```

Optional for pretrained water detector (6-band Sentinel scenes):

```bash
python -m pip install geoai-py
```

Then set in `.env`:

```env
WATER_DETECTOR_MODE=pretrained
PRETRAINED_WATER_MODEL_REPO_ID=geoai4cities/sentinel2-water-segmentation
```

## API Endpoints

- `GET /health`
- `GET /migration-paths`
- `POST /ingest/sentinel`
- `POST /process/scene`
- `POST /risk/score`
- `POST /risk/predict`
- `GET /risk/tiles`
- `POST /train/prithvi`
- `GET /train/status`

## Example Calls

### 1) Ingest local Sentinel-like scene

```bash
curl -X POST http://localhost:8000/ingest/sentinel \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "local",
    "local": {
      "scene_name": "demo-scene",
      "assets": {
        "B3": "data/sample/B3.tif",
        "B4": "data/sample/B4.tif",
        "B5": "data/sample/B5.tif",
        "B8": "data/sample/B8.tif"
      }
    }
  }'
```

### 2) Process scene to produce artifacts and grid

```bash
curl -X POST http://localhost:8000/process/scene \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "scene-abc123",
    "include_grid": true,
    "grid_block_size": 32,
    "migration_path_id": "columbia-mainstem"
  }'
```

### 3) Fetch heatmap-ready GeoJSON

```bash
curl "http://localhost:8000/risk/tiles?processed_scene_id=proc-abc123"
```

### 3b) Predict with trained model and then fetch model tiles

```bash
curl -X POST http://localhost:8000/risk/predict \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "scene-abc123",
    "model_checkpoint": "artifacts/models/weakrisk_baseline/best.pt",
    "grid_block_size": 32
  }'
```

Then request tiles using model source (same frontend shape):

```bash
curl "http://localhost:8000/risk/tiles?source=model&prediction_id=pred-abc123"
```

### 4) Build weak-label dataset sample

```bash
python scripts/build_dataset.py \
  --scene-dir data/sample \
  --bbox -123.2,49.1,-122.8,49.4
```

### 5) Tile dataset for training

```bash
python scripts/tile_dataset.py \
  --input-root data/processed \
  --tile-size 256 \
  --stride 256
```

### 6) Train baseline segmentation model

```bash
python scripts/train.py \
  --manifest data/processed/tiles_manifest.jsonl \
  --epochs 10 \
  --batch-size 8
```

### 7) CLI inference with cache-aware model prediction

```bash
python scripts/infer.py \
  --scene-id scene-abc123 \
  --model-checkpoint artifacts/models/weakrisk_baseline/best.pt
```

## Environment Variables

See `.env.example` for full list. Core settings include:
- app runtime (`APP_NAME`, `ENVIRONMENT`, `HOST`, `PORT`)
- storage and registry paths
- cache cap and clipping controls (`CACHE_MAX_SIZE_GB`, `CLIPPED_CACHE_MAX_DIMENSION`)
- migration path file path
- NDWI and heatmap thresholds
- water detector mode and gates (`WATER_DETECTOR_MODE`, `WATER_NIR_TO_GREEN_RATIO_MAX`, `WATER_NDVI_MAX`)
- optional pretrained water model repo id (`PRETRAINED_WATER_MODEL_REPO_ID`)
- Sentinel API placeholders (stub integration)
- Prithvi model name/flags (scaffold)

## Local Data Assumptions

- Local ingestion is the primary MVP path.
- Input assets are GeoTIFF bands with keys mappable to `B3`, `B4`, `B5`, `B8`.
- Bands should cover the same area; mismatch is handled with reprojection to a reference grid.
- For pretrained water detection, provide additional Sentinel bands `B2`, `B11`, and `B12`.

## Testing

Run:

```bash
pytest -q
```

Tests include:
- health endpoint
- index behavior
- risk normalization
- migration path loading
- ingestion + processing API smoke test with synthetic GeoTIFFs

## Limitations / TODOs

- Remote Sentinel ingestion (`provider=sentinel`) is scaffolded, not production-integrated.
- Temperature proxy is a stub (`0.0`) to keep MVP deterministic and lightweight.
- Prithvi training endpoint is scaffolded; script-level `--model prithvi-head` currently falls back to baseline until backbone integration is completed.
- `/risk/tiles` returns GeoJSON grid features (not XYZ tile server), which is intentional for fast frontend integration.
- Pretrained water detection requires optional dependency `geoai-py` and 6-band Sentinel input (`B2,B3,B4,B8,B11,B12`).
- Metadata cache uses JSON files for hackathon simplicity; SQLite is an easy future swap if concurrency grows.
