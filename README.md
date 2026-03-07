# Salmon Water Risk Backend (Hackathon MVP)

FastAPI backend MVP for detecting water-related environmental risk proxies along salmon migratory paths using Sentinel-2 style inputs.

This repository is optimized for an end-to-end local demo:
- local scene ingestion (fully implemented)
- raster preprocessing and spectral index calculations
- water-only risk scoring (`0..1`)
- heatmap-ready GeoJSON grid output
- migration-path-aware risk summaries
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
- Water-only risk scoring:
  - `risk_raw = 0.5*chlorophyll + 0.3*turbidity + 0.2*temperature_proxy`
  - `temperature_proxy` is currently a stub (`0.0`) with explicit TODO
  - normalized risk in `0..1` on water pixels
- GeoJSON grid aggregation with risk category metadata for frontend heatmaps
- Migration path loading from static GeoJSON + buffered intersection summary
- Artifact persistence (GeoTIFF + JSON + GeoJSON) and local registry tracking

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
      prithvi.py
    storage/
      metadata_store.py
  utils/
    bands.py
    geospatial.py
data/
  sample/
  migration_paths/
artifacts/
tests/
```

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

3. Copy env defaults:

```bash
cp .env.example .env
```

4. Run API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health`
- `GET /migration-paths`
- `POST /ingest/sentinel`
- `POST /process/scene`
- `POST /risk/score`
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

## Environment Variables

See `.env.example` for full list. Core settings include:
- app runtime (`APP_NAME`, `ENVIRONMENT`, `HOST`, `PORT`)
- storage and registry paths
- migration path file path
- NDWI and heatmap thresholds
- Sentinel API placeholders (stub integration)
- Prithvi model name/flags (scaffold)

## Local Data Assumptions

- Local ingestion is the primary MVP path.
- Input assets are GeoTIFF bands with keys mappable to `B3`, `B4`, `B5`, `B8`.
- Bands should cover the same area; mismatch is handled with reprojection to a reference grid.

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
- Prithvi training endpoints track/scaffold jobs; real fine-tuning is not implemented here.
- `/risk/tiles` returns GeoJSON grid features (not XYZ tile server), which is intentional for fast frontend integration.
