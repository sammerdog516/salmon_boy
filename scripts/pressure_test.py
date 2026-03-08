from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CallResult:
    ok: bool
    status_code: int
    latency_ms: float
    cache_hit: bool = False
    feature_count: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pressure test /risk/predict and /risk/tiles model path."
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--grid-block-size", type=int, default=32)
    parser.add_argument("--skip-tiles", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def predict_once(
    client: httpx.AsyncClient,
    base_url: str,
    scene_id: str,
    model_checkpoint: str | None,
    grid_block_size: int,
    force_recompute: bool,
) -> CallResult:
    payload: dict[str, Any] = {
        "scene_id": scene_id,
        "include_grid": True,
        "grid_block_size": grid_block_size,
        "force_recompute": force_recompute,
    }
    if model_checkpoint:
        payload["model_checkpoint"] = model_checkpoint

    start = time.perf_counter()
    try:
        response = await client.post(f"{base_url}/risk/predict", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if response.status_code != 200:
            return CallResult(
                ok=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
                error=response.text[:500],
            )
        body = response.json()
        return CallResult(
            ok=True,
            status_code=200,
            latency_ms=latency_ms,
            cache_hit=bool(body.get("cache_hit", False)),
            feature_count=len(body.get("grid", {}).get("features", []))
            if isinstance(body.get("grid"), dict)
            else 0,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return CallResult(
            ok=False,
            status_code=0,
            latency_ms=latency_ms,
            error=str(exc),
        )


async def predict_and_fetch_tiles(
    client: httpx.AsyncClient,
    base_url: str,
    scene_id: str,
    model_checkpoint: str | None,
    grid_block_size: int,
    force_recompute: bool,
    skip_tiles: bool,
) -> CallResult:
    payload: dict[str, Any] = {
        "scene_id": scene_id,
        "include_grid": True,
        "grid_block_size": grid_block_size,
        "force_recompute": force_recompute,
    }
    if model_checkpoint:
        payload["model_checkpoint"] = model_checkpoint

    start = time.perf_counter()
    try:
        predict_resp = await client.post(f"{base_url}/risk/predict", json=payload)
        if predict_resp.status_code != 200:
            return CallResult(
                ok=False,
                status_code=predict_resp.status_code,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error=predict_resp.text[:500],
            )
        predict_body = predict_resp.json()
        cache_hit = bool(predict_body.get("cache_hit", False))
        prediction_id = predict_body.get("prediction_id")
        if not skip_tiles:
            tiles_resp = await client.get(
                f"{base_url}/risk/tiles",
                params={"source": "model", "prediction_id": prediction_id},
            )
            if tiles_resp.status_code != 200:
                return CallResult(
                    ok=False,
                    status_code=tiles_resp.status_code,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    cache_hit=cache_hit,
                    error=tiles_resp.text[:500],
                )
            tiles_body = tiles_resp.json()
            features = tiles_body.get("features", {}).get("features", [])
            if not isinstance(features, list):
                return CallResult(
                    ok=False,
                    status_code=200,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    cache_hit=cache_hit,
                    error="Invalid tiles payload shape: missing FeatureCollection.features list",
                )
            feature_count = len(features)
        else:
            feature_count = len(predict_body.get("grid", {}).get("features", []))
        return CallResult(
            ok=True,
            status_code=200,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            cache_hit=cache_hit,
            feature_count=feature_count,
        )
    except Exception as exc:
        return CallResult(
            ok=False,
            status_code=0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=str(exc),
        )


async def run_pressure(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout=args.timeout_sec)
    limits = httpx.Limits(max_connections=max(10, args.concurrency * 2))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(f"{args.base_url}/health")
        if health.status_code != 200:
            raise RuntimeError(
                f"Health check failed ({health.status_code}): {health.text[:500]}"
            )

        warmup = await predict_once(
            client=client,
            base_url=args.base_url,
            scene_id=args.scene_id,
            model_checkpoint=args.model_checkpoint,
            grid_block_size=args.grid_block_size,
            force_recompute=args.force_recompute,
        )
        if not warmup.ok:
            raise RuntimeError(f"Warm-up predict failed: {warmup.error}")

        semaphore = asyncio.Semaphore(args.concurrency)

        async def worker() -> CallResult:
            async with semaphore:
                return await predict_and_fetch_tiles(
                    client=client,
                    base_url=args.base_url,
                    scene_id=args.scene_id,
                    model_checkpoint=args.model_checkpoint,
                    grid_block_size=args.grid_block_size,
                    force_recompute=args.force_recompute,
                    skip_tiles=args.skip_tiles,
                )

        tasks = [asyncio.create_task(worker()) for _ in range(args.iterations)]
        results = await asyncio.gather(*tasks)

    latencies = sorted([r.latency_ms for r in results])
    success = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]
    cache_hits = sum(1 for r in success if r.cache_hit)
    feature_counts = [r.feature_count for r in success]

    summary = {
        "base_url": args.base_url,
        "scene_id": args.scene_id,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "skip_tiles": args.skip_tiles,
        "success_count": len(success),
        "failure_count": len(failures),
        "success_rate": (len(success) / len(results)) if results else 0.0,
        "cache_hit_rate_on_success": (cache_hits / len(success)) if success else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50) if latencies else 0.0,
            "p95": percentile(latencies, 0.95) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "feature_count": {
            "min": min(feature_counts) if feature_counts else 0,
            "max": max(feature_counts) if feature_counts else 0,
            "mean": statistics.fmean(feature_counts) if feature_counts else 0.0,
        },
        "sample_errors": [f.error for f in failures[:5]],
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_pressure(args))
    output = json.dumps(summary, indent=2)
    print(output)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()

