from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile weakly labeled salmon-risk scene arrays into trainable patches."
    )
    parser.add_argument("--input-root", type=str, default="data/processed")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Optional sample_id filter. Repeat for multiple values.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-water-fraction", type=float, default=0.10)
    parser.add_argument("--min-positive-fraction", type=float, default=0.01)
    parser.add_argument("--negative-keep-ratio", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_sample_meta_files(input_root: Path, sample_ids: list[str] | None) -> list[Path]:
    scenes_dir = input_root / "scenes"
    if sample_ids:
        return [scenes_dir / f"{sample_id}_meta.json" for sample_id in sample_ids]
    return sorted(scenes_dir.glob("sample_*_meta.json"))


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    tiles_dir = input_root / "tiles"
    labels_tiles_dir = input_root / "labels" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_tiles_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=args.seed)
    meta_files = load_sample_meta_files(input_root, args.sample_id)
    if not meta_files:
        raise ValueError(f"No sample metadata files found under {input_root / 'scenes'}")

    manifest_path = input_root / "tiles_manifest.jsonl"
    records: list[dict[str, object]] = []

    for meta_path in meta_files:
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_id = str(meta["sample_id"])
        paths = meta.get("paths", {})
        bands_path = Path(paths["bands_npz"])
        risk_path = Path(paths["risk_npy"])
        water_mask_path = Path(paths["water_mask_npy"])
        binary_label_path = Path(paths["binary_label_npy"])

        if not (bands_path.exists() and risk_path.exists() and water_mask_path.exists()):
            continue

        bands_npz = np.load(bands_path)
        bands = bands_npz["bands"].astype(np.float32)
        risk = np.load(risk_path).astype(np.float32)
        water_mask = np.load(water_mask_path).astype(np.uint8)
        if binary_label_path.exists():
            binary_label = np.load(binary_label_path).astype(np.uint8)
        else:
            binary_label = ((risk >= 0.65) & (water_mask > 0)).astype(np.uint8)

        if bands.ndim != 3:
            raise ValueError(f"Expected bands shape (C,H,W), got {bands.shape} for {sample_id}")
        channels, height, width = bands.shape
        if channels != 4:
            raise ValueError(
                f"Expected 4 channels (B3,B4,B5,B8), got {channels} for {sample_id}"
            )

        tile_index = 0
        for row in range(0, max(1, height - args.tile_size + 1), args.stride):
            for col in range(0, max(1, width - args.tile_size + 1), args.stride):
                row_end = row + args.tile_size
                col_end = col + args.tile_size
                if row_end > height or col_end > width:
                    continue

                bands_tile = bands[:, row:row_end, col:col_end]
                water_tile = water_mask[row:row_end, col:col_end]
                label_tile = binary_label[row:row_end, col:col_end]

                water_fraction = float(water_tile.mean())
                if water_fraction < args.min_water_fraction:
                    continue

                positive_fraction = float(label_tile.mean())
                keep = (
                    positive_fraction >= args.min_positive_fraction
                    or rng.random() < args.negative_keep_ratio
                )
                if not keep:
                    continue

                tile_id = f"{sample_id}_r{row:05d}_c{col:05d}_{tile_index:05d}"
                split = "val" if rng.random() < args.val_fraction else "train"
                tile_bands_path = tiles_dir / f"{tile_id}_bands.npy"
                tile_label_path = labels_tiles_dir / f"{tile_id}_label.npy"

                np.save(tile_bands_path, bands_tile.astype(np.float32))
                np.save(tile_label_path, label_tile.astype(np.uint8))

                records.append(
                    {
                        "tile_id": tile_id,
                        "sample_id": sample_id,
                        "split": split,
                        "bands_path": str(tile_bands_path),
                        "label_path": str(tile_label_path),
                        "tile_size": int(args.tile_size),
                        "row": int(row),
                        "col": int(col),
                        "water_fraction": water_fraction,
                        "positive_fraction": positive_fraction,
                    }
                )
                tile_index += 1

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "manifest": str(manifest_path),
        "tile_count": len(records),
        "train_count": sum(1 for r in records if r["split"] == "train"),
        "val_count": sum(1 for r in records if r["split"] == "val"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

