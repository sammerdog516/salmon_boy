from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for scripts/train.py. Install with: python -m pip install torch"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weakly supervised risk segmentation model.")
    parser.add_argument("--manifest", type=str, default="data/processed/tiles_manifest.jsonl")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "prithvi-head"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="artifacts/models")
    parser.add_argument("--run-name", type=str, default="weakrisk_baseline")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"Manifest not found: {path}")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if "bands_path" not in payload or "label_path" not in payload:
            continue
        records.append(payload)
    if not records:
        raise ValueError(f"No usable records in manifest: {path}")
    return records


class TileDataset(Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        bands = np.load(str(record["bands_path"])).astype(np.float32)
        label = np.load(str(record["label_path"])).astype(np.float32)
        if bands.ndim != 3:
            raise ValueError(f"Expected bands shape (C,H,W), got {bands.shape}")
        if label.ndim != 2:
            raise ValueError(f"Expected label shape (H,W), got {label.shape}")
        x = torch.from_numpy(bands)
        y = torch.from_numpy(label[None, ...])  # 1xHxW
        return x, y


class TinySegNet(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_name: str) -> nn.Module:
    if model_name == "prithvi-head":
        # TODO: plug in ibm-nasa-geospatial/Prithvi-EO-1.0-100M backbone + frozen encoder.
        print("Prithvi backbone path is scaffolded; training baseline TinySegNet for now.")
    return TinySegNet(in_channels=4)


@dataclass
class EpochMetrics:
    loss: float
    iou: float


def iou_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    target_bin = (target >= 0.5).float()
    intersection = (preds * target_bin).sum().item()
    union = (preds + target_bin - preds * target_bin).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-6))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_iou += iou_score(logits.detach(), y)
        total_batches += 1

    if total_batches == 0:
        return EpochMetrics(loss=0.0, iou=0.0)
    return EpochMetrics(
        loss=total_loss / total_batches,
        iou=total_iou / total_batches,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    records = load_manifest(Path(args.manifest).resolve())
    train_records = [r for r in records if r.get("split") == "train"]
    val_records = [r for r in records if r.get("split") == "val"]
    if not train_records:
        raise ValueError("No train records found in manifest.")
    if not val_records:
        cutoff = max(1, int(0.2 * len(train_records)))
        val_records = train_records[:cutoff]
        train_records = train_records[cutoff:] or train_records

    train_ds = TileDataset(train_records)
    val_ds = TileDataset(val_records)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        payload = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_iou": train_metrics.iou,
            "val_loss": val_metrics.loss,
            "val_iou": val_metrics.iou,
        }
        history.append(payload)
        print(json.dumps(payload))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": payload,
        }
        torch.save(checkpoint, latest_path)
        if val_metrics.iou > best_val_iou:
            best_val_iou = val_metrics.iou
            torch.save(checkpoint, best_path)

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_val_iou": best_val_iou,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "history": history,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best_val_iou": best_val_iou}, indent=2))


if __name__ == "__main__":
    main()

