#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.dinov3_aux import (  # noqa: E402
    DinoV3AuxConfig,
    DinoV3KeypointAuxHead,
    build_point_targets,
    build_triplet_targets,
    load_dinov3_extractor,
    normalize_for_dinov3,
    sample_point_features,
)
from yoloposevf.run_archive import write_run_metadata  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class YoloPoseAuxDataset(Dataset):
    def __init__(self, dataset_yaml: Path, split: str, imgsz: int) -> None:
        self.dataset_yaml = dataset_yaml
        self.imgsz = int(imgsz)
        values = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        dataset_root = Path(values.get("path", dataset_yaml.parent))
        if not dataset_root.is_absolute():
            dataset_root = (dataset_yaml.parent / dataset_root).resolve()
        image_dir = dataset_root / values[split]
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Image split not found: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label split not found: {label_dir}")
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        self.records: list[tuple[Path, Path]] = []
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = labels.get(image_path.stem)
            if label_path is not None:
                self.records.append((image_path, label_path))
        if not self.records:
            raise ValueError(f"No image/label records found for split={split}: {dataset_yaml}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.records[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor, scale, pad_x, pad_y = letterbox_tensor(image, self.imgsz)
        keypoints = torch.zeros((3, 2), dtype=torch.float32)
        keypoint_mask = torch.zeros((3,), dtype=torch.bool)
        has_pose = label_path.stat().st_size > 0
        if has_pose:
            line = label_path.read_text(encoding="utf-8").strip().splitlines()[0].split()
            values = [float(item) for item in line]
            raw_keypoints = values[5:14]
            for kp_index in range(3):
                x_norm, y_norm, visibility = raw_keypoints[kp_index * 3 : kp_index * 3 + 3]
                x_px = x_norm * width * scale + pad_x
                y_px = y_norm * height * scale + pad_y
                keypoints[kp_index] = torch.tensor([x_px / self.imgsz, y_px / self.imgsz])
                keypoint_mask[kp_index] = visibility > 0
        return {
            "image": image_tensor,
            "keypoints": keypoints,
            "keypoint_mask": keypoint_mask,
            "image_label": torch.tensor(0 if has_pose else 1, dtype=torch.long),
            "path": str(image_path),
        }


def letterbox_tensor(image: Image.Image, imgsz: int) -> tuple[torch.Tensor, float, float, float]:
    width, height = image.size
    scale = min(float(imgsz) / max(width, 1), float(imgsz) / max(height, 1))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = Image.new("RGB", (imgsz, imgsz), (0, 0, 0))
    resized_image = image.resize(resized, Image.Resampling.BILINEAR)
    pad_x = (imgsz - resized[0]) / 2.0
    pad_y = (imgsz - resized[1]) / 2.0
    canvas.paste(resized_image, (round(pad_x), round(pad_y)))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, scale, pad_x, pad_y


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in items]),
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "keypoint_mask": torch.stack([item["keypoint_mask"] for item in items]),
        "image_label": torch.stack([item["image_label"] for item in items]),
        "path": [item["path"] for item in items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-DINOv3 auxiliary scorer for A/L/R keypoint plausibility.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_dinov3_keypoint_aux_y11m.yaml"))
    parser.add_argument("--data", type=Path, help="Override YOLO-Pose dataset YAML.")
    parser.add_argument("--device", type=str, help="Override device.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--name", type=str, help="Override run name.")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config only.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config) or {}
    if args.data is not None:
        cfg["data"] = str(args.data)
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.name is not None:
        cfg["name"] = args.name
    cfg.setdefault("data", "data/yolo_pose/vocal_fold_pose.yaml")
    cfg.setdefault("project", "Results/dinov3_keypoint_aux")
    cfg.setdefault("name", "dinov3_vits16_three_point_aux_manual")
    cfg.setdefault("imgsz", 960)
    cfg.setdefault("epochs", 30)
    cfg.setdefault("batch", 16)
    cfg.setdefault("workers", 4)
    cfg.setdefault("device", "0")
    cfg.setdefault("seed", 42)
    cfg.setdefault("exist_ok", False)
    cfg.setdefault("learning_rate", 1e-3)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("dinov3", {})
    dinov3 = dict(cfg["dinov3"])
    defaults = asdict(DinoV3AuxConfig())
    for key, value in defaults.items():
        dinov3.setdefault(key, value)
    cfg["dinov3"] = dinov3
    return cfg


def device_from_arg(value: str) -> torch.device:
    if value.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def run_epoch(
    *,
    extractor: Any,
    head: DinoV3KeypointAuxHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    aux_cfg: DinoV3AuxConfig,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    totals = {"loss": 0.0, "point": 0.0, "triplet": 0.0, "image": 0.0, "batches": 0.0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        keypoint_mask = batch["keypoint_mask"].to(device, non_blocking=True)
        image_label = batch["image_label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            feature_map, global_feature = extractor.forward_dense(normalize_for_dinov3(images, aux_cfg))

        point_targets, point_labels, point_valid = build_point_targets(
            keypoints,
            keypoint_mask,
            background_points=aux_cfg.background_points,
            min_background_distance=aux_cfg.min_background_distance,
        )
        point_features = sample_point_features(feature_map, point_targets)
        point_logits = head.point_logits(point_features, point_targets)
        if point_valid.any():
            point_loss = F.cross_entropy(point_logits[point_valid], point_labels[point_valid])
        else:
            point_loss = point_logits.new_zeros(())

        triplets, triplet_labels, triplet_owners = build_triplet_targets(
            keypoints,
            keypoint_mask,
            corrupted_triplets=aux_cfg.corrupted_triplets,
            corrupted_jitter=aux_cfg.corrupted_jitter,
        )
        if triplets.numel():
            triplet_features = sample_point_features(feature_map[triplet_owners], triplets)
            triplet_loss = F.cross_entropy(head.triplet_logits(triplet_features, triplets), triplet_labels)
        else:
            triplet_loss = point_logits.new_zeros(())

        image_loss = F.cross_entropy(head.image_logits(global_feature), image_label)
        total_loss = (
            aux_cfg.point_loss_weight * point_loss
            + aux_cfg.triplet_loss_weight * triplet_loss
            + aux_cfg.image_loss_weight * image_loss
        )
        if training:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            optimizer.step()

        totals["loss"] += float(total_loss.detach().cpu())
        totals["point"] += float(point_loss.detach().cpu())
        totals["triplet"] += float(triplet_loss.detach().cpu())
        totals["image"] += float(image_loss.detach().cpu())
        totals["batches"] += 1.0

    denom = max(totals.pop("batches"), 1.0)
    return {key: value / denom for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = effective_config(args)
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    cfg["data"] = str(data_path)
    project = Path(cfg["project"])
    if not project.is_absolute():
        project = PROJECT_ROOT / project
    run_dir = project / str(cfg["name"])
    cfg["run_dir"] = str(run_dir)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    if run_dir.exists() and not bool(cfg["exist_ok"]):
        raise SystemExit(f"Run folder already exists: {run_dir}")

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device_from_arg(str(cfg["device"]))
    aux_cfg = DinoV3AuxConfig(**cfg["dinov3"])

    extractor = load_dinov3_extractor(aux_cfg, device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    train_dataset = YoloPoseAuxDataset(data_path, "train", int(cfg["imgsz"]))
    val_dataset = YoloPoseAuxDataset(data_path, "val", int(cfg["imgsz"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch"]),
        shuffle=True,
        num_workers=int(cfg["workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    with torch.no_grad():
        probe = torch.zeros((1, 3, int(cfg["imgsz"]), int(cfg["imgsz"])), device=device)
        feature_map, _ = extractor.forward_dense(normalize_for_dinov3(probe, aux_cfg))
    feature_dim = int(feature_map.shape[1])
    head = DinoV3KeypointAuxHead(
        feature_dim,
        point_hidden_dim=aux_cfg.point_hidden_dim,
        triplet_hidden_dim=aux_cfg.triplet_hidden_dim,
        image_hidden_dim=aux_cfg.image_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "aux_history.jsonl"
    write_run_metadata(
        run_dir=run_dir,
        project_root=PROJECT_ROOT,
        command=sys.argv,
        config=cfg,
        extra={
            "feature_dim": feature_dim,
            "dataset_summary": {
                "train_records": len(train_dataset),
                "train_positive": sum(path.stat().st_size > 0 for _, path in train_dataset.records),
                "val_records": len(val_dataset),
                "val_positive": sum(path.stat().st_size > 0 for _, path in val_dataset.records),
            },
        },
    )

    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            extractor=extractor,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            aux_cfg=aux_cfg,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                extractor=extractor,
                head=head,
                loader=val_loader,
                optimizer=None,
                device=device,
                aux_cfg=aux_cfg,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_triplet={val_metrics['triplet']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "head": head.state_dict(),
                    "feature_dim": feature_dim,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "config": cfg,
                },
                weights_dir / "best_aux_head.pt",
            )
    torch.save(
        {
            "head": head.state_dict(),
            "feature_dim": feature_dim,
            "epoch": int(cfg["epochs"]),
            "val_loss": val_metrics["loss"],
            "config": cfg,
        },
        weights_dir / "last_aux_head.pt",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_checkpoint": str(weights_dir / "best_aux_head.pt"),
        "last_checkpoint": str(weights_dir / "last_aux_head.pt"),
    }
    (run_dir / "aux_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
