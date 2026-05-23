#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
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

from yoloposevf.keypoint_contrast import (  # noqa: E402
    GlobalProjectionHead,
    OrientedPatchProjectionHead,
    RejectClassificationHead,
    make_light_augmented_view,
    oriented_keypoint_patch_contrast_loss,
    supervised_image_contrast_loss,
)
from yoloposevf.run_archive import write_run_metadata  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetRecord:
    image_path: Path
    label_path: Path
    has_keypoints: bool


class YoloPoseContrastDataset(Dataset):
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
        self.records: list[DatasetRecord] = []
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = labels.get(image_path.stem)
            if label_path is None:
                continue
            self.records.append(
                DatasetRecord(
                    image_path=image_path,
                    label_path=label_path,
                    has_keypoints=label_path.stat().st_size > 0,
                )
            )
        if not self.records:
            raise ValueError(f"No image/label records found for split={split}: {dataset_yaml}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        image_tensor, scale, pad_x, pad_y = _letterbox_tensor(image, self.imgsz)
        keypoints = torch.zeros((3, 2), dtype=torch.float32)
        keypoint_mask = torch.zeros((3,), dtype=torch.bool)
        reject_label = torch.tensor(0 if record.has_keypoints else 1, dtype=torch.long)
        if record.has_keypoints:
            line = record.label_path.read_text(encoding="utf-8").strip().splitlines()[0].split()
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
            "reject_label": reject_label,
            "path": str(record.image_path),
        }


def _letterbox_tensor(image: Image.Image, imgsz: int) -> tuple[torch.Tensor, float, float, float]:
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
        "reject_label": torch.stack([item["reject_label"] for item in items]),
        "path": [item["path"] for item in items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain YOLO-Pose backbone/neck with anatomy-oriented keypoint contrast."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain_oriented_keypoint_contrast_manual_only_y11m.yaml"),
    )
    parser.add_argument("--model", type=str, help="Override initial YOLO-Pose checkpoint.")
    parser.add_argument("--data", type=Path, help="Override YOLO-Pose dataset YAML.")
    parser.add_argument("--device", type=str, help="Override device.")
    parser.add_argument("--name", type=str, help="Override run name.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config only.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config) or {}
    if args.model is not None:
        cfg["model"] = args.model
    if args.data is not None:
        cfg["data"] = str(args.data)
    if args.device is not None:
        cfg["device"] = args.device
    if args.name is not None:
        cfg["name"] = args.name
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg.setdefault("model", "yolo11m-pose.pt")
    cfg.setdefault("data", "data/yolo_pose_mixed_negative_60/vocal_fold_pose.yaml")
    cfg.setdefault("project", "Results/oriented_keypoint_contrast_pretrain")
    cfg.setdefault("name", "yolo11m_manual_oriented_kp_contrast")
    cfg.setdefault("imgsz", 960)
    cfg.setdefault("epochs", 40)
    cfg.setdefault("batch", 8)
    cfg.setdefault("workers", 4)
    cfg.setdefault("device", "0")
    cfg.setdefault("seed", 42)
    cfg.setdefault("exist_ok", False)
    cfg.setdefault("feature_index", 0)
    cfg.setdefault("learning_rate", 1e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("local_contrast_weight", 1.0)
    cfg.setdefault("image_reject_weight", 0.15)
    cfg.setdefault("image_reject_ce_weight", 0.5)
    cfg.setdefault("patch_size_input", 48.0)
    cfg.setdefault("patch_output_size", 5)
    cfg.setdefault("temperature", 0.10)
    cfg.setdefault("negative_count", 12)
    cfg.setdefault("negative_min_distance", 0.12)
    cfg.setdefault("augment_degrees", 4.0)
    cfg.setdefault("augment_scale", 0.06)
    cfg.setdefault("augment_translate", 0.025)
    cfg.setdefault("augment_brightness", 0.08)
    cfg.setdefault("augment_contrast", 0.12)
    return cfg


def feature_maps_from_model(model: torch.nn.Module, images: torch.Tensor, feature_layers: list[int]) -> list[torch.Tensor]:
    saved: list[Any] = []
    features: list[torch.Tensor] = []
    x: Any = images
    max_layer = max(feature_layers)
    wanted = set(feature_layers)
    for module in model.model:
        if module.f != -1:
            x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
        x = module(x)
        saved.append(x if module.i in model.save else None)
        if module.i in wanted:
            features.append(x)
        if module.i >= max_layer:
            break
    return features


def run_epoch(
    *,
    model: torch.nn.Module,
    local_head: OrientedPatchProjectionHead,
    global_head: GlobalProjectionHead,
    reject_head: RejectClassificationHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    local_head.train(training)
    global_head.train(training)
    reject_head.train(training)
    totals = {"loss": 0.0, "local": 0.0, "image": 0.0, "ce": 0.0, "batches": 0.0}

    feature_layers = [16, 19, 22]
    feature_index = int(cfg["feature_index"])
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        keypoint_mask = batch["keypoint_mask"].to(device, non_blocking=True)
        reject_label = batch["reject_label"].to(device, non_blocking=True)
        images2, keypoints2, mask2 = make_light_augmented_view(
            images,
            keypoints,
            keypoint_mask,
            degrees=float(cfg["augment_degrees"]),
            scale=float(cfg["augment_scale"]),
            translate=float(cfg["augment_translate"]),
            brightness=float(cfg["augment_brightness"]),
            contrast=float(cfg["augment_contrast"]),
        )
        valid_mask = keypoint_mask & mask2
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            features1 = feature_maps_from_model(model, images, feature_layers)[feature_index]
            features2 = feature_maps_from_model(model, images2, feature_layers)[feature_index]
            local_loss = oriented_keypoint_patch_contrast_loss(
                features1,
                features2,
                keypoints,
                keypoints2,
                valid_mask,
                local_head,
                temperature=float(cfg["temperature"]),
                patch_size_input=cfg["patch_size_input"],
                input_size=int(cfg["imgsz"]),
                output_size=int(cfg["patch_output_size"]),
                negative_count=int(cfg["negative_count"]),
                negative_min_distance=float(cfg["negative_min_distance"]),
            )
            global1 = F.adaptive_avg_pool2d(features1, (1, 1)).flatten(1)
            global2 = F.adaptive_avg_pool2d(features2, (1, 1)).flatten(1)
            z1 = global_head(global1)
            z2 = global_head(global2)
            image_loss = supervised_image_contrast_loss(z1, z2, reject_label, temperature=float(cfg["temperature"]))
            ce_loss = 0.5 * (
                F.cross_entropy(reject_head(global1), reject_label)
                + F.cross_entropy(reject_head(global2), reject_label)
            )
            total_loss = (
                float(cfg["local_contrast_weight"]) * local_loss
                + float(cfg["image_reject_weight"])
                * (image_loss + float(cfg["image_reject_ce_weight"]) * ce_loss)
            )
            if training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters())
                    + list(local_head.parameters())
                    + list(global_head.parameters())
                    + list(reject_head.parameters()),
                    10.0,
                )
                optimizer.step()

        totals["loss"] += float(total_loss.detach().cpu())
        totals["local"] += float(local_loss.detach().cpu())
        totals["image"] += float(image_loss.detach().cpu())
        totals["ce"] += float(ce_loss.detach().cpu())
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
    if run_dir.exists() and not bool(cfg["exist_ok"]):
        raise SystemExit(f"Run folder already exists: {run_dir}")
    if args.dry_run:
        print(json.dumps({**cfg, "run_dir": str(run_dir)}, indent=2, ensure_ascii=False))
        return

    from ultralytics import YOLO

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{cfg['device']}" if str(cfg["device"]).isdigit() and torch.cuda.is_available() else cfg["device"])

    train_dataset = YoloPoseContrastDataset(data_path, "train", int(cfg["imgsz"]))
    val_dataset = YoloPoseContrastDataset(data_path, "val", int(cfg["imgsz"]))
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

    yolo = YOLO(str(cfg["model"]))
    model = yolo.model.to(device)
    model.float()
    feature_layers = [16, 19, 22]
    with torch.no_grad():
        probe = torch.zeros((1, 3, int(cfg["imgsz"]), int(cfg["imgsz"])), device=device)
        feature_channels = feature_maps_from_model(model, probe, feature_layers)[int(cfg["feature_index"])].shape[1]

    local_head = OrientedPatchProjectionHead(
        feature_channels,
        patch_size=int(cfg["patch_output_size"]),
        hidden_dim=int(cfg.get("local_projection_hidden_dim", 512)),
        out_dim=int(cfg.get("local_projection_dim", 128)),
    ).to(device)
    global_head = GlobalProjectionHead(
        feature_channels,
        hidden_dim=int(cfg.get("global_projection_hidden_dim", 256)),
        out_dim=int(cfg.get("global_projection_dim", 128)),
    ).to(device)
    reject_head = RejectClassificationHead(
        feature_channels,
        hidden_dim=int(cfg.get("reject_hidden_dim", 128)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(local_head.parameters()) + list(global_head.parameters()) + list(reject_head.parameters()),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "pretrain_history.jsonl"
    write_run_metadata(
        run_dir=run_dir,
        project_root=PROJECT_ROOT,
        command=sys.argv,
        config=cfg,
        extra={
            "dataset_summary": {
                "train_records": len(train_dataset),
                "train_positive": sum(record.has_keypoints for record in train_dataset.records),
                "train_mixed_empty_label": sum(not record.has_keypoints for record in train_dataset.records),
                "val_records": len(val_dataset),
            }
        },
    )

    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model=model,
            local_head=local_head,
            global_head=global_head,
            reject_head=reject_head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                local_head=local_head,
                global_head=global_head,
                reject_head=reject_head,
                loader=val_loader,
                optimizer=None,
                device=device,
                cfg=cfg,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_local={val_metrics['local']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            yolo.model = copy.deepcopy(model).to("cpu")
            yolo.save(weights_dir / "best.pt")
            torch.save(
                {
                    "local_head": local_head.state_dict(),
                    "global_head": global_head.state_dict(),
                    "reject_head": reject_head.state_dict(),
                    "feature_channels": feature_channels,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "config": cfg,
                },
                weights_dir / "projection_heads_best.pt",
            )
            yolo.model = model

    yolo.model = copy.deepcopy(model).to("cpu")
    yolo.save(weights_dir / "last.pt")
    yolo.model = model
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_checkpoint": str(weights_dir / "best.pt"),
        "last_checkpoint": str(weights_dir / "last.pt"),
    }
    (run_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
