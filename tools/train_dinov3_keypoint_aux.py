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
    foreground_mask_from_images,
    load_dinov3_extractor,
    normalize_for_dinov3,
    sample_oriented_point_regions,
    sample_oriented_point_region_masks,
)
from yoloposevf.preprocess import crop_existing_black_borders  # noqa: E402
from yoloposevf.run_archive import write_run_metadata  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class YoloPoseAuxDataset(Dataset):
    def __init__(
        self,
        dataset_yaml: Path,
        split: str,
        imgsz: int,
        *,
        hard_negative_predictions: list[Path] | None = None,
        hard_negative_points: int = 0,
        hard_negative_min_confidence: float = 0.30,
        hard_negative_min_distance: float = 0.08,
        crop_black_border_luma_floor: float = 8.0,
    ) -> None:
        self.dataset_yaml = dataset_yaml
        self.imgsz = int(imgsz)
        self.hard_negative_points = int(hard_negative_points)
        self.hard_negative_min_distance = float(hard_negative_min_distance)
        self.crop_black_border_luma_floor = float(crop_black_border_luma_floor)
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
        self.hard_negatives = load_hard_negative_predictions(
            hard_negative_predictions or [],
            min_confidence=float(hard_negative_min_confidence),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.records[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        cropped_image, crop_bbox = crop_existing_black_borders(
            image,
            luma_floor=self.crop_black_border_luma_floor,
        )
        crop_left, crop_top, _, _ = crop_bbox
        image_tensor, scale, pad_x, pad_y, valid_mask = letterbox_tensor_and_valid_mask(cropped_image, self.imgsz)
        keypoints = torch.zeros((3, 2), dtype=torch.float32)
        keypoint_mask = torch.zeros((3,), dtype=torch.bool)
        has_pose = label_path.stat().st_size > 0
        if has_pose:
            line = label_path.read_text(encoding="utf-8").strip().splitlines()[0].split()
            values = [float(item) for item in line]
            raw_keypoints = values[5:14]
            for kp_index in range(3):
                x_norm, y_norm, visibility = raw_keypoints[kp_index * 3 : kp_index * 3 + 3]
                x_px = (x_norm * original_width - crop_left) * scale + pad_x
                y_px = (y_norm * original_height - crop_top) * scale + pad_y
                keypoints[kp_index] = torch.tensor([x_px / self.imgsz, y_px / self.imgsz])
                keypoint_mask[kp_index] = visibility > 0
        hard_points, hard_mask, hard_dirs = self._hard_negative_tensors(
            image_path,
            crop_left,
            crop_top,
            scale,
            pad_x,
            pad_y,
            keypoints,
            keypoint_mask,
        )
        return {
            "image": image_tensor,
            "valid_mask": valid_mask,
            "keypoints": keypoints,
            "keypoint_mask": keypoint_mask,
            "hard_negative_points": hard_points,
            "hard_negative_mask": hard_mask,
            "hard_negative_directions": hard_dirs,
            "image_label": torch.tensor(0 if has_pose else 1, dtype=torch.long),
            "path": str(image_path),
        }

    def _hard_negative_tensors(
        self,
        image_path: Path,
        crop_left: int,
        crop_top: int,
        scale: float,
        pad_x: float,
        pad_y: float,
        keypoints: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.zeros((self.hard_negative_points, 2), dtype=torch.float32)
        dirs = torch.zeros((self.hard_negative_points, 2), dtype=torch.float32)
        dirs[:, 1] = 1.0
        mask = torch.zeros((self.hard_negative_points,), dtype=torch.bool)
        if self.hard_negative_points <= 0:
            return points, mask, dirs
        visible = keypoints[keypoint_mask]
        used = 0
        for item in self.hard_negatives.get(image_path.stem, []):
            if item.get("coordinate_space") == "cropped":
                x_px = float(item["x"]) * scale + pad_x
                y_px = float(item["y"]) * scale + pad_y
            else:
                x_px = (float(item["x"]) - float(crop_left)) * scale + pad_x
                y_px = (float(item["y"]) - float(crop_top)) * scale + pad_y
            point = torch.tensor([x_px / self.imgsz, y_px / self.imgsz], dtype=torch.float32).clamp(0.0, 1.0)
            if visible.numel():
                distance = (visible - point).square().sum(dim=1).sqrt().min()
                if float(distance) < self.hard_negative_min_distance:
                    continue
            points[used] = point
            dirs[used] = torch.tensor([float(item["dx"]), float(item["dy"])], dtype=torch.float32)
            mask[used] = True
            used += 1
            if used >= self.hard_negative_points:
                break
        return points, mask, dirs


def letterbox_geometry(width: int, height: int, imgsz: int) -> tuple[float, tuple[int, int], float, float, int, int]:
    scale = min(float(imgsz) / max(width, 1), float(imgsz) / max(height, 1))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    pad_x = (imgsz - resized[0]) / 2.0
    pad_y = (imgsz - resized[1]) / 2.0
    return scale, resized, pad_x, pad_y, round(pad_x), round(pad_y)


def letterbox_tensor(image: Image.Image, imgsz: int) -> tuple[torch.Tensor, float, float, float]:
    width, height = image.size
    scale, resized, pad_x, pad_y, paste_x, paste_y = letterbox_geometry(width, height, imgsz)
    canvas = Image.new("RGB", (imgsz, imgsz), (0, 0, 0))
    resized_image = image.resize(resized, Image.Resampling.BILINEAR)
    canvas.paste(resized_image, (paste_x, paste_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, scale, pad_x, pad_y


def letterbox_tensor_and_valid_mask(image: Image.Image, imgsz: int) -> tuple[torch.Tensor, float, float, float, torch.Tensor]:
    width, height = image.size
    scale, resized, pad_x, pad_y, paste_x, paste_y = letterbox_geometry(width, height, imgsz)
    tensor, _, _, _ = letterbox_tensor(image, imgsz)
    valid_mask = torch.zeros((1, int(imgsz), int(imgsz)), dtype=torch.float32)
    x2 = min(paste_x + resized[0], int(imgsz))
    y2 = min(paste_y + resized[1], int(imgsz))
    valid_mask[:, max(paste_y, 0):y2, max(paste_x, 0):x2] = 1.0
    return tensor, scale, pad_x, pad_y, valid_mask


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in items]),
        "valid_mask": torch.stack([item["valid_mask"] for item in items]),
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "keypoint_mask": torch.stack([item["keypoint_mask"] for item in items]),
        "hard_negative_points": torch.stack([item["hard_negative_points"] for item in items]),
        "hard_negative_mask": torch.stack([item["hard_negative_mask"] for item in items]),
        "hard_negative_directions": torch.stack([item["hard_negative_directions"] for item in items]),
        "image_label": torch.stack([item["image_label"] for item in items]),
        "path": [item["path"] for item in items],
    }


def load_hard_negative_predictions(paths: list[Path], *, min_confidence: float) -> dict[str, list[dict[str, Any]]]:
    rows_by_stem: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Hard-negative prediction JSONL not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            keypoints = record.get("keypoints") or []
            if len(keypoints) < 3:
                continue
            coords = []
            for row in keypoints[:3]:
                if len(row) < 3 or float(row[2]) < float(min_confidence):
                    continue
                coords.append((float(row[0]), float(row[1]), float(row[2])))
            if not coords:
                continue
            direction = _prediction_direction(keypoints)
            padding_px = _prediction_padding_px(record)
            coordinate_space = "cropped" if padding_px > 0.0 else "image"
            stems = {
                Path(str(record[field])).stem
                for field in ("dinov3_source", "cropped_source", "source", "original_source")
                if record.get(field)
            }
            for stem in stems:
                rows = rows_by_stem.setdefault(stem, [])
                for x, y, conf in coords:
                    rows.append(
                        {
                            "x": x - padding_px if coordinate_space == "cropped" else x,
                            "y": y - padding_px if coordinate_space == "cropped" else y,
                            "conf": conf,
                            "dx": direction[0],
                            "dy": direction[1],
                            "coordinate_space": coordinate_space,
                        }
                    )
    return rows_by_stem


def _prediction_padding_px(record: dict[str, Any]) -> float:
    preprocess = record.get("preprocess") or {}
    return float(preprocess.get("padding_px") or 0.0)


def _prediction_direction(keypoints: list[list[float]]) -> tuple[float, float]:
    if len(keypoints) < 3:
        return 0.0, 1.0
    ax, ay = float(keypoints[0][0]), float(keypoints[0][1])
    lx, ly = float(keypoints[1][0]), float(keypoints[1][1])
    rx, ry = float(keypoints[2][0]), float(keypoints[2][1])
    dx = 0.5 * (lx + rx) - ax
    dy = 0.5 * (ly + ry) - ay
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    return dx / norm, dy / norm


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
    cfg.setdefault("hard_negative_predictions", [])
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
    totals = {"loss": 0.0, "point": 0.0, "batches": 0.0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        content_valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        keypoint_mask = batch["keypoint_mask"].to(device, non_blocking=True)
        hard_negative_points = batch["hard_negative_points"].to(device, non_blocking=True)
        hard_negative_mask = batch["hard_negative_mask"].to(device, non_blocking=True)
        hard_negative_directions = batch["hard_negative_directions"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            feature_map, global_feature = extractor.forward_dense(normalize_for_dinov3(images, aux_cfg))
        valid_mask_map = (
            content_valid_mask * foreground_mask_from_images(images, luma_floor=aux_cfg.valid_mask_luma_floor)
            if aux_cfg.include_valid_mask
            else None
        )

        point_targets, point_labels, point_valid, point_directions = build_point_targets(
            keypoints,
            keypoint_mask,
            background_points=aux_cfg.background_points,
            near_background_points=aux_cfg.near_background_points,
            min_background_distance=aux_cfg.min_background_distance,
            near_background_min_distance=aux_cfg.near_background_min_distance,
            near_background_max_distance=aux_cfg.near_background_max_distance,
            hard_negative_points01=hard_negative_points,
            hard_negative_mask=hard_negative_mask,
            hard_negative_directions01=hard_negative_directions,
        )
        point_features = sample_oriented_point_regions(
            feature_map,
            point_targets,
            point_directions,
            patch_size_input=aux_cfg.oriented_patch_size_input,
            input_size=int(getattr(aux_cfg, "input_size", images.shape[-1])),
            output_size=aux_cfg.oriented_patch_output_size,
        )
        point_valid_masks = (
            sample_oriented_point_region_masks(
                valid_mask_map,
                point_targets,
                point_directions,
                patch_size_input=aux_cfg.oriented_patch_size_input,
                input_size=int(getattr(aux_cfg, "input_size", images.shape[-1])),
                output_size=aux_cfg.oriented_patch_output_size,
            )
            if valid_mask_map is not None
            else None
        )
        point_logits = head.point_logits(point_features, point_targets, valid_mask=point_valid_masks)
        if point_valid.any():
            point_loss = F.cross_entropy(point_logits[point_valid], point_labels[point_valid])
        else:
            point_loss = point_logits.new_zeros(())

        del global_feature
        total_loss = aux_cfg.point_loss_weight * point_loss
        if training:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            optimizer.step()

        totals["loss"] += float(total_loss.detach().cpu())
        totals["point"] += float(point_loss.detach().cpu())
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

    hard_negative_predictions = [
        (PROJECT_ROOT / item if not Path(item).is_absolute() else Path(item))
        for item in cfg.get("hard_negative_predictions", [])
    ]
    train_dataset = YoloPoseAuxDataset(
        data_path,
        "train",
        int(cfg["imgsz"]),
        hard_negative_predictions=hard_negative_predictions,
        hard_negative_points=aux_cfg.hard_negative_points,
        hard_negative_min_confidence=aux_cfg.hard_negative_min_confidence,
        hard_negative_min_distance=aux_cfg.hard_negative_min_distance,
        crop_black_border_luma_floor=aux_cfg.crop_black_border_luma_floor,
    )
    val_dataset = YoloPoseAuxDataset(
        data_path,
        "val",
        int(cfg["imgsz"]),
        crop_black_border_luma_floor=aux_cfg.crop_black_border_luma_floor,
    )
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
        patch_output_size=aux_cfg.oriented_patch_output_size,
        point_hidden_dim=aux_cfg.point_hidden_dim,
        include_coordinates=aux_cfg.include_point_coordinates,
        include_valid_mask=aux_cfg.include_valid_mask,
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
                "train_hard_negative_records": sum(bool(train_dataset.hard_negatives.get(path.stem)) for path, _ in train_dataset.records),
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
            f"val_loss={val_metrics['loss']:.4f} val_point={val_metrics['point']:.4f}",
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
