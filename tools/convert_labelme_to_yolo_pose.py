#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize, clip_bbox, keypoint_bbox, points_bbox, union_bbox
from yoloposevf.yolo_labels import YoloPoseLabel


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class AnnotationConfig:
    bbox_label: str = "声门区域"
    point_labels: tuple[str, ...] = ("前联合", "左后方中点", "右后方中点")
    class_id: int = 0
    class_name: str = "glottic_roi"
    visibility: int = 2
    correction_margin_x: float = 0.04
    correction_margin_y: float = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe glottic ROI plus 3-keypoint annotations to YOLO-Pose format."
    )
    parser.add_argument("--labelme-dir", type=Path, default=Path("data/labelme"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/yolo_pose"))
    parser.add_argument("--config", type=Path, default=Path("configs/keypoints.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--split-source-dir", type=Path, help="Optionally mirror source images/JSON by split.")
    parser.add_argument(
        "--group-duplicates-by-image-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exact duplicate image contents in the same split.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on first invalid annotation.")
    return parser.parse_args()


def load_config(path: Path) -> AnnotationConfig:
    if not path.exists():
        return AnnotationConfig()
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = values.get("point_labels", AnnotationConfig().point_labels)
    return AnnotationConfig(
        bbox_label=values.get("bbox_label", AnnotationConfig().bbox_label),
        point_labels=tuple(labels),
        class_id=int(values.get("class_id", AnnotationConfig().class_id)),
        class_name=values.get("class_name", AnnotationConfig().class_name),
        visibility=int(values.get("visibility", AnnotationConfig().visibility)),
        correction_margin_x=float(values.get("correction_margin_x", AnnotationConfig().correction_margin_x)),
        correction_margin_y=float(values.get("correction_margin_y", AnnotationConfig().correction_margin_y)),
    )


def image_size_from_labelme(payload: dict[str, Any], image_path: Path) -> ImageSize:
    width = payload.get("imageWidth")
    height = payload.get("imageHeight")
    if width and height:
        return ImageSize(width=int(width), height=int(height))
    with Image.open(image_path) as image:
        return ImageSize(width=image.width, height=image.height)


def find_image(json_path: Path, payload: dict[str, Any], image_dir: Path) -> Path:
    candidates = []
    image_path = payload.get("imagePath")
    if image_path:
        candidates.append(json_path.parent / image_path)
        candidates.append(image_dir / Path(image_path).name)
    candidates.extend(image_dir / f"{json_path.stem}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No image found for {json_path.name}; checked imagePath and {image_dir}")


def points_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    return points_bbox(points)


def extract_roi(
    shapes: list[dict[str, Any]],
    cfg: AnnotationConfig,
) -> tuple[tuple[float, float, float, float], tuple[tuple[float, float], ...]]:
    matches = [shape for shape in shapes if shape.get("label") == cfg.bbox_label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one bbox shape labelled {cfg.bbox_label!r}, found {len(matches)}")
    points = matches[0].get("points") or []
    if len(points) < 2:
        raise ValueError(f"Bbox shape {cfg.bbox_label!r} must contain at least two points")
    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    return points_to_bbox(points), polygon


def extract_keypoints(shapes: list[dict[str, Any]], cfg: AnnotationConfig) -> tuple[tuple[float, float, float], ...]:
    keypoints = []
    for label in cfg.point_labels:
        matches = [shape for shape in shapes if shape.get("label") == label]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one point shape labelled {label!r}, found {len(matches)}")
        points = matches[0].get("points") or []
        if len(points) != 1:
            raise ValueError(f"Point shape {label!r} must contain exactly one point")
        keypoints.append((float(points[0][0]), float(points[0][1]), float(cfg.visibility)))
    return tuple(keypoints)


def corrected_bbox(
    manual_bbox: tuple[float, float, float, float],
    keypoints: tuple[tuple[float, float, float], ...],
    image_size: ImageSize,
    cfg: AnnotationConfig,
) -> tuple[float, float, float, float]:
    kp_bbox = keypoint_bbox(
        keypoints,
        margin_x=cfg.correction_margin_x,
        margin_y=cfg.correction_margin_y,
        image_size=image_size,
    )
    return union_bbox(clip_bbox(manual_bbox, image_size), kp_bbox, image_size=image_size)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_items(
    items: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    if group_key is None:
        groups = [[item] for item in items]
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(str(item[group_key]), []).append(item)
        groups = list(grouped.values())

    random.Random(seed).shuffle(groups)
    total = sum(len(group) for group in groups)
    test_n = round(total * test_ratio)
    val_n = round(total * val_ratio)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "val": [], "train": []}

    for group in groups:
        if len(splits["test"]) + len(group) <= test_n:
            splits["test"].extend(group)
        elif len(splits["val"]) + len(group) <= val_n:
            splits["val"].extend(group)
        else:
            splits["train"].extend(group)
    return splits


def _duplicate_groups(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["image_sha256"]), []).append(item)
    return {group_id: group for group_id, group in grouped.items() if len(group) > 1}


def link_or_copy_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_file(source: Path, destination: Path, mode: str) -> None:
    link_or_copy_image(source, destination, mode)


def write_dataset_yaml(out_dir: Path, cfg: AnnotationConfig) -> None:
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {cfg.class_id: cfg.class_name},
        "kpt_shape": [len(cfg.point_labels), 3],
    }
    (out_dir / "vocal_fold_pose.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def convert(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    items = []
    errors = []
    for json_path in sorted(args.labelme_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            image_path = find_image(json_path, payload, args.image_dir)
            image_size = image_size_from_labelme(payload, image_path)
            shapes = payload.get("shapes") or []
            keypoints = extract_keypoints(shapes, cfg)
            manual_bbox, manual_roi_polygon = extract_roi(shapes, cfg)
            bbox = corrected_bbox(manual_bbox, keypoints, image_size, cfg)
            label = YoloPoseLabel(cfg.class_id, bbox, keypoints, image_size)
            items.append(
                {
                    "json_path": json_path,
                    "image_path": image_path,
                    "image_size": asdict(image_size),
                    "label": label,
                    "manual_bbox_xyxy": manual_bbox,
                    "manual_roi_polygon": manual_roi_polygon,
                    "image_sha256": file_sha256(image_path) if args.group_duplicates_by_image_hash else image_path.stem,
                }
            )
        except Exception as exc:
            if args.strict:
                raise
            errors.append({"annotation": str(json_path), "error": str(exc)})

    splits = split_items(
        items,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        group_key="image_sha256" if args.group_duplicates_by_image_hash else None,
    )
    for split, split_items_ in splits.items():
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "roi_polygons").mkdir(parents=True, exist_ok=True)
        roi_records = []
        for item in split_items_:
            image_path = item["image_path"]
            label: YoloPoseLabel = item["label"]
            image_dest = args.out_dir / "images" / split / image_path.name
            label_dest = args.out_dir / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy_image(image_path, image_dest, args.copy_mode)
            label_dest.write_text(label.to_line() + "\n", encoding="utf-8")
            roi_records.append(
                {
                    "stem": image_path.stem,
                    "image": str(image_path),
                    "labelme": str(item["json_path"]),
                    "image_sha256": item["image_sha256"],
                    "manual_bbox_xyxy": list(item["manual_bbox_xyxy"]),
                    "manual_roi_polygon": [list(point) for point in item["manual_roi_polygon"]],
                    "keypoints": [list(point) for point in label.keypoints],
                }
            )
            if args.split_source_dir is not None and args.copy_mode != "none":
                link_or_copy_file(
                    image_path,
                    args.split_source_dir / "images" / split / image_path.name,
                    args.copy_mode,
                )
                link_or_copy_file(
                    item["json_path"],
                    args.split_source_dir / "labelme" / split / item["json_path"].name,
                    args.copy_mode,
                )
        roi_path = args.out_dir / "roi_polygons" / f"{split}.jsonl"
        roi_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in roi_records),
            encoding="utf-8",
        )

    write_dataset_yaml(args.out_dir, cfg)
    manifest = {
        "config": asdict(cfg),
        "counts": {split: len(values) for split, values in splits.items()},
        "group_duplicates_by_image_hash": bool(args.group_duplicates_by_image_hash),
        "duplicate_image_groups": [
            {"image_sha256": group_id, "stems": sorted(item["image_path"].stem for item in group)}
            for group_id, group in _duplicate_groups(items).items()
        ],
        "errors": errors,
    }
    (args.out_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = convert(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    if manifest["errors"]:
        print(f"Skipped {len(manifest['errors'])} invalid annotation(s); see conversion_manifest.json")


if __name__ == "__main__":
    main()
