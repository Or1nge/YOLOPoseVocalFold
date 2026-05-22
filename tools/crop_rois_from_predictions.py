#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = (
    "混杂图片",
    "正常",
    "声带任克水肿",
    "声带囊肿",
    "声带息肉",
    "声带白斑",
    "声带肉芽肿",
    "喉癌",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop accepted vocal-fold ROIs from prediction JSONL while preserving class folders.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="*", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--crop-actions",
        nargs="*",
        default=["auto_accept"],
        help="Prediction actions to crop. Defaults to high-confidence auto_accept only.",
    )
    parser.add_argument(
        "--copy-original-actions",
        nargs="*",
        default=[],
        help="Prediction actions to keep as uncropped original images.",
    )
    parser.add_argument(
        "--copy-original-classes",
        nargs="*",
        default=[],
        help="Optional class-name allowlist for --copy-original-actions. Empty means all classes.",
    )
    parser.add_argument(
        "--fallback-original-on-crop-failure",
        action="store_true",
        help="Copy the original image when a requested crop cannot be produced.",
    )
    parser.add_argument(
        "--copy-original-source",
        choices=["source", "original_source"],
        default="source",
        help="Which record path to use when retaining/falling back to an original image.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=0,
        help="If >0, stretch every saved output to this square size in pixels.",
    )
    parser.add_argument("--crop-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def relative_to_root(path: Path, root: Path) -> Path | None:
    path_abs = path if path.is_absolute() else path.absolute()
    root_abs = root if root.is_absolute() else root.absolute()
    try:
        return path_abs.relative_to(root_abs)
    except ValueError:
        return None


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def polygon_size(points: Sequence[Sequence[float]]) -> tuple[int, int]:
    top_width = math.dist(points[0], points[1])
    bottom_width = math.dist(points[3], points[2])
    right_height = math.dist(points[1], points[2])
    left_height = math.dist(points[0], points[3])
    width = max(1, int(round(max(top_width, bottom_width))))
    height = max(1, int(round(max(left_height, right_height))))
    return width, height


def crop_polygon(image: Image.Image, polygon: Sequence[Sequence[float]]) -> Image.Image | None:
    if len(polygon) != 4:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    width, height = polygon_size(polygon)
    source = np.asarray([[float(x), float(y)] for x, y in polygon], dtype=np.float32)
    target = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), matrix, (width, height))
    return Image.fromarray(warped)


def crop_prediction(image_path: Path, record: dict[str, Any], mode: str) -> Image.Image | None:
    image = Image.open(image_path).convert("RGB")
    if mode == "polygon":
        polygon = record.get("usable_box_polygon") or record.get("final_box_polygon")
        if polygon:
            crop = crop_polygon(image, polygon)
            if crop is not None:
                return crop
    bbox = record.get("usable_bbox") or record.get("final_bbox")
    if not bbox:
        return None
    clamped = clamp_bbox(bbox, image.width, image.height)
    if clamped is None:
        return None
    return image.crop(clamped)


def square_resize(image: Image.Image, output_size: int) -> Image.Image:
    if output_size <= 0:
        return image
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return image.convert("RGB").resize((output_size, output_size), resample=resample)


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=max(1, min(int(quality), 100)), subsampling=0)
    else:
        image.save(path)


def save_original_image(source: Path, destination: Path, *, output_size: int, quality: int) -> None:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_size <= 0:
        shutil.copy2(source, destination)
        return
    image = Image.open(source).convert("RGB")
    save_image(square_resize(image, output_size), destination, quality)


def original_source_for_record(record: dict[str, Any], source: Path, mode: str) -> Path:
    if mode == "original_source":
        original = record.get("original_source")
        if original:
            return Path(str(original))
    return source


def class_from_relative(rel_path: Path, allowed_classes: set[str]) -> str | None:
    if not rel_path.parts:
        return None
    first = rel_path.parts[0]
    return first if first in allowed_classes else None


def main() -> None:
    args = parse_args()
    allowed_classes = set(args.classes)
    crop_actions = set(args.crop_actions)
    copy_original_actions = set(args.copy_original_actions)
    copy_original_classes = set(args.copy_original_classes)
    source_root = args.source_root if args.source_root.is_absolute() else args.source_root.absolute()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in args.classes:
        (args.out_dir / class_name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    failures: Counter[str] = Counter()

    for record in read_jsonl(args.predictions):
        source = Path(str(record.get("source", "")))
        if not source.exists():
            failures["missing_source"] += 1
            continue
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel_path = relative_to_root(source, source_root)
        if rel_path is None:
            failures["outside_source_root"] += 1
            continue
        class_name = class_from_relative(rel_path, allowed_classes)
        if class_name is None:
            failures["outside_class_set"] += 1
            continue

        action = str(record.get("action", ""))
        counts[class_name]["total"] += 1
        counts[class_name][action or "unknown"] += 1
        out_rel = rel_path
        out_path = args.out_dir / out_rel
        cropped = False
        copied_original = False
        original_output_source = ""
        error = ""
        if action in crop_actions:
            try:
                crop = crop_prediction(source, record, args.crop_mode)
                if crop is None:
                    failures["empty_crop"] += 1
                    error = "empty_crop"
                else:
                    crop = square_resize(crop, args.output_size)
                    save_image(crop, out_path, args.jpeg_quality)
                    cropped = True
                    counts[class_name]["cropped"] += 1
            except Exception as exc:  # noqa: BLE001 - manifest should preserve per-image failures.
                failures["crop_error"] += 1
                error = f"{type(exc).__name__}: {exc}"
            if (not cropped) and args.fallback_original_on_crop_failure:
                original = original_source_for_record(record, source, args.copy_original_source)
                original_output_source = str(original)
                if original.exists():
                    save_original_image(original, out_path, output_size=args.output_size, quality=args.jpeg_quality)
                    copied_original = True
                    counts[class_name]["copied_original"] += 1
                else:
                    failures["missing_original_source"] += 1
                    error = "missing_original_source" if not error else f"{error};missing_original_source"
        elif action in copy_original_actions and (not copy_original_classes or class_name in copy_original_classes):
            original = original_source_for_record(record, source, args.copy_original_source)
            original_output_source = str(original)
            if original.exists():
                save_original_image(original, out_path, output_size=args.output_size, quality=args.jpeg_quality)
                copied_original = True
                counts[class_name]["copied_original"] += 1
            else:
                failures["missing_original_source"] += 1
                error = "missing_original_source"

        rows.append(
            {
                "source": str(source),
                "relative_path": str(rel_path),
                "class_name": class_name,
                "action": action,
                "cropped": cropped,
                "copied_original": copied_original,
                "original_output_source": original_output_source,
                "crop_path": str(out_path) if cropped else "",
                "output_path": str(out_path) if cropped or copied_original else "",
                "final_confidence": record.get("final_confidence", ""),
                "bbox_confidence": record.get("bbox_confidence", ""),
                "keypoint_confidence": record.get("keypoint_confidence", ""),
                "glottic_angle_degrees": record.get("glottic_angle_degrees", ""),
                "roi_area_ratio": record.get("roi_area_ratio", ""),
                "roi_dark_fraction": record.get("roi_dark_fraction", ""),
                "flags": ";".join(record.get("flags", [])),
                "error": error,
            }
        )

    manifest_path = args.out_dir / "roi_crop_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "relative_path",
            "class_name",
            "action",
            "cropped",
            "copied_original",
            "original_output_source",
            "crop_path",
            "output_path",
            "final_confidence",
            "bbox_confidence",
            "keypoint_confidence",
            "glottic_angle_degrees",
            "roi_area_ratio",
            "roi_dark_fraction",
            "flags",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_root": str(source_root),
        "predictions": str(args.predictions),
        "out_dir": str(args.out_dir),
        "classes": list(args.classes),
        "crop_actions": sorted(crop_actions),
        "copy_original_actions": sorted(copy_original_actions),
        "copy_original_classes": sorted(copy_original_classes),
        "fallback_original_on_crop_failure": bool(args.fallback_original_on_crop_failure),
        "copy_original_source": args.copy_original_source,
        "output_size": args.output_size,
        "crop_mode": args.crop_mode,
        "total_records": len(rows),
        "counts_by_class": {class_name: dict(counts[class_name]) for class_name in args.classes},
        "failures": dict(failures),
    }
    summary_path = args.out_dir / "roi_crop_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
