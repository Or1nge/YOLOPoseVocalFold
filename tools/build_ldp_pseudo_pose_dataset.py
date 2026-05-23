#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize, clip_bbox, keypoint_bbox, union_bbox  # noqa: E402
from yoloposevf.yolo_labels import YoloPoseLabel  # noqa: E402


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
DEFAULT_POSITIVE_CLASSES = tuple(class_name for class_name in DEFAULT_CLASSES if class_name != "混杂图片")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO-Pose fine-tuning dataset from LDP pseudo labels and mixed-image negatives."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positive-classes", nargs="*", default=list(DEFAULT_POSITIVE_CLASSES))
    parser.add_argument("--negative-classes", nargs="*", default=["混杂图片"])
    parser.add_argument("--positive-actions", nargs="*", default=["auto_accept", "manual_review"])
    parser.add_argument("--hard-negative-actions", nargs="*", default=["auto_accept"])
    parser.add_argument("--negative-repeat", type=int, default=1)
    parser.add_argument("--hard-negative-repeat", type=int, default=8)
    parser.add_argument("--min-positive-confidence", type=float, default=0.40)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="JSONL manifest of LDP holdout records to exclude from pseudo-label training.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--prefix", default="ldp_pseudo")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def copy_base_dataset(base_dataset: Path, out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_dir}")
        shutil.rmtree(out_dir)
    shutil.copytree(base_dataset, out_dir, symlinks=True)


def update_dataset_yaml(out_dir: Path) -> dict[str, Any]:
    yaml_path = out_dir / "vocal_fold_pose.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    payload["path"] = str(out_dir.resolve())
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def class_from_source(source: Path) -> str:
    return source.parent.name


def stable_source_key(record: dict[str, Any]) -> str:
    source = record.get("original_source") or record.get("source") or ""
    return str(Path(str(source)).resolve())


def load_excluded_sources(path: Path | None) -> set[str]:
    if path is None:
        return set()
    excluded = set()
    for record in read_jsonl(path):
        key = record.get("source_key") or record.get("original_source") or record.get("source")
        if key:
            excluded.add(str(Path(str(key)).resolve()))
    return excluded


def unique_image_name(prefix: str, class_name: str, source: Path, repeat_index: int | None = None) -> str:
    safe_class = class_name.replace("/", "_")
    repeat = "" if repeat_index is None else f"__r{repeat_index:02d}"
    source_key = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{safe_class}__{source.stem}__{source_key}{repeat}{source.suffix.lower()}"


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)
    else:
        destination.symlink_to(source.resolve())


def link_copy_or_materialize(record: dict[str, Any], source: Path, destination: Path, mode: str) -> None:
    if source.exists():
        link_or_copy(source, destination, mode)
        return

    original = Path(str(record.get("original_source", "")))
    preprocess = record.get("preprocess") or {}
    if original.exists() and preprocess.get("type") == "blackpad":
        padding = int(preprocess.get("padding_px") or 0)
        padded_width = int(preprocess.get("padded_width") or 0)
        padded_height = int(preprocess.get("padded_height") or 0)
        if padding < 0 or padded_width <= 0 or padded_height <= 0:
            raise FileNotFoundError(f"Cannot materialize blackpad image for {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as image:
            rgb = image.convert("RGB")
            canvas = Image.new("RGB", (padded_width, padded_height), "black")
            canvas.paste(rgb, (padding, padding))
            canvas.save(destination, quality=95)
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)
        return

    raise FileNotFoundError(f"Source image not found and cannot be materialized: {source}")


def image_size(record: dict[str, Any], image_path: Path) -> ImageSize:
    payload = record.get("image_size") or {}
    width = payload.get("width")
    height = payload.get("height")
    if width and height:
        return ImageSize(width=int(width), height=int(height))
    with Image.open(image_path) as image:
        return ImageSize(width=image.width, height=image.height)


def pseudo_label_from_prediction(record: dict[str, Any], image_path: Path) -> YoloPoseLabel | None:
    size = image_size(record, image_path)
    bbox = record.get("final_bbox") or record.get("final_bbox_xyxy") or record.get("bbox_yolo")
    keypoints = record.get("keypoints")
    if not bbox or len(bbox) < 4 or not keypoints or len(keypoints) != 3:
        return None
    kpts = []
    for keypoint in keypoints:
        if len(keypoint) < 2:
            return None
        x = min(max(float(keypoint[0]), 1e-3), max(float(size.width) - 1e-3, 1e-3))
        y = min(max(float(keypoint[1]), 1e-3), max(float(size.height) - 1e-3, 1e-3))
        kpts.append((x, y, 2.0))
    clipped_bbox = clip_bbox(tuple(float(value) for value in bbox[:4]), size)
    kp_bbox = keypoint_bbox(kpts, margin_x=0.05, margin_y=0.05, image_size=size)
    clipped_bbox = union_bbox(clipped_bbox, kp_bbox, image_size=size)
    return YoloPoseLabel(class_id=0, bbox_xyxy=clipped_bbox, keypoints=tuple(kpts), image_size=size)


def write_negative(
    *,
    record: dict[str, Any],
    image_path: Path,
    class_name: str,
    images_dir: Path,
    labels_dir: Path,
    prefix: str,
    copy_mode: str,
    repeat_index: int,
    label_type: str,
) -> dict[str, Any]:
    dest_name = unique_image_name(prefix, class_name, image_path, repeat_index)
    image_dest = images_dir / dest_name
    label_dest = labels_dir / f"{Path(dest_name).stem}.txt"
    link_copy_or_materialize(record, image_path, image_dest, copy_mode)
    label_dest.write_text("", encoding="utf-8")
    return {
        "source": str(image_path.resolve()),
        "image": str(image_dest.resolve()),
        "label": str(label_dest.resolve()),
        "class_name": class_name,
        "label_type": label_type,
        "repeat_index": repeat_index,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.negative_repeat < 0 or args.hard_negative_repeat < 0:
        raise ValueError("repeat values must be non-negative")
    copy_base_dataset(args.base_dataset, args.out_dir, overwrite=args.overwrite)
    dataset_yaml = update_dataset_yaml(args.out_dir)

    images_dir = args.out_dir / "images" / args.split
    labels_dir = args.out_dir / "labels" / args.split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    positive_classes = set(args.positive_classes)
    negative_classes = set(args.negative_classes)
    positive_actions = set(args.positive_actions)
    hard_negative_actions = set(args.hard_negative_actions)
    excluded_sources = load_excluded_sources(args.exclude_manifest)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for record in read_jsonl(args.predictions):
        source = Path(str(record.get("source", "")))
        if stable_source_key(record) in excluded_sources:
            counts["skipped_holdout"] += 1
            continue
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            counts["skipped_nonimage"] += 1
            continue
        class_source = Path(str(record.get("original_source") or source))
        class_name = class_from_source(class_source)
        action = str(record.get("action", ""))
        final_confidence = float(record.get("final_confidence") or 0.0)

        if class_name in negative_classes:
            repeat = args.negative_repeat
            label_type = "ldp_negative"
            if action in hard_negative_actions:
                repeat += args.hard_negative_repeat
                label_type = "ldp_hard_negative"
            for repeat_index in range(1, repeat + 1):
                row = write_negative(
                    record=record,
                    image_path=source,
                    class_name=class_name,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    prefix=args.prefix,
                    copy_mode=args.copy_mode,
                    repeat_index=repeat_index,
                    label_type=label_type,
                )
                row.update({"action": action, "final_confidence": final_confidence})
                rows.append(row)
                counts[label_type] += 1
            continue

        if (
            class_name in positive_classes
            and action in positive_actions
            and final_confidence > float(args.min_positive_confidence)
        ):
            label = pseudo_label_from_prediction(record, source)
            if label is None:
                counts["skipped_bad_pseudo_label"] += 1
                continue
            dest_name = unique_image_name(args.prefix, class_name, source)
            image_dest = images_dir / dest_name
            label_dest = labels_dir / f"{Path(dest_name).stem}.txt"
            link_copy_or_materialize(record, source, image_dest, args.copy_mode)
            label_dest.write_text(label.to_line() + "\n", encoding="utf-8")
            rows.append(
                {
                    "source": str(source.resolve()),
                    "image": str(image_dest.resolve()),
                    "label": str(label_dest.resolve()),
                    "class_name": class_name,
                    "label_type": "ldp_pseudo_positive",
                    "action": action,
                    "final_confidence": final_confidence,
                    "repeat_index": "",
                }
            )
            counts["ldp_pseudo_positive"] += 1
            counts[f"ldp_pseudo_positive_{action}"] += 1
        else:
            counts["skipped_not_training_sample"] += 1

    manifest_csv = args.out_dir / "ldp_pseudo_samples_manifest.csv"
    fieldnames = [
        "source",
        "image",
        "label",
        "class_name",
        "label_type",
        "action",
        "final_confidence",
        "repeat_index",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "base_dataset": str(args.base_dataset.resolve()),
        "predictions": str(args.predictions.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "split": args.split,
        "copy_mode": args.copy_mode,
        "positive_classes": sorted(positive_classes),
        "negative_classes": sorted(negative_classes),
        "positive_actions": sorted(positive_actions),
        "hard_negative_actions": sorted(hard_negative_actions),
        "negative_repeat": args.negative_repeat,
        "hard_negative_repeat": args.hard_negative_repeat,
        "min_positive_confidence": args.min_positive_confidence,
        "positive_confidence_operator": ">",
        "exclude_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else None,
        "excluded_source_count": len(excluded_sources),
        "counts": dict(counts),
        "added_sample_count": len(rows),
        "label_contract": "empty label file means background/no glottic ROI",
        "dataset_yaml": dataset_yaml,
        "manifest_csv": str(manifest_csv.resolve()),
    }
    (args.out_dir / "ldp_pseudo_samples_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(json.dumps({"out_dir": manifest["out_dir"], "counts": manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
