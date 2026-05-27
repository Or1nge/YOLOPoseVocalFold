#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.metrics import evaluate_sample, summarize_metrics
from yoloposevf.yolo_labels import read_yolo_pose_label


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def transform_point_with_preprocess(point: tuple[float, float], preprocess: dict[str, object]) -> tuple[float, float]:
    crop = preprocess.get("crop_bbox_xyxy")
    if not isinstance(crop, list) or len(crop) < 2:
        return point
    padding = float(preprocess.get("padding_px") or 0.0)
    return float(point[0]) - float(crop[0]) + padding, float(point[1]) - float(crop[1]) + padding


def transform_bbox_with_preprocess(bbox: tuple[float, float, float, float], preprocess: dict[str, object]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    points = [
        transform_point_with_preprocess((x1, y1), preprocess),
        transform_point_with_preprocess((x2, y1), preprocess),
        transform_point_with_preprocess((x2, y2), preprocess),
        transform_point_with_preprocess((x1, y2), preprocess),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def transform_keypoints_with_preprocess(
    keypoints: list[tuple[float, float, float]],
    preprocess: dict[str, object],
) -> list[tuple[float, float, float]]:
    transformed = []
    for keypoint in keypoints:
        x, y = transform_point_with_preprocess((float(keypoint[0]), float(keypoint[1])), preprocess)
        transformed.append((x, y, float(keypoint[2])))
    return transformed


def transform_polygon_with_preprocess(
    polygon: object,
    preprocess: dict[str, object],
) -> list[list[float]] | None:
    if not isinstance(polygon, list):
        return None
    transformed = []
    for point in polygon:
        if not isinstance(point, list) or len(point) < 2:
            return None
        x, y = transform_point_with_preprocess((float(point[0]), float(point[1])), preprocess)
        transformed.append([x, y])
    return transformed


def image_size_from_prediction_or_target(prediction: dict[str, object], fallback: ImageSize) -> ImageSize:
    image_size = prediction.get("image_size")
    if isinstance(image_size, dict):
        width = image_size.get("width")
        height = image_size.get("height")
        if width and height:
            return ImageSize(int(width), int(height))
    preprocess = prediction.get("preprocess")
    if isinstance(preprocess, dict):
        width = preprocess.get("model_input_width")
        height = preprocess.get("model_input_height")
        if width and height:
            return ImageSize(int(width), int(height))
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate postprocessed predictions against YOLO-Pose labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_pose"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--out-dir", type=Path, default=Path("Results/evaluation"))
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        source = Path(str(payload.get("source", ""))).stem
        if source:
            records[source] = payload
    return records


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def read_roi_metadata(dataset_dir: Path, split: str) -> dict[str, dict[str, object]]:
    path = dataset_dir / "roi_polygons" / f"{split}.jsonl"
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        stem = str(payload.get("stem", ""))
        if stem:
            records[stem] = payload
    return records


def main() -> None:
    args = parse_args()
    predictions = read_predictions(args.predictions)
    roi_metadata = read_roi_metadata(args.dataset_dir, args.split)
    labels_dir = args.dataset_dir / "labels" / args.split
    images_dir = args.dataset_dir / "images" / args.split

    samples = []
    missing_predictions = []
    invalid_predictions = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        prediction = predictions.get(label_path.stem)
        if prediction is None:
            missing_predictions.append(label_path.stem)
            continue
        predicted_bbox = prediction.get("final_bbox_xyxy") or prediction.get("final_bbox")
        predicted_box_polygon = prediction.get("final_box_polygon") or prediction.get("roi_polygon")
        if not predicted_bbox or not prediction.get("keypoints"):
            invalid_predictions.append(label_path.stem)
            continue
        with Image.open(image_path) as image:
            image_size = ImageSize(image.width, image.height)
        target = read_yolo_pose_label(label_path, image_size)
        roi_record = roi_metadata.get(label_path.stem, {})
        metric_image_size = image_size_from_prediction_or_target(prediction, image_size)
        target_bbox = target.bbox_xyxy
        target_keypoints = target.keypoints
        target_roi_polygon = roi_record.get("manual_roi_polygon")
        preprocess = prediction.get("preprocess")
        if isinstance(preprocess, dict):
            target_bbox = transform_bbox_with_preprocess(target_bbox, preprocess)
            target_keypoints = transform_keypoints_with_preprocess(target_keypoints, preprocess)
            target_roi_polygon = transform_polygon_with_preprocess(target_roi_polygon, preprocess)
        samples.append(
            evaluate_sample(
                source=label_path.stem,
                predicted_bbox=predicted_bbox,
                target_bbox=target_bbox,
                predicted_keypoints=prediction.get("keypoints", []),
                target_keypoints=target_keypoints,
                image_size=metric_image_size,
                action=str(prediction.get("action", "")),
                final_confidence=float(prediction.get("final_confidence", 0.0)),
                predicted_roi_polygon=predicted_box_polygon,
                target_roi_polygon=target_roi_polygon,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.split}_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "bbox_iou",
                "containment_rate",
                "normalized_keypoint_error",
                "pck",
                "action",
                "final_confidence",
                "roi_polygon_containment_rate",
                "roi_area_ratio_to_target",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)

    summary = summarize_metrics(samples)
    summary["missing_predictions"] = missing_predictions
    summary["invalid_predictions"] = invalid_predictions
    summary["all_prediction_actions"] = dict(Counter(str(item.get("action", "")) for item in predictions.values()))
    summary["usable_prediction_count"] = sum(
        1 for item in predictions.values() if item.get("usable_box_polygon") is not None or item.get("usable_bbox") is not None
    )
    summary["rejected_prediction_count"] = sum(
        1 for item in predictions.values() if item.get("action") == "reject_or_relabel"
    )
    summary_path = args.out_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
