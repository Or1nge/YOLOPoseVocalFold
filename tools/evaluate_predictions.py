#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.metrics import evaluate_sample, summarize_metrics
from yoloposevf.yolo_labels import read_yolo_pose_label


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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
        if not prediction.get("final_bbox") or not prediction.get("keypoints"):
            invalid_predictions.append(label_path.stem)
            continue
        with Image.open(image_path) as image:
            image_size = ImageSize(image.width, image.height)
        target = read_yolo_pose_label(label_path, image_size)
        roi_record = roi_metadata.get(label_path.stem, {})
        samples.append(
            evaluate_sample(
                source=label_path.stem,
                predicted_bbox=prediction["final_bbox"],
                target_bbox=target.bbox_xyxy,
                predicted_keypoints=prediction.get("keypoints", []),
                target_keypoints=target.keypoints,
                image_size=image_size,
                action=str(prediction.get("action", "")),
                final_confidence=float(prediction.get("final_confidence", 0.0)),
                predicted_roi_polygon=prediction.get("roi_polygon"),
                target_roi_polygon=roi_record.get("manual_roi_polygon"),
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
    summary_path = args.out_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
