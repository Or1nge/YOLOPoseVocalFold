#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize, bbox_area, containment_rate
from yoloposevf.yolo_labels import read_yolo_pose_label


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a converted YOLO-Pose vocal fold dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_pose"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo_pose/validation_report.json"))
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def validate_dataset(dataset_dir: Path) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        labels_dir = dataset_dir / "labels" / split
        images_dir = dataset_dir / "images" / split
        image_stems = {
            image_path.stem
            for image_path in images_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_stems = set()
        counts[f"{split}_images"] += len(image_stems)
        for label_path in sorted(labels_dir.glob("*.txt")):
            label_stems.add(label_path.stem)
            counts[f"{split}_labels"] += 1
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                issues.append({"split": split, "label": str(label_path), "issue": "missing_image"})
                continue
            try:
                if not label_path.read_text(encoding="utf-8").strip():
                    counts[f"{split}_negative_labels"] += 1
                    continue
                with Image.open(image_path) as image:
                    image_size = ImageSize(image.width, image.height)
                label = read_yolo_pose_label(label_path, image_size)
                if bbox_area(label.bbox_xyxy) <= 1:
                    issues.append({"split": split, "label": str(label_path), "issue": "empty_bbox"})
                if containment_rate(label.bbox_xyxy, [kp[:2] for kp in label.keypoints]) < 1.0:
                    issues.append({"split": split, "label": str(label_path), "issue": "bbox_does_not_contain_all_keypoints"})
                for index, (x, y, visibility) in enumerate(label.keypoints, start=1):
                    if not (0 <= x <= image_size.width and 0 <= y <= image_size.height):
                        issues.append({"split": split, "label": str(label_path), "issue": f"kp{index}_outside_image"})
                    if visibility not in (0, 1, 2):
                        issues.append({"split": split, "label": str(label_path), "issue": f"kp{index}_bad_visibility"})
            except Exception as exc:
                issues.append({"split": split, "label": str(label_path), "issue": "parse_error", "detail": str(exc)})
        for stem in sorted(image_stems - label_stems):
            issues.append({"split": split, "image": str(images_dir / stem), "issue": "missing_label"})

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "counts": dict(counts),
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"issue_count": report["issue_count"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
