#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
FLAG_LABELS = {
    "anterior_point_not_below_posterior_points": "bad_ant",
    "weak_anterior_posterior_orientation": "weak_ant",
    "roi_too_bright": "bright",
    "low_roi_dark_fraction": "low_dark",
    "roi_area_too_small": "small_roi",
    "low_roi_area": "low_area",
    "keypoints_outside_image": "kp_oob",
    "implausible_keypoint_angle": "bad_angle",
    "low_keypoint_confidence": "low_kp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prediction overlays for quick visual review.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_pose"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_roi_metadata(dataset_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    path = dataset_dir / "roi_polygons" / f"{split}.jsonl"
    if not path.exists():
        return {}
    return {record["stem"]: record for record in read_jsonl(path)}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        path = images_dir / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_polygon(draw: ImageDraw.ImageDraw, points: list[list[float]], color: str, width: int) -> None:
    if len(points) >= 2:
        xy = [(float(x), float(y)) for x, y in points]
        draw.line(xy + [xy[0]], fill=color, width=width)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: list[float], color: str, width: int) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def draw_keypoints(draw: ImageDraw.ImageDraw, keypoints: list[list[float]]) -> None:
    colors = ["red", "deepskyblue", "magenta"]
    labels = ["A", "L", "R"]
    for index, point in enumerate(keypoints):
        x, y = float(point[0]), float(point[1])
        radius = 5
        color = colors[index % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
        draw.text((x + radius + 3, y - radius - 3), labels[index] if index < len(labels) else str(index), fill=color)


def action_color(action: str) -> str:
    if action == "reject_or_relabel":
        return "red"
    if action == "manual_review":
        return "yellow"
    return "orange"


def action_label(action: str) -> str:
    if action == "auto_accept":
        return "usable"
    if action == "manual_review":
        return "review"
    if action == "reject_or_relabel":
        return "reject"
    return action or "unknown"


def bbox_wh(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def compact_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def compact_flags(flags: list[str]) -> str:
    if not flags:
        return "none"
    labels = [FLAG_LABELS.get(flag, flag) for flag in flags[:3]]
    return compact_text(",".join(labels), 28)


def render_overlay(
    image_path: Path,
    prediction: dict[str, Any],
    roi_record: dict[str, Any],
    out_path: Path,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    if roi_record.get("manual_roi_polygon"):
        draw_polygon(draw, roi_record["manual_roi_polygon"], "lime", 4)
    if prediction.get("roi_polygon"):
        draw_polygon(draw, prediction["roi_polygon"], "cyan", 3)
    action = str(prediction.get("action", ""))
    box_color = action_color(action)
    box_polygon = prediction.get("usable_box_polygon") or prediction.get("final_box_polygon")
    box_bbox = prediction.get("usable_bbox") or prediction.get("final_bbox")
    if box_polygon:
        draw_polygon(draw, box_polygon, box_color, 5)
    elif box_bbox:
        draw_bbox(draw, box_bbox, box_color, 3)
    if prediction.get("keypoints"):
        draw_keypoints(draw, prediction["keypoints"])

    confidence = float(prediction.get("final_confidence", 0.0))
    angle = prediction.get("glottic_angle_degrees")
    angle_text = f"{float(angle):.1f}" if angle is not None else "NA"
    wh = bbox_wh(prediction.get("final_bbox"))
    wh_text = f"{wh[0]:.0f}x{wh[1]:.0f}" if wh is not None else "NA"
    roi_area = prediction.get("roi_area_ratio")
    roi_text = f"{float(roi_area):.3f}" if roi_area is not None else "NA"
    dark_fraction = prediction.get("roi_dark_fraction")
    dark_text = f"{float(dark_fraction):.3f}" if dark_fraction is not None else "NA"
    anterior_ratio = prediction.get("anterior_y_offset_ratio")
    anterior_text = f"{float(anterior_ratio):.2f}" if anterior_ratio is not None else "NA"
    flags = compact_flags(prediction.get("flags", []))
    line1 = f"{compact_text(image_path.stem, 30)} | {action_label(action)}"
    line2 = f"conf={confidence:.3f} angle={angle_text} wh={wh_text} roi={roi_text}"
    line3 = f"bbox={float(prediction.get('bbox_confidence', 0.0)):.3f} kp={float(prediction.get('keypoint_confidence', 0.0)):.3f} dark={dark_text} ant={anterior_text} flags={flags}"
    draw.rectangle((0, 0, image.width, 72), fill=(0, 0, 0))
    draw.text((8, 5), line1, fill="white", font=label_font)
    draw.text((8, 27), line2, fill="white", font=label_font)
    draw.text((8, 49), line3, fill="white", font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path, quality=92)
    return overlay


def make_contact_sheet(images: list[Image.Image], out_path: Path, thumb_width: int, columns: int) -> None:
    if not images:
        return
    thumbs = []
    for image in images:
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * scale))))
        thumbs.append(thumb)
    cell_height = max(thumb.height for thumb in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_width * columns, cell_height * rows), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    roi_metadata = read_roi_metadata(args.dataset_dir, args.split)
    images_dir = args.dataset_dir / "images" / args.split
    label_font = font(14)
    rendered = []

    for prediction in predictions:
        source_path = Path(str(prediction.get("source", "")))
        stem = source_path.stem
        if not stem:
            continue
        image_path = find_image(images_dir, stem)
        if image_path is None and source_path.exists():
            image_path = source_path
        if image_path is None:
            continue
        out_path = args.out_dir / f"{stem}_overlay.jpg"
        rendered.append(render_overlay(image_path, prediction, roi_metadata.get(stem, {}), out_path, label_font))

    make_contact_sheet(rendered, args.out_dir / "contact_sheet.jpg", args.thumb_width, args.columns)
    print(json.dumps({"rendered": len(rendered), "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
