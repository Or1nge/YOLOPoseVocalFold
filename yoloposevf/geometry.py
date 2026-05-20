from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, hypot, pi
from typing import Iterable, Sequence


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clip_bbox(bbox: Sequence[float], image_size: ImageSize) -> BBox:
    x1, y1, x2, y2 = map(float, bbox)
    x1 = clamp(x1, 0.0, float(image_size.width))
    x2 = clamp(x2, 0.0, float(image_size.width))
    y1 = clamp(y1, 0.0, float(image_size.height))
    y2 = clamp(y2, 0.0, float(image_size.height))
    return normalize_xyxy((x1, y1, x2, y2))


def normalize_xyxy(bbox: Sequence[float]) -> BBox:
    x1, y1, x2, y2 = map(float, bbox)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def xyxy_to_yolo(bbox: Sequence[float], image_size: ImageSize) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = normalize_xyxy(bbox)
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    return (
        ((x1 + x2) / 2.0) / image_size.width,
        ((y1 + y2) / 2.0) / image_size.height,
        width / image_size.width,
        height / image_size.height,
    )


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_size: ImageSize,
) -> BBox:
    box_w = width * image_size.width
    box_h = height * image_size.height
    cx = x_center * image_size.width
    cy = y_center * image_size.height
    return normalize_xyxy((cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0))


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_xyxy(bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = normalize_xyxy(a)
    bx1, by1, bx2, by2 = normalize_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def keypoint_bbox(
    keypoints: Sequence[Sequence[float]],
    margin_x: float = 0.15,
    margin_y: float = 0.15,
    image_size: ImageSize | None = None,
) -> BBox:
    points = [tuple(map(float, point[:2])) for point in keypoints if len(point) >= 2]
    if not points:
        raise ValueError("keypoints must be an Nx2 or Nx3 array")
    x1 = min(point[0] for point in points)
    y1 = min(point[1] for point in points)
    x2 = max(point[0] for point in points)
    y2 = max(point[1] for point in points)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    bbox = (x1 - w * margin_x, y1 - h * margin_y, x2 + w * margin_x, y2 + h * margin_y)
    if image_size is not None:
        return clip_bbox(bbox, image_size)
    return normalize_xyxy(bbox)


def expand_bbox(
    bbox: Sequence[float],
    margin_x: float,
    margin_y: float,
    image_size: ImageSize | None = None,
) -> BBox:
    x1, y1, x2, y2 = normalize_xyxy(bbox)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    expanded = (x1 - w * margin_x, y1 - h * margin_y, x2 + w * margin_x, y2 + h * margin_y)
    if image_size is not None:
        return clip_bbox(expanded, image_size)
    return normalize_xyxy(expanded)


def union_bbox(*boxes: Sequence[float], image_size: ImageSize | None = None) -> BBox:
    valid_boxes = [normalize_xyxy(box) for box in boxes if bbox_area(box) > 0]
    if not valid_boxes:
        raise ValueError("at least one non-empty bbox is required")
    x1 = min(box[0] for box in valid_boxes)
    y1 = min(box[1] for box in valid_boxes)
    x2 = max(box[2] for box in valid_boxes)
    y2 = max(box[3] for box in valid_boxes)
    merged = (x1, y1, x2, y2)
    if image_size is not None:
        return clip_bbox(merged, image_size)
    return normalize_xyxy(merged)


def contains_point(bbox: Sequence[float], point: Sequence[float], tolerance: float = 0.0) -> bool:
    x1, y1, x2, y2 = normalize_xyxy(bbox)
    x, y = float(point[0]), float(point[1])
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def containment_rate(bbox: Sequence[float], keypoints: Sequence[Sequence[float]]) -> float:
    points = list(keypoints)
    if not points:
        return 0.0
    inside = sum(1 for point in points if contains_point(bbox, point))
    return inside / len(points)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def segment_angle(a: Sequence[float], b: Sequence[float]) -> float:
    return atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def angle_difference_degrees(a: float, b: float) -> float:
    diff = abs((a - b + pi) % (2 * pi) - pi)
    return degrees(diff)


def _soft_score(value: float, good: float, bad: float, larger_is_worse: bool = True) -> float:
    if larger_is_worse:
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        return 1.0 - (value - good) / (bad - good)
    if value >= good:
        return 1.0
    if value <= bad:
        return 0.0
    return (value - bad) / (good - bad)


def geometry_score(
    keypoints: Sequence[Sequence[float]],
    bbox: Sequence[float],
    image_size: ImageSize,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.75,
    min_aspect: float = 0.35,
    max_aspect: float = 5.0,
) -> float:
    points = [tuple(map(float, point[:2])) for point in keypoints if len(point) >= 2]
    if len(points) != 4:
        return 0.0

    left_len = distance(points[0], points[1])
    right_len = distance(points[2], points[3])
    max_len = max(left_len, right_len)
    if max_len <= 1e-6:
        length_score = 0.0
    else:
        length_ratio_gap = abs(left_len - right_len) / max_len
        length_score = _soft_score(length_ratio_gap, good=0.25, bad=0.80)

    left_angle = segment_angle(points[0], points[1])
    right_angle = segment_angle(points[2], points[3])
    angle_gap = angle_difference_degrees(left_angle, right_angle)
    if angle_gap > 90:
        angle_gap = 180 - angle_gap
    angle_score = _soft_score(angle_gap, good=20.0, bad=75.0)

    x1, y1, x2, y2 = normalize_xyxy(bbox)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    aspect = width / height
    if min_aspect <= aspect <= max_aspect:
        aspect_score = 1.0
    else:
        aspect_score = 0.4 if 0.2 <= aspect <= 8.0 else 0.0

    area_ratio = bbox_area(bbox) / max(float(image_size.width * image_size.height), 1.0)
    if min_area_ratio <= area_ratio <= max_area_ratio:
        area_score = 1.0
    else:
        area_score = 0.4 if 0.001 <= area_ratio <= 0.90 else 0.0

    contain_score = containment_rate(bbox, points)
    scores = [length_score, angle_score, aspect_score, area_score, contain_score]
    return float(clamp(sum(scores) / len(scores), 0.0, 1.0))


def normalized_keypoint_error(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    normalizer: float,
) -> float:
    pred = [tuple(map(float, point[:2])) for point in predicted if len(point) >= 2]
    tgt = [tuple(map(float, point[:2])) for point in target if len(point) >= 2]
    if len(pred) != len(tgt):
        raise ValueError("predicted and target keypoints must have the same number of points")
    if normalizer <= 0:
        normalizer = 1.0
    errors = [distance(pred_point, target_point) / normalizer for pred_point, target_point in zip(pred, tgt)]
    return float(sum(errors) / len(errors))


def pck(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    threshold: float,
) -> float:
    pred = [tuple(map(float, point[:2])) for point in predicted if len(point) >= 2]
    tgt = [tuple(map(float, point[:2])) for point in target if len(point) >= 2]
    if len(pred) != len(tgt):
        raise ValueError("predicted and target keypoints must have the same number of points")
    hits = [distance(pred_point, target_point) <= threshold for pred_point, target_point in zip(pred, tgt)]
    return float(sum(hits) / len(hits))


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    va = [float(value) for value in a]
    vb = [float(value) for value in b]
    denom = hypot(*va) * hypot(*vb)
    if denom <= 1e-8:
        return 0.0
    dot = sum(left * right for left, right in zip(va, vb))
    return float(clamp(dot / denom, -1.0, 1.0))


def angle_from_vectors(a: Iterable[float], b: Iterable[float]) -> float:
    return degrees(acos(cosine_similarity(a, b)))
