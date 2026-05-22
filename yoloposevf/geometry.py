from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, hypot, pi
from typing import Iterable, Sequence


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class OrientedROI:
    polygon: Polygon
    bbox_xyxy: BBox
    direction: Point
    normal: Point
    base_center: Point
    height: float
    half_width: float


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


def points_bbox(points: Sequence[Sequence[float]], image_size: ImageSize | None = None) -> BBox:
    if not points:
        raise ValueError("points must contain at least one point")
    x1 = min(float(point[0]) for point in points)
    y1 = min(float(point[1]) for point in points)
    x2 = max(float(point[0]) for point in points)
    y2 = max(float(point[1]) for point in points)
    bbox = normalize_xyxy((x1, y1, x2, y2))
    if image_size is not None:
        return clip_bbox(bbox, image_size)
    return bbox


def _unit_vector(vector: Sequence[float], fallback: Sequence[float] = (1.0, 0.0)) -> Point:
    length = hypot(float(vector[0]), float(vector[1]))
    if length <= 1e-8:
        fallback_length = hypot(float(fallback[0]), float(fallback[1]))
        if fallback_length <= 1e-8:
            return 1.0, 0.0
        return float(fallback[0]) / fallback_length, float(fallback[1]) / fallback_length
    return float(vector[0]) / length, float(vector[1]) / length


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _sub(a: Sequence[float], b: Sequence[float]) -> Point:
    return float(a[0]) - float(b[0]), float(a[1]) - float(b[1])


def _add(a: Sequence[float], b: Sequence[float]) -> Point:
    return float(a[0]) + float(b[0]), float(a[1]) + float(b[1])


def _mul(a: Sequence[float], scale: float) -> Point:
    return float(a[0]) * scale, float(a[1]) * scale


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def angle_bisector_direction(
    anterior: Sequence[float],
    left_posterior: Sequence[float],
    right_posterior: Sequence[float],
) -> Point:
    left_vec = _sub(left_posterior, anterior)
    right_vec = _sub(right_posterior, anterior)
    midpoint_vec = (
        (float(left_posterior[0]) + float(right_posterior[0])) / 2.0 - float(anterior[0]),
        (float(left_posterior[1]) + float(right_posterior[1])) / 2.0 - float(anterior[1]),
    )
    left_unit = _unit_vector(left_vec, fallback=midpoint_vec)
    right_unit = _unit_vector(right_vec, fallback=midpoint_vec)
    bisector = (left_unit[0] + right_unit[0], left_unit[1] + right_unit[1])
    return _unit_vector(bisector, fallback=midpoint_vec)


def included_angle_degrees(
    vertex: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    first_vec = _sub(first, vertex)
    second_vec = _sub(second, vertex)
    first_length = hypot(first_vec[0], first_vec[1])
    second_length = hypot(second_vec[0], second_vec[1])
    if first_length <= 1e-8 or second_length <= 1e-8:
        return 0.0
    cosine = clamp(_dot(first_vec, second_vec) / (first_length * second_length), -1.0, 1.0)
    return float(degrees(acos(cosine)))


def angle_bisector_roi_from_three_points(
    keypoints: Sequence[Sequence[float]],
    image_size: ImageSize | None = None,
    base_backtrack_fraction: float = 0.10,
    posterior_margin_fraction: float = 0.18,
    side_margin_fraction: float = 0.18,
    min_base_backtrack_px: float = 4.0,
    min_posterior_margin_px: float = 4.0,
    min_side_margin_px: float = 4.0,
) -> OrientedROI:
    points = [tuple(map(float, point[:2])) for point in keypoints if len(point) >= 2]
    if len(points) != 3:
        raise ValueError("angle-bisector ROI requires exactly 3 keypoints")

    anterior, left_posterior, right_posterior = points
    direction = angle_bisector_direction(anterior, left_posterior, right_posterior)
    normal = (-direction[1], direction[0])
    posterior_offsets = [_sub(left_posterior, anterior), _sub(right_posterior, anterior)]
    posterior_depths = [_dot(offset, direction) for offset in posterior_offsets]
    posterior_depth = max(posterior_depths)
    if posterior_depth <= 1e-6:
        midpoint = (
            (left_posterior[0] + right_posterior[0]) / 2.0,
            (left_posterior[1] + right_posterior[1]) / 2.0,
        )
        posterior_depth = max(distance(anterior, midpoint), distance(anterior, left_posterior), 1.0)

    lateral_offsets = [_dot(offset, normal) for offset in posterior_offsets]
    negative_extent = max((-offset for offset in lateral_offsets if offset < 0.0), default=0.0)
    positive_extent = max((offset for offset in lateral_offsets if offset > 0.0), default=0.0)

    backtrack = max(min_base_backtrack_px, posterior_depth * max(base_backtrack_fraction, 0.0))
    posterior_margin = max(
        min_posterior_margin_px,
        posterior_depth * max(posterior_margin_fraction, 0.0),
    )
    side_margin_fraction = max(side_margin_fraction, 0.0)
    negative_half_width = negative_extent + max(min_side_margin_px, negative_extent * side_margin_fraction)
    positive_half_width = positive_extent + max(min_side_margin_px, positive_extent * side_margin_fraction)
    half_width = max(negative_half_width, positive_half_width)
    height = backtrack + posterior_depth + posterior_margin

    base_center = _add(anterior, _mul(direction, -backtrack))
    far_center = _add(base_center, _mul(direction, height))
    left_base = _add(base_center, _mul(normal, -negative_half_width))
    right_base = _add(base_center, _mul(normal, positive_half_width))
    right_far = _add(far_center, _mul(normal, positive_half_width))
    left_far = _add(far_center, _mul(normal, -negative_half_width))
    polygon: Polygon = (left_base, right_base, right_far, left_far)
    bbox = points_bbox(polygon, image_size=image_size)
    return OrientedROI(
        polygon=polygon,
        bbox_xyxy=bbox,
        direction=direction,
        normal=normal,
        base_center=base_center,
        height=float(height),
        half_width=float(half_width),
    )


def polygon_signed_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return area / 2.0


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(polygon_signed_area(points))


def _line_intersection(
    start: Sequence[float],
    end: Sequence[float],
    clip_start: Sequence[float],
    clip_end: Sequence[float],
) -> Point:
    segment = _sub(end, start)
    edge = _sub(clip_end, clip_start)
    denom = _cross(segment, edge)
    if abs(denom) <= 1e-8:
        return float(end[0]), float(end[1])
    t = _cross(_sub(clip_start, start), edge) / denom
    return float(start[0]) + segment[0] * t, float(start[1]) + segment[1] * t


def convex_polygon_intersection(
    subject_polygon: Sequence[Sequence[float]],
    clip_polygon: Sequence[Sequence[float]],
) -> Polygon:
    if len(subject_polygon) < 3 or len(clip_polygon) < 3:
        return tuple()

    output = [tuple(map(float, point[:2])) for point in subject_polygon]
    clip = [tuple(map(float, point[:2])) for point in clip_polygon]
    clip_sign = 1.0 if polygon_signed_area(clip) >= 0 else -1.0

    def inside(point: Sequence[float], edge_start: Sequence[float], edge_end: Sequence[float]) -> bool:
        return clip_sign * _cross(_sub(edge_end, edge_start), _sub(point, edge_start)) >= -1e-6

    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            current_inside = inside(current, clip_start, clip_end)
            previous_inside = inside(previous, clip_start, clip_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
    return tuple(output)


def polygon_containment_rate(
    target_polygon: Sequence[Sequence[float]],
    containing_polygon: Sequence[Sequence[float]],
) -> float:
    target_area = polygon_area(target_polygon)
    if target_area <= 1e-8:
        return 0.0
    intersection = convex_polygon_intersection(target_polygon, containing_polygon)
    return float(clamp(polygon_area(intersection) / target_area, 0.0, 1.0))


def contains_point_in_polygon(
    polygon: Sequence[Sequence[float]],
    point: Sequence[float],
    tolerance: float = 1e-6,
) -> bool:
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        edge_x, edge_y = x2 - x1, y2 - y1
        point_x, point_y = x - x1, y - y1
        cross = edge_x * point_y - edge_y * point_x
        dot = point_x * edge_x + point_y * edge_y
        edge_len_sq = edge_x * edge_x + edge_y * edge_y
        if abs(cross) <= tolerance and -tolerance <= dot <= edge_len_sq + tolerance:
            return True
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= x_intersect + tolerance:
                inside = not inside
        previous = current
    return inside


def polygon_keypoint_containment_rate(
    polygon: Sequence[Sequence[float]],
    keypoints: Sequence[Sequence[float]],
) -> float:
    points = list(keypoints)
    if not points:
        return 0.0
    inside = sum(1 for point in points if contains_point_in_polygon(polygon, point))
    return inside / len(points)


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
    min_glottic_angle_degrees: float = 20.0,
    good_glottic_angle_degrees: float = 35.0,
    max_glottic_angle_degrees: float = 130.0,
) -> float:
    points = [tuple(map(float, point[:2])) for point in keypoints if len(point) >= 2]
    if len(points) == 3:
        anterior, left_posterior, right_posterior = points
        glottic_angle = included_angle_degrees(anterior, left_posterior, right_posterior)
        if glottic_angle < min_glottic_angle_degrees or glottic_angle > max_glottic_angle_degrees:
            return 0.0
        angle_score = _soft_score(
            glottic_angle,
            good=good_glottic_angle_degrees,
            bad=min_glottic_angle_degrees,
            larger_is_worse=False,
        )

        direction = angle_bisector_direction(anterior, left_posterior, right_posterior)
        posterior_offsets = [_sub(left_posterior, anterior), _sub(right_posterior, anterior)]
        posterior_depths = [_dot(offset, direction) for offset in posterior_offsets]
        posterior_score = sum(1 for value in posterior_depths if value > 0) / 2.0

        normal = (-direction[1], direction[0])
        lateral_values = [_dot(offset, normal) for offset in posterior_offsets]
        lateral_separation = abs(lateral_values[0] - lateral_values[1])
        depth = max(max(posterior_depths), 1.0)
        opening_score = _soft_score(lateral_separation / depth, good=0.15, bad=0.02, larger_is_worse=False)

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
            area_score = 0.4 if 0.001 <= area_ratio <= 0.95 else 0.0

        contain_score = containment_rate(bbox, points)
        scores = [posterior_score, opening_score, angle_score, aspect_score, area_score, contain_score]
        return float(clamp(sum(scores) / len(scores), 0.0, 1.0))

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
