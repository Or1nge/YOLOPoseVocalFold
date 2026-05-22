from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from yoloposevf.geometry import (
    BBox,
    ImageSize,
    angle_bisector_roi_from_three_points,
    bbox_area,
    bbox_iou,
    clip_bbox,
    containment_rate,
    geometry_score,
    included_angle_degrees,
    keypoint_bbox,
    polygon_area,
    polygon_keypoint_containment_rate,
    union_bbox,
)


@dataclass(frozen=True)
class PostprocessConfig:
    kp_margin_x: float = 0.18
    kp_margin_y: float = 0.18
    roi_base_backtrack_fraction: float = 0.10
    roi_posterior_margin_fraction: float = 0.18
    roi_side_margin_fraction: float = 0.18
    roi_min_base_backtrack_px: float = 4.0
    roi_min_posterior_margin_px: float = 4.0
    roi_min_side_margin_px: float = 4.0
    min_keypoint_conf: float = 0.35
    high_keypoint_conf: float = 0.65
    review_threshold: float = 0.50
    auto_accept_threshold: float = 0.80
    low_consistency_iou: float = 0.35
    min_glottic_angle_degrees: float = 20.0
    good_glottic_angle_degrees: float = 35.0
    max_glottic_angle_degrees: float = 130.0
    confidence_keypoint_mode: str = "mean"
    confidence_curve: str = "power"
    confidence_gamma: float = 1.0
    confidence_tanh_midpoint: float = 0.50
    confidence_tanh_steepness: float = 4.0
    confidence_consistency_weight: float = 0.25
    min_roi_area_ratio: float = 0.0
    good_roi_area_ratio: float = 0.0
    keypoint_image_bounds_tolerance_px: float = 0.0
    min_anterior_y_offset_ratio: float = 0.0
    good_anterior_y_offset_ratio: float = 0.0
    roi_dark_luma_threshold: float = 0.0
    min_roi_dark_fraction: float = 0.0
    good_roi_dark_fraction: float = 0.0
    fusion_mode: str = "angle_bisector"


@dataclass(frozen=True)
class PosePrediction:
    bbox: BBox
    bbox_conf: float
    keypoints: tuple[tuple[float, float, float], ...]
    image_size: ImageSize
    source: str | None = None


def _keypoint_confidence(keypoints: Sequence[Sequence[float]], mode: str) -> float:
    confs = [float(kp[2]) if len(kp) >= 3 else 1.0 for kp in keypoints]
    if not confs:
        return 0.0
    if mode == "min":
        return min(confs)
    return sum(confs) / len(confs)


def _select_keypoints(
    keypoints: Sequence[Sequence[float]],
    min_conf: float,
) -> list[tuple[float, float, float]]:
    selected = []
    for kp in keypoints:
        conf = float(kp[2]) if len(kp) >= 3 else 1.0
        if conf >= min_conf:
            selected.append((float(kp[0]), float(kp[1]), conf))
    return selected


def decide_action(final_confidence: float, cfg: PostprocessConfig) -> str:
    if final_confidence >= cfg.auto_accept_threshold:
        return "auto_accept"
    if final_confidence >= cfg.review_threshold:
        return "manual_review"
    return "reject_or_relabel"


def _bbox_to_polygon(bbox: Sequence[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _lower_bound_factor(value: float | None, low: float, good: float) -> float:
    low = max(float(low), 0.0)
    good = max(float(good), 0.0)
    if value is None or good <= 0.0:
        return 1.0
    if good <= low:
        return 1.0 if value >= low else 0.0
    if value >= good:
        return 1.0
    if value <= low:
        return 0.0
    return float((value - low) / (good - low))


def _confidence_factor(confidence: float, cfg: PostprocessConfig) -> float:
    confidence = max(0.0, min(1.0, float(confidence)))
    curve = str(cfg.confidence_curve).lower()
    if curve == "tanh":
        midpoint = max(0.0, min(1.0, float(cfg.confidence_tanh_midpoint)))
        steepness = max(float(cfg.confidence_tanh_steepness), 1e-6)
        low = math.tanh(steepness * (0.0 - midpoint))
        high = math.tanh(steepness * (1.0 - midpoint))
        if high <= low:
            return confidence
        value = math.tanh(steepness * (confidence - midpoint))
        return max(0.0, min(1.0, float((value - low) / (high - low))))
    if curve != "power":
        raise ValueError(f"Unsupported confidence_curve: {cfg.confidence_curve}")
    confidence_gamma = max(float(cfg.confidence_gamma), 0.0)
    return confidence**confidence_gamma


def _max_keypoint_outside_image_px(
    keypoints: Sequence[Sequence[float]],
    image_size: ImageSize,
) -> float:
    max_distance = 0.0
    for point in keypoints:
        x, y = float(point[0]), float(point[1])
        max_distance = max(max_distance, -x, x - image_size.width, -y, y - image_size.height)
    return max(0.0, max_distance)


def _anterior_y_offset_ratio(keypoints: Sequence[Sequence[float]]) -> float | None:
    if len(keypoints) != 3:
        return None
    anterior, left, right = keypoints
    posterior_mid_y = (float(left[1]) + float(right[1])) / 2.0
    posterior_width = math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
    if posterior_width <= 0.0:
        return None
    return float((float(anterior[1]) - posterior_mid_y) / posterior_width)


def fuse_prediction(prediction: PosePrediction, cfg: PostprocessConfig) -> dict[str, Any]:
    bbox_yolo = clip_bbox(prediction.bbox, prediction.image_size)
    keypoints = [tuple(map(float, kp)) for kp in prediction.keypoints]
    selected_kps = _select_keypoints(keypoints, cfg.min_keypoint_conf)
    flags: list[str] = []
    roi_polygon: list[list[float]] | None = None
    roi_area: float | None = None

    if len(selected_kps) == 3:
        roi = angle_bisector_roi_from_three_points(
            selected_kps,
            image_size=prediction.image_size,
            base_backtrack_fraction=cfg.roi_base_backtrack_fraction,
            posterior_margin_fraction=cfg.roi_posterior_margin_fraction,
            side_margin_fraction=cfg.roi_side_margin_fraction,
            min_base_backtrack_px=cfg.roi_min_base_backtrack_px,
            min_posterior_margin_px=cfg.roi_min_posterior_margin_px,
            min_side_margin_px=cfg.roi_min_side_margin_px,
        )
        bbox_kp = roi.bbox_xyxy
        roi_polygon = [list(point) for point in roi.polygon]
        roi_area = polygon_area(roi.polygon)
    elif len(selected_kps) >= 2:
        bbox_kp = keypoint_bbox(
            selected_kps,
            margin_x=cfg.kp_margin_x,
            margin_y=cfg.kp_margin_y,
            image_size=prediction.image_size,
        )
    else:
        bbox_kp = bbox_yolo
        flags.append("too_few_reliable_keypoints")

    if cfg.fusion_mode in {"angle_bisector", "keypoints"} and len(selected_kps) == 3:
        final_bbox = bbox_kp
        final_box_polygon = roi_polygon
    elif cfg.fusion_mode == "keypoints" and len(selected_kps) >= 2:
        final_bbox = bbox_kp
        final_box_polygon = _bbox_to_polygon(final_bbox)
    elif cfg.fusion_mode == "yolo":
        final_bbox = bbox_yolo
        if selected_kps:
            final_bbox = union_bbox(final_bbox, keypoint_bbox(selected_kps, 0.02, 0.02), image_size=prediction.image_size)
        final_box_polygon = _bbox_to_polygon(final_bbox)
    else:
        final_bbox = union_bbox(bbox_yolo, bbox_kp, image_size=prediction.image_size)
        final_box_polygon = _bbox_to_polygon(final_bbox)

    consistency = bbox_iou(bbox_yolo, bbox_kp)
    if consistency < cfg.low_consistency_iou:
        flags.append("low_bbox_keypoint_consistency")

    glottic_angle = (
        included_angle_degrees(keypoints[0], keypoints[1], keypoints[2])
        if len(keypoints) == 3
        else None
    )
    if glottic_angle is not None and (
        glottic_angle < cfg.min_glottic_angle_degrees
        or glottic_angle > cfg.max_glottic_angle_degrees
    ):
        flags.append("implausible_keypoint_angle")

    geom = geometry_score(
        keypoints,
        final_bbox,
        prediction.image_size,
        min_glottic_angle_degrees=cfg.min_glottic_angle_degrees,
        good_glottic_angle_degrees=cfg.good_glottic_angle_degrees,
        max_glottic_angle_degrees=cfg.max_glottic_angle_degrees,
    )
    kp_conf = _keypoint_confidence(keypoints, cfg.confidence_keypoint_mode)
    min_kp_conf = _keypoint_confidence(keypoints, "min")
    if min_kp_conf < cfg.min_keypoint_conf:
        flags.append("low_keypoint_confidence")

    keypoint_xy = [kp[:2] for kp in keypoints]
    contained = (
        polygon_keypoint_containment_rate(final_box_polygon, keypoint_xy)
        if final_box_polygon is not None
        else containment_rate(final_bbox, keypoint_xy)
    )
    if contained < 1.0:
        flags.append("keypoints_outside_final_box")

    image_area = max(float(prediction.image_size.width * prediction.image_size.height), 1.0)
    final_bbox_area_ratio = bbox_area(final_bbox) / image_area
    roi_area_ratio = roi_area / image_area if roi_area is not None else None
    roi_area_factor = _lower_bound_factor(roi_area_ratio, cfg.min_roi_area_ratio, cfg.good_roi_area_ratio)
    if roi_area_ratio is not None and cfg.good_roi_area_ratio > 0.0 and roi_area_ratio < cfg.good_roi_area_ratio:
        flags.append("roi_area_too_small" if roi_area_ratio <= cfg.min_roi_area_ratio else "low_roi_area")

    keypoint_outside_image_px = _max_keypoint_outside_image_px(keypoints, prediction.image_size)
    image_bounds_factor = 1.0
    if keypoint_outside_image_px > max(float(cfg.keypoint_image_bounds_tolerance_px), 0.0):
        flags.append("keypoints_outside_image")
        image_bounds_factor = 0.0

    anterior_offset_ratio = _anterior_y_offset_ratio(keypoints)
    anterior_position_factor = _lower_bound_factor(
        anterior_offset_ratio,
        cfg.min_anterior_y_offset_ratio,
        cfg.good_anterior_y_offset_ratio,
    )
    if (
        anterior_offset_ratio is not None
        and cfg.good_anterior_y_offset_ratio > 0.0
        and anterior_offset_ratio < cfg.good_anterior_y_offset_ratio
    ):
        flags.append(
            "anterior_point_not_below_posterior_points"
            if anterior_offset_ratio <= cfg.min_anterior_y_offset_ratio
            else "weak_anterior_posterior_orientation"
        )

    bbox_conf_factor = _confidence_factor(float(prediction.bbox_conf), cfg)
    keypoint_conf_factor = _confidence_factor(kp_conf, cfg)
    consistency_factor = consistency ** max(float(cfg.confidence_consistency_weight), 0.0)
    final_confidence = float(
        bbox_conf_factor
        * keypoint_conf_factor
        * geom
        * consistency_factor
        * roi_area_factor
        * image_bounds_factor
        * anterior_position_factor
    )
    final_confidence = max(0.0, min(1.0, final_confidence))
    action = decide_action(final_confidence, cfg)
    usable_bbox = list(final_bbox) if action != "reject_or_relabel" else None
    usable_box_polygon = final_box_polygon if action != "reject_or_relabel" else None

    return {
        "source": prediction.source,
        "bbox_yolo": list(bbox_yolo),
        "bbox_keypoints": list(bbox_kp),
        "roi_polygon": roi_polygon,
        "roi_polygon_area": roi_area,
        "final_bbox": list(final_bbox),
        "final_bbox_xyxy": list(final_bbox),
        "final_box_polygon": final_box_polygon,
        "usable_bbox": usable_bbox,
        "usable_box_polygon": usable_box_polygon,
        "bbox_confidence": float(prediction.bbox_conf),
        "keypoint_confidence": kp_conf,
        "min_keypoint_confidence": min_kp_conf,
        "glottic_angle_degrees": glottic_angle,
        "geometry_score": geom,
        "consistency_score": consistency,
        "roi_area_ratio": roi_area_ratio,
        "roi_area_factor": roi_area_factor,
        "final_bbox_area_ratio": final_bbox_area_ratio,
        "max_keypoint_outside_image_px": keypoint_outside_image_px,
        "image_bounds_factor": image_bounds_factor,
        "anterior_y_offset_ratio": anterior_offset_ratio,
        "anterior_position_factor": anterior_position_factor,
        "bbox_confidence_factor": bbox_conf_factor,
        "keypoint_confidence_factor": keypoint_conf_factor,
        "containment_rate": contained,
        "final_confidence": final_confidence,
        "action": action,
        "flags": flags,
        "config": asdict(cfg),
    }


def load_postprocess_config(values: dict[str, Any] | None) -> PostprocessConfig:
    if values is None:
        return PostprocessConfig()
    valid = {field.name for field in PostprocessConfig.__dataclass_fields__.values()}
    filtered = {key: value for key, value in values.items() if key in valid}
    return PostprocessConfig(**filtered)
