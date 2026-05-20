from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from yoloposevf.geometry import (
    BBox,
    ImageSize,
    bbox_iou,
    clip_bbox,
    containment_rate,
    expand_bbox,
    geometry_score,
    keypoint_bbox,
    union_bbox,
)


@dataclass(frozen=True)
class PostprocessConfig:
    kp_margin_x: float = 0.18
    kp_margin_y: float = 0.18
    min_keypoint_conf: float = 0.35
    high_keypoint_conf: float = 0.65
    review_threshold: float = 0.50
    auto_accept_threshold: float = 0.80
    low_consistency_iou: float = 0.35
    confidence_keypoint_mode: str = "mean"
    fusion_mode: str = "union"


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


def fuse_prediction(prediction: PosePrediction, cfg: PostprocessConfig) -> dict[str, Any]:
    bbox_yolo = clip_bbox(prediction.bbox, prediction.image_size)
    keypoints = [tuple(map(float, kp)) for kp in prediction.keypoints]
    selected_kps = _select_keypoints(keypoints, cfg.min_keypoint_conf)
    flags: list[str] = []

    if len(selected_kps) >= 2:
        bbox_kp = keypoint_bbox(
            selected_kps,
            margin_x=cfg.kp_margin_x,
            margin_y=cfg.kp_margin_y,
            image_size=prediction.image_size,
        )
    else:
        bbox_kp = bbox_yolo
        flags.append("too_few_reliable_keypoints")

    if cfg.fusion_mode == "keypoints" and len(selected_kps) == 4:
        final_bbox = bbox_kp
    elif cfg.fusion_mode == "yolo":
        final_bbox = bbox_yolo
        if selected_kps:
            final_bbox = union_bbox(final_bbox, keypoint_bbox(selected_kps, 0.02, 0.02), image_size=prediction.image_size)
    else:
        final_bbox = union_bbox(bbox_yolo, bbox_kp, image_size=prediction.image_size)

    consistency = bbox_iou(bbox_yolo, bbox_kp)
    if consistency < cfg.low_consistency_iou:
        flags.append("low_bbox_keypoint_consistency")

    geom = geometry_score(keypoints, final_bbox, prediction.image_size)
    kp_conf = _keypoint_confidence(keypoints, cfg.confidence_keypoint_mode)
    min_kp_conf = _keypoint_confidence(keypoints, "min")
    if min_kp_conf < cfg.min_keypoint_conf:
        flags.append("low_keypoint_confidence")

    contained = containment_rate(final_bbox, [kp[:2] for kp in keypoints])
    if contained < 1.0:
        flags.append("keypoints_outside_final_bbox")

    final_confidence = float(prediction.bbox_conf * kp_conf * geom * consistency)
    final_confidence = max(0.0, min(1.0, final_confidence))
    action = decide_action(final_confidence, cfg)

    return {
        "source": prediction.source,
        "bbox_yolo": list(bbox_yolo),
        "bbox_keypoints": list(bbox_kp),
        "final_bbox": list(final_bbox),
        "bbox_confidence": float(prediction.bbox_conf),
        "keypoint_confidence": kp_conf,
        "min_keypoint_confidence": min_kp_conf,
        "geometry_score": geom,
        "consistency_score": consistency,
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
