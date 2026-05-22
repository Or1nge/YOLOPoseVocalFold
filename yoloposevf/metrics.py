from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from yoloposevf.geometry import (
    ImageSize,
    bbox_iou,
    containment_rate,
    normalized_keypoint_error,
    pck,
    polygon_area,
    polygon_containment_rate,
    polygon_keypoint_containment_rate,
)

ROI_CONTAINMENT_TARGET = 0.87


@dataclass(frozen=True)
class SampleMetrics:
    source: str
    bbox_iou: float
    containment_rate: float
    normalized_keypoint_error: float
    pck: float
    action: str
    final_confidence: float
    roi_polygon_containment_rate: float | None = None
    roi_area_ratio_to_target: float | None = None


def evaluate_sample(
    source: str,
    predicted_bbox: Sequence[float],
    target_bbox: Sequence[float],
    predicted_keypoints: Sequence[Sequence[float]],
    target_keypoints: Sequence[Sequence[float]],
    image_size: ImageSize,
    action: str,
    final_confidence: float,
    predicted_roi_polygon: Sequence[Sequence[float]] | None = None,
    target_roi_polygon: Sequence[Sequence[float]] | None = None,
    pck_fraction_of_width: float = 0.02,
) -> SampleMetrics:
    normalizer = max(float(image_size.width), 1.0)
    threshold = max(float(image_size.width) * pck_fraction_of_width, 1.0)
    roi_containment = None
    roi_area_ratio = None
    if predicted_roi_polygon is not None and target_roi_polygon is not None:
        target_area = polygon_area(target_roi_polygon)
        pred_area = polygon_area(predicted_roi_polygon)
        roi_containment = polygon_containment_rate(target_roi_polygon, predicted_roi_polygon)
        roi_area_ratio = pred_area / target_area if target_area > 0 else None
    keypoint_containment = (
        polygon_keypoint_containment_rate(predicted_roi_polygon, predicted_keypoints)
        if predicted_roi_polygon is not None
        else containment_rate(predicted_bbox, predicted_keypoints)
    )
    return SampleMetrics(
        source=source,
        bbox_iou=bbox_iou(predicted_bbox, target_bbox),
        containment_rate=keypoint_containment,
        normalized_keypoint_error=normalized_keypoint_error(
            predicted_keypoints,
            target_keypoints,
            normalizer=normalizer,
        ),
        pck=pck(predicted_keypoints, target_keypoints, threshold=threshold),
        action=action,
        final_confidence=float(final_confidence),
        roi_polygon_containment_rate=roi_containment,
        roi_area_ratio_to_target=roi_area_ratio,
    )


def summarize_metrics(samples: Sequence[SampleMetrics]) -> dict[str, object]:
    if not samples:
        return {"count": 0}
    actions = Counter(sample.action for sample in samples)
    roi_containment = [
        sample.roi_polygon_containment_rate
        for sample in samples
        if sample.roi_polygon_containment_rate is not None
    ]
    roi_area_ratios = [
        sample.roi_area_ratio_to_target
        for sample in samples
        if sample.roi_area_ratio_to_target is not None
    ]
    summary = {
        "count": len(samples),
        "mean_bbox_iou": mean(sample.bbox_iou for sample in samples),
        "mean_containment_rate": mean(sample.containment_rate for sample in samples),
        "mean_normalized_keypoint_error": mean(sample.normalized_keypoint_error for sample in samples),
        "mean_pck": mean(sample.pck for sample in samples),
        "mean_final_confidence": mean(sample.final_confidence for sample in samples),
        "actions": dict(actions),
    }
    if roi_containment:
        summary["mean_roi_polygon_containment_rate"] = mean(roi_containment)
        summary["roi_polygon_containment_target"] = ROI_CONTAINMENT_TARGET
        summary["roi_polygon_containment_ge_87_rate"] = sum(
            value >= ROI_CONTAINMENT_TARGET for value in roi_containment
        ) / len(roi_containment)
    if roi_area_ratios:
        summary["mean_roi_area_ratio_to_target"] = mean(roi_area_ratios)
    return summary
