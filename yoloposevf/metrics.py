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
)


@dataclass(frozen=True)
class SampleMetrics:
    source: str
    bbox_iou: float
    containment_rate: float
    normalized_keypoint_error: float
    pck: float
    action: str
    final_confidence: float


def evaluate_sample(
    source: str,
    predicted_bbox: Sequence[float],
    target_bbox: Sequence[float],
    predicted_keypoints: Sequence[Sequence[float]],
    target_keypoints: Sequence[Sequence[float]],
    image_size: ImageSize,
    action: str,
    final_confidence: float,
    pck_fraction_of_width: float = 0.02,
) -> SampleMetrics:
    normalizer = max(float(image_size.width), 1.0)
    threshold = max(float(image_size.width) * pck_fraction_of_width, 1.0)
    return SampleMetrics(
        source=source,
        bbox_iou=bbox_iou(predicted_bbox, target_bbox),
        containment_rate=containment_rate(predicted_bbox, predicted_keypoints),
        normalized_keypoint_error=normalized_keypoint_error(
            predicted_keypoints,
            target_keypoints,
            normalizer=normalizer,
        ),
        pck=pck(predicted_keypoints, target_keypoints, threshold=threshold),
        action=action,
        final_confidence=float(final_confidence),
    )


def summarize_metrics(samples: Sequence[SampleMetrics]) -> dict[str, object]:
    if not samples:
        return {"count": 0}
    actions = Counter(sample.action for sample in samples)
    return {
        "count": len(samples),
        "mean_bbox_iou": mean(sample.bbox_iou for sample in samples),
        "mean_containment_rate": mean(sample.containment_rate for sample in samples),
        "mean_normalized_keypoint_error": mean(sample.normalized_keypoint_error for sample in samples),
        "mean_pck": mean(sample.pck for sample in samples),
        "mean_final_confidence": mean(sample.final_confidence for sample in samples),
        "actions": dict(actions),
    }

