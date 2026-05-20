from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContainmentLossConfig:
    """Configuration for the predicted-bbox/keypoint containment penalty."""

    margin: float = 0.0
    reduction: str = "mean"
    normalize_by_box_size: bool = True


def containment_penalty_numpy(
    boxes_xyxy: Sequence[Sequence[float]] | np.ndarray,
    keypoints_xy: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    visibility: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    margin: float = 0.0,
    reduction: str = "mean",
    normalize_by_box_size: bool = True,
) -> float | np.ndarray:
    """Return a hinge penalty for keypoints outside predicted boxes.

    Parameters use absolute or normalized coordinates, but boxes and keypoints
    must share the same coordinate system. `boxes_xyxy` is shaped `(N, 4)`;
    `keypoints_xy` is shaped `(N, K, 2)`. Visibility values greater than zero
    contribute to the loss.
    """

    boxes = _as_float_array(boxes_xyxy, ndim=2, name="boxes_xyxy")
    keypoints = _as_float_array(keypoints_xy, ndim=3, name="keypoints_xy")
    if boxes.shape[0] != keypoints.shape[0] or boxes.shape[1] != 4 or keypoints.shape[2] != 2:
        raise ValueError("expected boxes shape (N, 4) and keypoints shape (N, K, 2)")

    distances = _outside_distances_numpy(
        boxes,
        keypoints,
        margin=margin,
        normalize_by_box_size=normalize_by_box_size,
    )
    if visibility is not None:
        mask = _as_float_array(visibility, ndim=2, name="visibility") > 0
        if mask.shape != distances.shape:
            raise ValueError("visibility must have shape (N, K)")
        distances = np.where(mask, distances, 0.0)
        denom = np.maximum(mask.sum(axis=1), 1)
    else:
        denom = np.full(distances.shape[0], distances.shape[1], dtype=np.float64)

    per_sample = (distances**2).sum(axis=1) / denom
    return _reduce_numpy(per_sample, reduction)


def containment_penalty_torch(
    boxes_xyxy,
    keypoints_xy,
    visibility=None,
    *,
    margin: float = 0.0,
    reduction: str = "mean",
    normalize_by_box_size: bool = True,
):
    """Torch variant used by the experimental Ultralytics hook.

    Torch is imported lazily so pure unit tests can run without a training stack.
    """

    import torch

    boxes = boxes_xyxy.float()
    keypoints = keypoints_xy.float()
    if boxes.ndim != 2 or keypoints.ndim != 3 or boxes.shape[1] != 4 or keypoints.shape[2] != 2:
        raise ValueError("expected boxes shape (N, 4) and keypoints shape (N, K, 2)")
    if boxes.shape[0] != keypoints.shape[0]:
        raise ValueError("boxes and keypoints batch dimensions must match")

    x1, y1, x2, y2 = _sorted_box_edges_torch(boxes)
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    dx = torch.relu((x1[:, None] - margin) - x) + torch.relu(x - (x2[:, None] + margin))
    dy = torch.relu((y1[:, None] - margin) - y) + torch.relu(y - (y2[:, None] + margin))
    distances = dx + dy
    if normalize_by_box_size:
        scale = torch.clamp(torch.maximum(x2 - x1, y2 - y1), min=1e-6)
        distances = distances / scale[:, None]

    if visibility is not None:
        mask = visibility.float() > 0
        if mask.shape != distances.shape:
            raise ValueError("visibility must have shape (N, K)")
        distances = torch.where(mask, distances, torch.zeros_like(distances))
        denom = torch.clamp(mask.sum(dim=1).float(), min=1.0)
    else:
        denom = torch.full(
            (distances.shape[0],),
            distances.shape[1],
            dtype=distances.dtype,
            device=distances.device,
        )

    per_sample = (distances.square().sum(dim=1) / denom)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be one of: none, mean, sum")


def _outside_distances_numpy(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    *,
    margin: float,
    normalize_by_box_size: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = _sorted_box_edges_numpy(boxes)
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    dx = np.maximum((x1[:, None] - margin) - x, 0.0) + np.maximum(x - (x2[:, None] + margin), 0.0)
    dy = np.maximum((y1[:, None] - margin) - y, 0.0) + np.maximum(y - (y2[:, None] + margin), 0.0)
    distances = dx + dy
    if normalize_by_box_size:
        scale = np.maximum(np.maximum(x2 - x1, y2 - y1), 1e-6)
        distances = distances / scale[:, None]
    return distances


def _as_float_array(values: Sequence | np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    return array


def _sorted_box_edges_numpy(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    return x1, y1, x2, y2


def _sorted_box_edges_torch(boxes) -> tuple:
    import torch

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    return x1, y1, x2, y2


def _reduce_numpy(per_sample: np.ndarray, reduction: str) -> float | np.ndarray:
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return float(per_sample.sum())
    if reduction == "mean":
        return float(per_sample.mean())
    raise ValueError("reduction must be one of: none, mean, sum")
