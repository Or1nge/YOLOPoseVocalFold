from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yoloposevf.geometry import ImageSize, xyxy_to_yolo, yolo_to_xyxy


@dataclass(frozen=True)
class YoloPoseLabel:
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, float], ...]
    image_size: ImageSize

    def to_line(self) -> str:
        x_center, y_center, width, height = xyxy_to_yolo(self.bbox_xyxy, self.image_size)
        values: list[float | int] = [self.class_id, x_center, y_center, width, height]
        for x, y, visibility in self.keypoints:
            values.extend([x / self.image_size.width, y / self.image_size.height, int(visibility)])
        return " ".join(_format_yolo_value(value) for value in values)


def _format_yolo_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}".rstrip("0").rstrip(".")


def read_yolo_pose_label(label_path: Path, image_size: ImageSize) -> YoloPoseLabel:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label_path} must contain exactly one ROI object, found {len(lines)}")
    parts = lines[0].split()
    if len(parts) < 8 or (len(parts) - 5) % 3 != 0:
        raise ValueError(
            f"{label_path} must have class bbox plus N keypoints: got {len(parts)} values"
        )
    values = [float(part) for part in parts]
    class_id = int(values[0])
    bbox_xyxy = yolo_to_xyxy(values[1], values[2], values[3], values[4], image_size)
    keypoints = []
    for idx in range(5, len(values), 3):
        keypoints.append(
            (
                values[idx] * image_size.width,
                values[idx + 1] * image_size.height,
                values[idx + 2],
            )
        )
    return YoloPoseLabel(class_id=class_id, bbox_xyxy=bbox_xyxy, keypoints=tuple(keypoints), image_size=image_size)
