from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class BlackPadInfo:
    type: str
    original_width: int
    original_height: int
    crop_bbox_xyxy: tuple[int, int, int, int]
    crop_was_applied: bool
    crop_fallback: str | None
    cropped_width: int
    cropped_height: int
    cropped_source: str | None
    padding_px: int
    padding_fraction: float
    padding_min_px: int
    padded_width: int
    padded_height: int
    model_input_width: int
    model_input_height: int
    no_black_width: int
    no_black_height: int
    no_black_bbox_in_model_input: tuple[int, int, int, int]
    black_border_luma_floor: float
    edge_artifact_crop_bbox_xyxy: tuple[int, int, int, int] | None
    edge_artifact_crop_was_applied: bool
    edge_artifact_crop_reason: str | None
    edge_artifact_dark_fraction: float | None
    edge_artifact_border_dark_fraction: float | None
    edge_artifact_mask_fraction: float | None

    def to_dict(self) -> dict[str, int | float | str | bool | None | list[int]]:
        values = asdict(self)
        values["crop_bbox_xyxy"] = list(self.crop_bbox_xyxy)
        values["no_black_bbox_in_model_input"] = list(self.no_black_bbox_in_model_input)
        if self.edge_artifact_crop_bbox_xyxy is not None:
            values["edge_artifact_crop_bbox_xyxy"] = list(self.edge_artifact_crop_bbox_xyxy)
        return values


@dataclass(frozen=True)
class EdgeArtifactCropInfo:
    bbox_xyxy: tuple[int, int, int, int]
    was_applied: bool
    reason: str | None
    dark_fraction: float | None
    border_dark_fraction: float | None
    mask_fraction: float | None


def blackpad_padding(width: int, height: int, fraction: float = 0.30, min_padding: int = 80) -> int:
    return max(int(min_padding), int(round(max(width, height) * float(fraction))))


def foreground_bbox_for_image(image: Image.Image, *, luma_floor: float = 8.0) -> tuple[int, int, int, int]:
    width, height = image.size
    threshold = float(luma_floor)
    mask = image.convert("L").point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return 0, 0, width, height
    left, top, right, bottom = bbox
    return int(left), int(top), int(right), int(bottom)


def _empty_edge_artifact_info(width: int, height: int, reason: str | None = None) -> EdgeArtifactCropInfo:
    return EdgeArtifactCropInfo(
        bbox_xyxy=(0, 0, int(width), int(height)),
        was_applied=False,
        reason=reason,
        dark_fraction=None,
        border_dark_fraction=None,
        mask_fraction=None,
    )


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _edge_band_mask(height: int, width: int, fraction: float = 0.08) -> np.ndarray:
    band = max(2, int(round(min(height, width) * float(fraction))))
    band = min(band, max(height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            return np.zeros_like(mask, dtype=bool)
        border_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        border_labels = border_labels[border_labels != 0]
        if border_labels.size == 0:
            return np.zeros_like(mask, dtype=bool)
        return np.isin(labels, border_labels)
    except ImportError:
        pass

    height, width = mask.shape
    connected = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        if mask[0, x]:
            stack.append((0, x))
        if mask[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if mask[y, 0]:
            stack.append((y, 0))
        if mask[y, width - 1]:
            stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if connected[y, x] or not mask[y, x]:
            continue
        connected[y, x] = True
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if not connected[ny, nx] and mask[ny, nx]:
                    stack.append((ny, nx))
    return connected


def detect_edge_artifact_crop(
    image: Image.Image,
    *,
    black_border_luma_floor: float = 8.0,
    min_dark_fraction: float = 0.035,
    min_border_dark_fraction: float = 0.12,
    min_side_dark_fraction: float = 0.28,
    min_removed_margin_fraction: float = 0.012,
) -> EdgeArtifactCropInfo:
    """Find black edge artifacts and white/gray outside regions in one crop.

    The trigger is deliberately based on dark pixels near the outer edge. Once
    that trigger is met, the crop removes every edge-connected artifact pixel:
    near-black border/background plus neutral bright UI/corner regions attached
    to it. This keeps the old pure-black-border behavior while handling rounded
    black frames with white corners.
    """
    image = image.convert("RGB")
    width, height = image.size
    if width < 8 or height < 8:
        return _empty_edge_artifact_info(width, height, "too_small")

    arr = np.asarray(image, dtype=np.uint8)
    arr_i = arr.astype(np.int16)
    r = arr_i[:, :, 0]
    g = arr_i[:, :, 1]
    b = arr_i[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = arr_i.max(axis=2) - arr_i.min(axis=2)

    dark_threshold = max(float(black_border_luma_floor), 24.0)
    dark = luma <= dark_threshold
    edge_band = _edge_band_mask(height, width)
    border_dark_fraction = float(dark[edge_band].mean())
    dark_fraction = float(dark.mean())

    band = max(2, int(round(min(height, width) * 0.08)))
    side_dark_fraction = max(
        float(dark[:band, :].mean()),
        float(dark[-band:, :].mean()),
        float(dark[:, :band].mean()),
        float(dark[:, -band:].mean()),
    )
    has_edge_black = (
        dark_fraction >= float(min_dark_fraction)
        and (
            border_dark_fraction >= float(min_border_dark_fraction)
            or side_dark_fraction >= float(min_side_dark_fraction)
        )
    )
    if not has_edge_black:
        return EdgeArtifactCropInfo(
            bbox_xyxy=(0, 0, int(width), int(height)),
            was_applied=False,
            reason="no_large_edge_black_region",
            dark_fraction=dark_fraction,
            border_dark_fraction=border_dark_fraction,
            mask_fraction=None,
        )

    neutral_bright = ((luma >= 150.0) & (sat <= 65)) | (luma >= 242.0)
    edge_artifact = _edge_connected_mask(dark | neutral_bright)
    mask_fraction = float(edge_artifact.mean())
    content_mask = ~edge_artifact
    bbox = _mask_bbox(content_mask)
    if bbox is None:
        return EdgeArtifactCropInfo(
            bbox_xyxy=(0, 0, int(width), int(height)),
            was_applied=False,
            reason="edge_artifact_removed_all_pixels",
            dark_fraction=dark_fraction,
            border_dark_fraction=border_dark_fraction,
            mask_fraction=mask_fraction,
        )

    left, top, right, bottom = bbox
    removed_margin = max(left, top, width - right, height - bottom)
    min_removed_margin = max(1, int(round(min(width, height) * float(min_removed_margin_fraction))))
    crop_area = max(right - left, 0) * max(bottom - top, 0)
    full_area = max(width * height, 1)
    if removed_margin < min_removed_margin or crop_area < full_area * 0.04:
        return EdgeArtifactCropInfo(
            bbox_xyxy=(0, 0, int(width), int(height)),
            was_applied=False,
            reason="edge_artifact_crop_not_plausible",
            dark_fraction=dark_fraction,
            border_dark_fraction=border_dark_fraction,
            mask_fraction=mask_fraction,
        )

    return EdgeArtifactCropInfo(
        bbox_xyxy=(int(left), int(top), int(right), int(bottom)),
        was_applied=True,
        reason="large_edge_black_region",
        dark_fraction=dark_fraction,
        border_dark_fraction=border_dark_fraction,
        mask_fraction=mask_fraction,
    )


def _combined_black_border_bbox(
    image: Image.Image,
    *,
    luma_floor: float = 8.0,
) -> tuple[tuple[int, int, int, int], str | None, EdgeArtifactCropInfo]:
    width, height = image.size
    edge_info = detect_edge_artifact_crop(image, black_border_luma_floor=luma_floor)
    outer_left, outer_top, outer_right, outer_bottom = edge_info.bbox_xyxy
    edge_cropped = image.crop(edge_info.bbox_xyxy)
    inner_bbox, crop_fallback = _foreground_bbox_and_fallback(edge_cropped, luma_floor=luma_floor)
    inner_left, inner_top, inner_right, inner_bottom = inner_bbox
    crop_bbox = (
        max(0, min(width, outer_left + inner_left)),
        max(0, min(height, outer_top + inner_top)),
        max(0, min(width, outer_left + inner_right)),
        max(0, min(height, outer_top + inner_bottom)),
    )
    if crop_bbox[0] >= crop_bbox[2] or crop_bbox[1] >= crop_bbox[3]:
        return (0, 0, width, height), "combined_crop_empty", edge_info
    return crop_bbox, crop_fallback, edge_info


def _foreground_bbox_and_fallback(
    image: Image.Image,
    *,
    luma_floor: float = 8.0,
) -> tuple[tuple[int, int, int, int], str | None]:
    width, height = image.size
    threshold = float(luma_floor)
    mask = image.convert("L").point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, width, height), "no_foreground_luma_above_floor"
    left, top, right, bottom = bbox
    return (int(left), int(top), int(right), int(bottom)), None


def crop_existing_black_borders(
    image: Image.Image,
    *,
    luma_floor: float = 8.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = image.convert("RGB")
    bbox, _, _ = _combined_black_border_bbox(image, luma_floor=luma_floor)
    cropped = image.crop(bbox)
    return cropped, bbox


def _save_rgb_image(image: Image.Image, destination: Path, *, jpeg_quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = int(jpeg_quality)
    image.save(destination, **save_kwargs)


def _preprocess_info(
    *,
    preprocess_type: str,
    original_width: int,
    original_height: int,
    crop_bbox: tuple[int, int, int, int],
    crop_fallback: str | None,
    cropped_width: int,
    cropped_height: int,
    cropped_source: Path | None,
    padding_px: int,
    padding_fraction: float,
    padding_min_px: int,
    model_input_width: int,
    model_input_height: int,
    black_border_luma_floor: float,
    edge_artifact_info: EdgeArtifactCropInfo | None = None,
) -> BlackPadInfo:
    crop_was_applied = crop_bbox != (0, 0, int(original_width), int(original_height))
    if edge_artifact_info is None:
        edge_artifact_info = _empty_edge_artifact_info(original_width, original_height)
    return BlackPadInfo(
        type=preprocess_type,
        original_width=int(original_width),
        original_height=int(original_height),
        crop_bbox_xyxy=tuple(int(value) for value in crop_bbox),
        crop_was_applied=bool(crop_was_applied),
        crop_fallback=crop_fallback,
        cropped_width=int(cropped_width),
        cropped_height=int(cropped_height),
        cropped_source=str(cropped_source.resolve()) if cropped_source is not None else None,
        padding_px=int(padding_px),
        padding_fraction=float(padding_fraction),
        padding_min_px=int(padding_min_px),
        padded_width=int(model_input_width),
        padded_height=int(model_input_height),
        model_input_width=int(model_input_width),
        model_input_height=int(model_input_height),
        no_black_width=int(cropped_width),
        no_black_height=int(cropped_height),
        no_black_bbox_in_model_input=(
            int(padding_px),
            int(padding_px),
            int(padding_px + cropped_width),
            int(padding_px + cropped_height),
        ),
        black_border_luma_floor=float(black_border_luma_floor),
        edge_artifact_crop_bbox_xyxy=edge_artifact_info.bbox_xyxy,
        edge_artifact_crop_was_applied=bool(edge_artifact_info.was_applied),
        edge_artifact_crop_reason=edge_artifact_info.reason,
        edge_artifact_dark_fraction=edge_artifact_info.dark_fraction,
        edge_artifact_border_dark_fraction=edge_artifact_info.border_dark_fraction,
        edge_artifact_mask_fraction=edge_artifact_info.mask_fraction,
    )


def crop_black_border_image_file(
    source: Path,
    destination: Path,
    *,
    black_border_luma_floor: float = 8.0,
    jpeg_quality: int = 95,
) -> BlackPadInfo:
    with Image.open(source) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        crop_bbox, crop_fallback, edge_info = _combined_black_border_bbox(
            image,
            luma_floor=black_border_luma_floor,
        )
        cropped = image.crop(crop_bbox)
        cropped_width, cropped_height = cropped.size

    _save_rgb_image(cropped, destination, jpeg_quality=jpeg_quality)
    return _preprocess_info(
        preprocess_type="crop_black_border",
        original_width=original_width,
        original_height=original_height,
        crop_bbox=crop_bbox,
        crop_fallback=crop_fallback,
        cropped_width=cropped_width,
        cropped_height=cropped_height,
        cropped_source=destination,
        padding_px=0,
        padding_fraction=0.0,
        padding_min_px=0,
        model_input_width=cropped_width,
        model_input_height=cropped_height,
        black_border_luma_floor=black_border_luma_floor,
        edge_artifact_info=edge_info,
    )


def blackpad_image_file(
    source: Path,
    destination: Path,
    *,
    fraction: float = 0.30,
    min_padding: int = 80,
    black_border_luma_floor: float = 8.0,
    cropped_destination: Path | None = None,
    jpeg_quality: int = 95,
) -> BlackPadInfo:
    with Image.open(source) as image:
        padded, cropped, info = blackpad_image(
            image,
            fraction=fraction,
            min_padding=min_padding,
            black_border_luma_floor=black_border_luma_floor,
            cropped_source=cropped_destination,
        )

    if cropped_destination is not None:
        _save_rgb_image(cropped, cropped_destination, jpeg_quality=jpeg_quality)
    _save_rgb_image(padded, destination, jpeg_quality=jpeg_quality)
    return info


def blackpad_image(
    image: Image.Image,
    *,
    fraction: float = 0.30,
    min_padding: int = 80,
    black_border_luma_floor: float = 8.0,
    cropped_source: Path | None = None,
) -> tuple[Image.Image, Image.Image, BlackPadInfo]:
    image = image.convert("RGB")
    original_width, original_height = image.size
    crop_bbox, crop_fallback, edge_info = _combined_black_border_bbox(
        image,
        luma_floor=black_border_luma_floor,
    )
    cropped = image.crop(crop_bbox)
    cropped_width, cropped_height = cropped.size
    padding = blackpad_padding(cropped_width, cropped_height, fraction=fraction, min_padding=min_padding)
    padded = ImageOps.expand(cropped, border=padding, fill=(0, 0, 0))
    info = _preprocess_info(
        preprocess_type="crop_black_border_then_blackpad",
        original_width=original_width,
        original_height=original_height,
        crop_bbox=crop_bbox,
        crop_fallback=crop_fallback,
        cropped_width=cropped_width,
        cropped_height=cropped_height,
        cropped_source=cropped_source,
        padding_px=padding,
        padding_fraction=float(fraction),
        padding_min_px=int(min_padding),
        model_input_width=padded.width,
        model_input_height=padded.height,
        black_border_luma_floor=black_border_luma_floor,
        edge_artifact_info=edge_info,
    )
    return padded, cropped, info
