from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

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

    def to_dict(self) -> dict[str, int | float | str | bool | None | list[int]]:
        values = asdict(self)
        values["crop_bbox_xyxy"] = list(self.crop_bbox_xyxy)
        values["no_black_bbox_in_model_input"] = list(self.no_black_bbox_in_model_input)
        return values


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
    bbox, _ = _foreground_bbox_and_fallback(image, luma_floor=luma_floor)
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
) -> BlackPadInfo:
    crop_was_applied = crop_bbox != (0, 0, int(original_width), int(original_height))
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
        crop_bbox, crop_fallback = _foreground_bbox_and_fallback(image, luma_floor=black_border_luma_floor)
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
        image = image.convert("RGB")
        original_width, original_height = image.size
        crop_bbox, crop_fallback = _foreground_bbox_and_fallback(image, luma_floor=black_border_luma_floor)
        cropped = image.crop(crop_bbox)
        cropped_width, cropped_height = cropped.size
        padding = blackpad_padding(cropped_width, cropped_height, fraction=fraction, min_padding=min_padding)
        padded = ImageOps.expand(cropped, border=padding, fill=(0, 0, 0))

    if cropped_destination is not None:
        _save_rgb_image(cropped, cropped_destination, jpeg_quality=jpeg_quality)
    _save_rgb_image(padded, destination, jpeg_quality=jpeg_quality)
    return _preprocess_info(
        preprocess_type="crop_black_border_then_blackpad",
        original_width=original_width,
        original_height=original_height,
        crop_bbox=crop_bbox,
        crop_fallback=crop_fallback,
        cropped_width=cropped_width,
        cropped_height=cropped_height,
        cropped_source=cropped_destination,
        padding_px=padding,
        padding_fraction=float(fraction),
        padding_min_px=int(min_padding),
        model_input_width=padded.width,
        model_input_height=padded.height,
        black_border_luma_floor=black_border_luma_floor,
    )
