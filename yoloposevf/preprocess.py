from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class BlackPadInfo:
    type: str
    padding_px: int
    original_width: int
    original_height: int
    padded_width: int
    padded_height: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CropBlackPadInfo:
    type: str
    black_luma_floor: float
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

    def to_dict(self) -> dict[str, bool | float | int | list[int] | str | None]:
        payload = asdict(self)
        payload["crop_bbox_xyxy"] = list(self.crop_bbox_xyxy)
        payload["no_black_bbox_in_model_input"] = list(self.no_black_bbox_in_model_input)
        return payload


def blackpad_padding(width: int, height: int, fraction: float = 0.30, min_padding: int = 80) -> int:
    return max(int(min_padding), int(round(max(width, height) * float(fraction))))


def black_border_crop_bbox(
    image: Image.Image,
    *,
    black_luma_floor: float = 8.0,
) -> tuple[int, int, int, int] | None:
    luma = image.convert("L")
    mask = luma.point(lambda value: 255 if value > float(black_luma_floor) else 0)
    return mask.getbbox()


def save_rgb_image(image: Image.Image, destination: Path, *, jpeg_quality: int = 95) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = int(jpeg_quality)
    image.save(destination, **save_kwargs)


def blackpad_image_file(
    source: Path,
    destination: Path,
    *,
    fraction: float = 0.30,
    min_padding: int = 80,
    jpeg_quality: int = 95,
) -> BlackPadInfo:
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        padding = blackpad_padding(width, height, fraction=fraction, min_padding=min_padding)
        padded = ImageOps.expand(image, border=padding, fill=(0, 0, 0))

    save_rgb_image(padded, destination, jpeg_quality=jpeg_quality)
    return BlackPadInfo(
        type="blackpad",
        padding_px=padding,
        original_width=width,
        original_height=height,
        padded_width=padded.width,
        padded_height=padded.height,
    )


def crop_black_border_then_blackpad_image_file(
    source: Path,
    destination: Path,
    *,
    cropped_destination: Path | None = None,
    fraction: float = 0.30,
    min_padding: int = 80,
    black_luma_floor: float = 8.0,
    jpeg_quality: int = 95,
) -> CropBlackPadInfo:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if destination_resolved == source_resolved:
        raise ValueError("preprocessed inference input must not overwrite the source image")
    if cropped_destination is not None and cropped_destination.resolve() == source_resolved:
        raise ValueError("cropped inference input must not overwrite the source image")

    with Image.open(source) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        bbox = black_border_crop_bbox(image, black_luma_floor=black_luma_floor)
        crop_fallback = None
        if bbox is None:
            bbox = (0, 0, original_width, original_height)
            crop_fallback = "full_image_no_foreground"
        cropped = image.crop(bbox)
        cropped_width, cropped_height = cropped.size
        padding = blackpad_padding(cropped_width, cropped_height, fraction=fraction, min_padding=min_padding)
        padded = ImageOps.expand(cropped, border=padding, fill=(0, 0, 0))
        crop_bbox = tuple(int(value) for value in bbox)

    cropped_source: str | None = None
    if cropped_destination is not None:
        cropped_destination_resolved = cropped_destination.resolve()
        if cropped_destination_resolved != destination_resolved:
            save_rgb_image(cropped, cropped_destination, jpeg_quality=jpeg_quality)
        cropped_source = str(cropped_destination_resolved)
    save_rgb_image(padded, destination, jpeg_quality=jpeg_quality)

    return CropBlackPadInfo(
        type="crop_black_border_then_blackpad",
        black_luma_floor=float(black_luma_floor),
        original_width=original_width,
        original_height=original_height,
        crop_bbox_xyxy=crop_bbox,
        crop_was_applied=crop_bbox != (0, 0, original_width, original_height),
        crop_fallback=crop_fallback,
        cropped_width=cropped_width,
        cropped_height=cropped_height,
        cropped_source=cropped_source,
        padding_px=padding,
        padding_fraction=float(fraction),
        padding_min_px=int(min_padding),
        padded_width=padded.width,
        padded_height=padded.height,
        model_input_width=padded.width,
        model_input_height=padded.height,
        no_black_width=cropped_width,
        no_black_height=cropped_height,
        no_black_bbox_in_model_input=(
            int(padding),
            int(padding),
            int(padding + cropped_width),
            int(padding + cropped_height),
        ),
    )
