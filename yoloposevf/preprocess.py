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


def blackpad_padding(width: int, height: int, fraction: float = 0.30, min_padding: int = 80) -> int:
    return max(int(min_padding), int(round(max(width, height) * float(fraction))))


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

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = int(jpeg_quality)
    padded.save(destination, **save_kwargs)
    return BlackPadInfo(
        type="blackpad",
        padding_px=padding,
        original_width=width,
        original_height=height,
        padded_width=padded.width,
        padded_height=padded.height,
    )
