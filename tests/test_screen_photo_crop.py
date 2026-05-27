from __future__ import annotations

import numpy as np
from PIL import Image

from yoloposevf.screen_photo_crop import (
    classify_screen_photo,
    crop_screen_photo_window,
    screen_artifact_signals,
)


def _rgb_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def test_screen_artifact_signals_clean_tissue_image_has_low_scores() -> None:
    img = _rgb_image(200, 150, color=(180, 60, 40))
    signals = screen_artifact_signals(img)
    assert signals["stripe_col"] <= 0.24
    assert signals["stripe_row"] <= 0.24
    assert signals["blue_col"] <= 0.06
    assert signals["blue_row"] <= 0.06


def test_classify_screen_photo_clean_image_not_triggered() -> None:
    img = _rgb_image(300, 200, color=(170, 80, 50))
    triggered, signals = classify_screen_photo(img)
    assert triggered is False
    assert signals["reason"] == ["none"]


def test_classify_screen_photo_blue_stripe_at_right_triggers() -> None:
    # Solid blue stripe along the rightmost 10 columns should produce a high
    # blue_col signal.
    img = _rgb_image(400, 300, color=(160, 70, 40))
    pixels = img.load()
    for y in range(300):
        for x in range(390, 400):
            pixels[x, y] = (30, 60, 180)
    triggered, signals = classify_screen_photo(img)
    assert triggered is True
    assert "blue_col" in signals["reason"]


def test_classify_screen_photo_blue_stripe_at_bottom_triggers() -> None:
    img = _rgb_image(300, 400, color=(160, 70, 40))
    pixels = img.load()
    for y in range(390, 400):
        for x in range(300):
            pixels[x, y] = (30, 60, 180)
    triggered, signals = classify_screen_photo(img)
    assert triggered is True
    assert "blue_row" in signals["reason"]


def test_crop_screen_photo_window_returns_valid_crop_on_synthetic() -> None:
    # Create a simple image: tissue-like centre with dark surround.
    img = Image.new("RGB", (500, 400), color=(10, 10, 10))
    pixels = img.load()
    for y in range(80, 320):
        for x in range(60, 440):
            pixels[x, y] = (180, 50, 30)
    cropped, box = crop_screen_photo_window(img)
    assert cropped.width > 0
    assert cropped.height > 0
    assert cropped.width <= img.width
    assert cropped.height <= img.height
    assert len(box) == 4
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= img.width and box[3] <= img.height


def test_crop_screen_photo_window_preserves_rgb_mode() -> None:
    img = _rgb_image(200, 150, color=(150, 55, 35))
    cropped, _box = crop_screen_photo_window(img)
    assert cropped.mode == "RGB"


def test_signals_dict_has_expected_keys() -> None:
    img = _rgb_image(100, 100, color=(170, 80, 50))
    signals = screen_artifact_signals(img)
    for key in ("stripe_col", "stripe_row", "blue_col", "blue_row"):
        assert key in signals
        assert isinstance(signals[key], float)


def test_classify_screen_photo_reason_includes_none_when_clean() -> None:
    img = _rgb_image(100, 100, color=(170, 80, 50))
    _triggered, signals = classify_screen_photo(img)
    assert signals["reason"] == ["none"]


def test_classify_screen_photo_empty_reason_means_not_triggered() -> None:
    # reason list should be ["none"] when not triggered, never empty.
    img = _rgb_image(100, 100, color=(170, 80, 50))
    triggered, signals = classify_screen_photo(img)
    assert triggered is False
    assert len(signals["reason"]) >= 1
