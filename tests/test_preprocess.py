from __future__ import annotations

from PIL import Image

from yoloposevf.preprocess import blackpad_image_file, blackpad_padding


def test_blackpad_padding_uses_long_side_and_minimum() -> None:
    assert blackpad_padding(100, 200, fraction=0.30, min_padding=80) == 80
    assert blackpad_padding(1000, 500, fraction=0.30, min_padding=80) == 300


def test_blackpad_image_file_adds_equal_four_side_border(tmp_path) -> None:
    source = tmp_path / "sample.png"
    destination = tmp_path / "padded.png"
    Image.new("RGB", (100, 50), color=(200, 100, 50)).save(source)

    info = blackpad_image_file(source, destination, fraction=0.30, min_padding=20)

    assert info.padding_px == 30
    assert info.original_width == 100
    assert info.original_height == 50
    assert info.padded_width == 160
    assert info.padded_height == 110
    with Image.open(destination) as image:
        assert image.size == (160, 110)
        assert image.getpixel((0, 0)) == (0, 0, 0)
