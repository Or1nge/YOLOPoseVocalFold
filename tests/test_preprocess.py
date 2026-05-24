from __future__ import annotations

from PIL import Image

from tools.predict_roi import prepare_inference_images
from yoloposevf.preprocess import (
    blackpad_image_file,
    blackpad_padding,
    crop_black_border_image_file,
    crop_existing_black_borders,
)


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
    assert info.crop_bbox_xyxy == (0, 0, 100, 50)
    assert info.crop_was_applied is False
    assert info.cropped_width == 100
    assert info.cropped_height == 50
    assert info.padded_width == 160
    assert info.padded_height == 110
    assert info.model_input_width == 160
    assert info.model_input_height == 110
    assert info.no_black_bbox_in_model_input == (30, 30, 130, 80)
    with Image.open(destination) as image:
        assert image.size == (160, 110)
        assert image.getpixel((0, 0)) == (0, 0, 0)


def test_blackpad_image_file_crops_existing_black_border_before_padding(tmp_path) -> None:
    source = tmp_path / "black_border.png"
    destination = tmp_path / "padded.png"
    cropped_destination = tmp_path / "cropped.png"
    image = Image.new("RGB", (20, 12), color=(0, 0, 0))
    for y in range(3, 9):
        for x in range(4, 16):
            image.putpixel((x, y), (160, 120, 80))
    image.save(source)

    info = blackpad_image_file(
        source,
        destination,
        fraction=0.50,
        min_padding=2,
        cropped_destination=cropped_destination,
    )

    assert info.type == "crop_black_border_then_blackpad"
    assert info.original_width == 20
    assert info.original_height == 12
    assert info.to_dict()["crop_bbox_xyxy"] == [4, 3, 16, 9]
    assert info.to_dict()["crop_was_applied"] is True
    assert info.to_dict()["cropped_source"] == str(cropped_destination.resolve())
    assert info.cropped_width == 12
    assert info.cropped_height == 6
    assert info.padding_px == 6
    assert info.padding_fraction == 0.50
    assert info.padding_min_px == 2
    assert info.model_input_width == 24
    assert info.model_input_height == 18
    assert info.no_black_width == 12
    assert info.no_black_height == 6
    assert info.to_dict()["no_black_bbox_in_model_input"] == [6, 6, 18, 12]
    with Image.open(cropped_destination) as cropped:
        assert cropped.size == (12, 6)
    with Image.open(destination) as padded:
        assert padded.size == (24, 18)
        assert padded.getpixel((6, 6)) == (160, 120, 80)


def test_crop_existing_black_borders_keeps_all_black_images_full_size() -> None:
    image = Image.new("RGB", (10, 8), color=(0, 0, 0))
    cropped, bbox = crop_existing_black_borders(image)

    assert bbox == (0, 0, 10, 8)
    assert cropped.size == (10, 8)


def test_crop_black_border_image_file_skips_extra_padding(tmp_path) -> None:
    source = tmp_path / "black_border.png"
    destination = tmp_path / "cropped_model_input.png"
    image = Image.new("RGB", (20, 12), color=(0, 0, 0))
    for y in range(3, 9):
        for x in range(4, 16):
            image.putpixel((x, y), (160, 120, 80))
    image.save(source)

    info = crop_black_border_image_file(source, destination)

    assert info.type == "crop_black_border"
    assert info.padding_px == 0
    assert info.padding_fraction == 0.0
    assert info.padding_min_px == 0
    assert info.model_input_width == 12
    assert info.model_input_height == 6
    assert info.no_black_bbox_in_model_input == (0, 0, 12, 6)
    with Image.open(destination) as image:
        assert image.size == (12, 6)


def test_prediction_preprocess_always_crops_then_blackpads(tmp_path) -> None:
    source_dir = tmp_path / "images"
    source_dir.mkdir()
    source = source_dir / "sample.png"
    image = Image.new("RGB", (20, 12), color=(0, 0, 0))
    for y in range(3, 9):
        for x in range(4, 16):
            image.putpixel((x, y), (160, 120, 80))
    image.save(source)

    images, metadata = prepare_inference_images(
        source_dir,
        manifest=None,
        out_path=tmp_path / "predictions.jsonl",
        blackpad_input_dir=None,
        cropped_input_dir=None,
        blackpad_fraction=0.50,
        blackpad_min_padding=2,
        black_border_luma_floor=8.0,
    )

    assert len(images) == 1
    record = metadata[str(images[0].resolve())]
    assert record["source"] != record["cropped_source"]
    assert record["dinov3_source"] == record["cropped_source"]
    assert record["original_source"] == str(source.resolve())
    assert record["preprocess"]["type"] == "crop_black_border_then_blackpad"
    assert record["preprocess"]["crop_bbox_xyxy"] == [4, 3, 16, 9]
    assert record["preprocess"]["padding_px"] == 6
    with Image.open(images[0]) as image:
        assert image.size == (24, 18)
