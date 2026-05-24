from __future__ import annotations

from PIL import Image

from tools.predict_roi import prepare_inference_images
from yoloposevf.preprocess import (
    black_border_crop_bbox,
    blackpad_image_file,
    blackpad_padding,
    crop_black_border_then_blackpad_image_file,
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
    assert info.padded_width == 160
    assert info.padded_height == 110
    with Image.open(destination) as image:
        assert image.size == (160, 110)
        assert image.getpixel((0, 0)) == (0, 0, 0)


def test_black_border_crop_bbox_returns_full_image_when_no_border() -> None:
    image = Image.new("RGB", (100, 50), color=(200, 100, 50))

    assert black_border_crop_bbox(image, black_luma_floor=8.0) == (0, 0, 100, 50)


def test_crop_black_border_then_blackpad_removes_existing_border_and_adds_uniform_padding(tmp_path) -> None:
    source = tmp_path / "bordered.png"
    destination = tmp_path / "model_input.png"
    cropped_destination = tmp_path / "cropped.png"
    image = Image.new("RGB", (30, 20), color=(0, 0, 0))
    foreground = Image.new("RGB", (10, 8), color=(180, 90, 40))
    image.paste(foreground, (4, 3))
    image.save(source)

    info = crop_black_border_then_blackpad_image_file(
        source,
        destination,
        cropped_destination=cropped_destination,
        fraction=0.30,
        min_padding=6,
        black_luma_floor=8.0,
    )

    assert info.type == "crop_black_border_then_blackpad"
    assert info.original_width == 30
    assert info.original_height == 20
    assert info.crop_bbox_xyxy == (4, 3, 14, 11)
    assert info.crop_was_applied is True
    assert info.cropped_width == 10
    assert info.cropped_height == 8
    assert info.padding_px == 6
    assert info.padding_fraction == 0.30
    assert info.padding_min_px == 6
    assert info.model_input_width == 22
    assert info.model_input_height == 20
    assert info.no_black_width == 10
    assert info.no_black_height == 8
    assert info.no_black_bbox_in_model_input == (6, 6, 16, 14)
    assert info.cropped_source == str(cropped_destination.resolve())
    assert info.to_dict()["no_black_bbox_in_model_input"] == [6, 6, 16, 14]
    with Image.open(cropped_destination) as cropped:
        assert cropped.size == (10, 8)
        assert cropped.getpixel((0, 0)) == (180, 90, 40)
    with Image.open(destination) as padded:
        assert padded.size == (22, 20)
        assert padded.getpixel((0, 0)) == (0, 0, 0)
        assert padded.getpixel((6, 6)) == (180, 90, 40)


def test_crop_black_border_then_blackpad_falls_back_for_all_black_image(tmp_path) -> None:
    source = tmp_path / "all_black.png"
    destination = tmp_path / "all_black_model_input.png"
    Image.new("RGB", (12, 10), color=(0, 0, 0)).save(source)

    info = crop_black_border_then_blackpad_image_file(source, destination, fraction=0.30, min_padding=5)

    assert info.crop_bbox_xyxy == (0, 0, 12, 10)
    assert info.crop_was_applied is False
    assert info.crop_fallback == "full_image_no_foreground"
    assert info.cropped_width == 12
    assert info.cropped_height == 10
    assert info.padding_px == 5
    assert info.model_input_width == 22
    assert info.model_input_height == 20


def test_crop_black_border_then_blackpad_always_adds_uniform_blackpad(tmp_path) -> None:
    source = tmp_path / "bordered.png"
    destination = tmp_path / "model_input.png"
    cropped_destination = tmp_path / "cropped_only.png"
    image = Image.new("RGB", (20, 14), color=(0, 0, 0))
    image.paste(Image.new("RGB", (8, 6), color=(160, 110, 70)), (3, 4))
    image.save(source)

    info = crop_black_border_then_blackpad_image_file(
        source,
        destination,
        cropped_destination=cropped_destination,
        fraction=0.30,
        min_padding=6,
    )

    assert info.type == "crop_black_border_then_blackpad"
    assert info.crop_bbox_xyxy == (3, 4, 11, 10)
    assert info.padding_px == 6
    assert info.cropped_width == 8
    assert info.cropped_height == 6
    assert info.model_input_width == 20
    assert info.model_input_height == 18
    assert info.no_black_bbox_in_model_input == (6, 6, 14, 12)
    assert info.cropped_source == str(cropped_destination.resolve())
    with Image.open(cropped_destination) as cropped:
        assert cropped.size == (8, 6)
        assert cropped.getpixel((0, 0)) == (160, 110, 70)
    with Image.open(destination) as model_input:
        assert model_input.size == (20, 18)
        assert model_input.getpixel((0, 0)) == (0, 0, 0)
        assert model_input.getpixel((6, 6)) == (160, 110, 70)


def test_prepare_inference_images_uses_crop_then_blackpad_metadata(tmp_path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source = source_dir / "sample.png"
    image = Image.new("RGB", (20, 14), color=(0, 0, 0))
    image.paste(Image.new("RGB", (8, 6), color=(160, 110, 70)), (3, 4))
    image.save(source)

    images, metadata = prepare_inference_images(
        source_dir,
        out_path=tmp_path / "predictions.jsonl",
        blackpad_input_dir=None,
        cropped_input_dir=None,
        blackpad_fraction=0.0,
        blackpad_min_padding=3,
        black_border_luma_floor=8.0,
    )

    assert len(images) == 1
    model_input = images[0]
    record = metadata[str(model_input.resolve())]
    assert model_input.parent.name == "predictions_blackpad_inputs"
    assert record["source"] == str(model_input.resolve())
    assert record["original_source"] == str(source.resolve())
    assert record["cropped_source"].endswith("_cropped_inputs/sample.png")
    assert record["preprocess"]["type"] == "crop_black_border_then_blackpad"
    assert record["preprocess"]["crop_bbox_xyxy"] == [3, 4, 11, 10]
    assert record["preprocess"]["padding_px"] == 3
    assert record["preprocess"]["model_input_width"] == 14
    assert record["preprocess"]["model_input_height"] == 12
    assert record["preprocess"]["no_black_bbox_in_model_input"] == [3, 3, 11, 9]
    with Image.open(model_input) as model_image:
        assert model_image.size == (14, 12)
        assert model_image.getpixel((0, 0)) == (0, 0, 0)
        assert model_image.getpixel((3, 3)) == (160, 110, 70)


def test_prepare_inference_images_always_returns_blackpad_input_for_single_file(tmp_path) -> None:
    source = tmp_path / "sample.png"
    image = Image.new("RGB", (20, 14), color=(0, 0, 0))
    image.paste(Image.new("RGB", (8, 6), color=(160, 110, 70)), (3, 4))
    image.save(source)

    images, metadata = prepare_inference_images(
        source,
        out_path=tmp_path / "predictions.jsonl",
        blackpad_input_dir=None,
        cropped_input_dir=None,
        blackpad_fraction=0.30,
        blackpad_min_padding=5,
        black_border_luma_floor=8.0,
    )

    assert len(images) == 1
    model_input = images[0]
    record = metadata[str(model_input.resolve())]
    assert model_input.parent.name == "predictions_blackpad_inputs"
    assert model_input.name == "sample.png"
    assert record["source"] == str(model_input.resolve())
    assert record["original_source"] == str(source.resolve())
    assert record["cropped_source"].endswith("_cropped_inputs/sample.png")
    assert record["cropped_source"] != str(model_input.resolve())
    assert record["preprocess"]["type"] == "crop_black_border_then_blackpad"
    assert record["preprocess"]["padding_px"] == 5
    assert record["preprocess"]["model_input_width"] == 18
    assert record["preprocess"]["model_input_height"] == 16
    assert record["preprocess"]["no_black_bbox_in_model_input"] == [5, 5, 13, 11]
    with Image.open(model_input) as model_image:
        assert model_image.size == (18, 16)
        assert model_image.getpixel((0, 0)) == (0, 0, 0)
        assert model_image.getpixel((5, 5)) == (160, 110, 70)
    with Image.open(record["cropped_source"]) as cropped:
        assert cropped.size == (8, 6)
