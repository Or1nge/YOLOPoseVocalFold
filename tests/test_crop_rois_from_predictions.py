from __future__ import annotations

from pathlib import Path

from tools.crop_rois_from_predictions import original_source_for_record


def test_preprocessed_source_prefers_existing_cropped_source(tmp_path: Path) -> None:
    source = tmp_path / "blackpad.png"
    cropped = tmp_path / "cropped.png"
    original = tmp_path / "original.png"
    source.write_bytes(b"source")
    cropped.write_bytes(b"cropped")
    original.write_bytes(b"original")
    record = {
        "source": str(source),
        "cropped_source": str(cropped),
        "original_source": str(original),
    }

    selected = original_source_for_record(record, source, "preprocessed_source")

    assert selected == cropped


def test_preprocessed_source_falls_back_to_original_then_source(tmp_path: Path) -> None:
    source = tmp_path / "blackpad.png"
    missing_cropped = tmp_path / "missing_cropped.png"
    original = tmp_path / "original.png"
    source.write_bytes(b"source")
    original.write_bytes(b"original")
    record = {
        "source": str(source),
        "cropped_source": str(missing_cropped),
        "original_source": str(original),
    }

    assert original_source_for_record(record, source, "preprocessed_source") == original

    original.unlink()
    assert original_source_for_record(record, source, "preprocessed_source") == source
