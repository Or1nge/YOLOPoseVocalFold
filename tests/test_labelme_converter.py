import json
from pathlib import Path

from PIL import Image

from tools.convert_labelme_to_yolo_pose import convert, parse_args


def test_labelme_conversion_creates_yolo_label(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    labelme_dir = tmp_path / "labelme"
    out_dir = tmp_path / "yolo"
    image_dir.mkdir()
    labelme_dir.mkdir()
    Image.new("RGB", (100, 80)).save(image_dir / "sample.png")
    payload = {
        "imagePath": "sample.png",
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [
            {"label": "vocal_fold_roi", "shape_type": "rectangle", "points": [[20, 20], [70, 60]]},
            {"label": "kp1", "shape_type": "point", "points": [[25, 25]]},
            {"label": "kp2", "shape_type": "point", "points": [[65, 25]]},
            {"label": "kp3", "shape_type": "point", "points": [[25, 55]]},
            {"label": "kp4", "shape_type": "point", "points": [[65, 55]]},
        ],
    }
    (labelme_dir / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert",
            "--labelme-dir",
            str(labelme_dir),
            "--image-dir",
            str(image_dir),
            "--out-dir",
            str(out_dir),
            "--copy-mode",
            "copy",
            "--val-ratio",
            "0",
            "--test-ratio",
            "0",
        ],
    )
    args = parse_args()
    manifest = convert(args)
    assert manifest["counts"]["train"] == 1
    label = out_dir / "labels" / "train" / "sample.txt"
    assert label.exists()
    assert len(label.read_text(encoding="utf-8").split()) == 17

