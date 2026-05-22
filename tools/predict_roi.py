#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.postprocess import PosePrediction, PostprocessConfig, decide_action, fuse_prediction, load_postprocess_config
from yoloposevf.preprocess import IMAGE_EXTENSIONS, blackpad_image_file


ALGORITHM_VERSION = "V1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-Pose inference and anatomy-constrained ROI postprocessing.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path, default=Path("configs/postprocess.yaml"))
    parser.add_argument("--out", type=Path, default=Path("Results/predictions/predictions.jsonl"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str)
    parser.add_argument("--imgsz", type=int, help="Inference image size. Use the training size for best keypoint precision.")
    parser.add_argument("--tta", action="store_true", help="Use rotation/scale test-time augmentation without flips.")
    parser.add_argument("--tta-degrees", default="-6,0,6", help="Comma-separated rotation degrees for --tta.")
    parser.add_argument("--tta-scales", default="0.95,1.0,1.05", help="Comma-separated scale factors for --tta.")
    parser.add_argument("--no-blackpad", action="store_true", help="Disable the V1.1 black-border pre-enhancement.")
    parser.add_argument("--blackpad-fraction", type=float, default=0.30)
    parser.add_argument("--blackpad-min-padding", type=int, default=80)
    parser.add_argument("--blackpad-input-dir", type=Path, help="Directory for generated black-padded inference inputs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def default_blackpad_input_dir(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_blackpad_inputs"


def relative_image_path(image_path: Path, source: Path) -> Path:
    if source.is_file():
        return Path(image_path.name)
    return image_path.relative_to(source)


def prepare_inference_images(
    source: Path,
    *,
    out_path: Path,
    blackpad_enabled: bool,
    blackpad_input_dir: Path | None,
    blackpad_fraction: float,
    blackpad_min_padding: int,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    images = iter_images(source)
    metadata: dict[str, dict[str, Any]] = {}
    if not blackpad_enabled:
        for image_path in images:
            metadata[str(image_path.resolve())] = {
                "source": str(image_path),
                "original_source": str(image_path),
                "preprocess": {"type": "none"},
            }
        return images, metadata

    pad_root = blackpad_input_dir or default_blackpad_input_dir(out_path)
    padded_images: list[Path] = []
    for image_path in images:
        rel = relative_image_path(image_path, source)
        destination = pad_root / rel
        info = blackpad_image_file(
            image_path,
            destination,
            fraction=blackpad_fraction,
            min_padding=blackpad_min_padding,
        )
        padded_images.append(destination)
        metadata[str(destination.resolve())] = {
            "source": str(destination),
            "original_source": str(image_path),
            "preprocess": info.to_dict(),
        }
    return padded_images, metadata


def result_to_prediction(
    result: Any,
    *,
    source: str | None = None,
    inverse_affine: Sequence[Sequence[float]] | None = None,
) -> PosePrediction | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    if result.keypoints is None or result.keypoints.data is None:
        return None
    keypoints = result.keypoints.data.cpu().numpy()[best]
    height, width = result.orig_shape[:2]
    bbox = tuple(float(v) for v in boxes[best])
    keypoint_rows = [tuple(float(v) for v in kp[:3]) for kp in keypoints]
    if inverse_affine is not None:
        bbox = transform_bbox_xyxy(bbox, inverse_affine, width=width, height=height)
        keypoint_rows = [
            (*transform_point((kp[0], kp[1]), inverse_affine), float(kp[2]))
            for kp in keypoint_rows
        ]
    return PosePrediction(
        bbox=bbox,
        bbox_conf=float(confs[best]),
        keypoints=tuple(keypoint_rows),
        image_size=ImageSize(width=int(width), height=int(height)),
        source=source or str(result.path),
    )


def transform_point(point: Sequence[float], matrix: Sequence[Sequence[float]]) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    return (
        float(matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]),
        float(matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]),
    )


def transform_bbox_xyxy(
    bbox: Sequence[float],
    matrix: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    corners = [
        transform_point((x1, y1), matrix),
        transform_point((x2, y1), matrix),
        transform_point((x2, y2), matrix),
        transform_point((x1, y2), matrix),
    ]
    xs = [min(max(point[0], 0.0), float(width)) for point in corners]
    ys = [min(max(point[1], 0.0), float(height)) for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(weight, 0.0) for _, weight in pairs)
    if total <= 0:
        return float(pairs[len(pairs) // 2][0])
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += max(weight, 0.0)
        if cumulative >= total / 2.0:
            return float(value)
    return float(pairs[-1][0])


def prediction_weight(prediction: PosePrediction) -> float:
    keypoint_conf = sum(kp[2] for kp in prediction.keypoints) / max(len(prediction.keypoints), 1)
    return max(float(prediction.bbox_conf), 0.0) * max(float(keypoint_conf), 0.0)


def lower_bound_factor(value: float | None, low: float, good: float) -> float:
    low = max(float(low), 0.0)
    good = max(float(good), 0.0)
    if value is None or good <= 0.0:
        return 1.0
    if good <= low:
        return 1.0 if value >= low else 0.0
    if value >= good:
        return 1.0
    if value <= low:
        return 0.0
    return float((value - low) / (good - low))


def polygon_dark_fraction(image_path: Path, polygon: Sequence[Sequence[float]], luma_threshold: float) -> float | None:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    if not image_path.exists() or not polygon:
        return None
    image = Image.open(image_path).convert("L")
    mask = Image.new("1", image.size, 0)
    points = [(float(point[0]), float(point[1])) for point in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool)
    if not mask_array.any():
        return 0.0
    return float((image_array[mask_array] < float(luma_threshold)).mean())


def apply_image_quality_gates(record: dict[str, Any], image_path: Path, cfg: PostprocessConfig) -> dict[str, Any]:
    dark_gate_enabled = cfg.roi_dark_luma_threshold > 0.0 and cfg.good_roi_dark_fraction > 0.0
    if not dark_gate_enabled:
        return record

    polygon = record.get("final_box_polygon") or record.get("roi_polygon")
    dark_fraction = polygon_dark_fraction(image_path, polygon, cfg.roi_dark_luma_threshold) if polygon else None
    dark_factor = lower_bound_factor(dark_fraction, cfg.min_roi_dark_fraction, cfg.good_roi_dark_fraction)
    record["roi_dark_fraction"] = dark_fraction
    record["roi_dark_factor"] = dark_factor
    if dark_fraction is not None and dark_fraction < cfg.good_roi_dark_fraction:
        flags = record.setdefault("flags", [])
        flags.append("roi_too_bright" if dark_fraction <= cfg.min_roi_dark_fraction else "low_roi_dark_fraction")

    record["pre_image_gate_confidence"] = float(record.get("final_confidence", 0.0))
    record["final_confidence"] = max(0.0, min(1.0, record["pre_image_gate_confidence"] * dark_factor))
    record["action"] = decide_action(record["final_confidence"], cfg)
    if record["action"] == "reject_or_relabel":
        record["usable_bbox"] = None
        record["usable_box_polygon"] = None
    else:
        record["usable_bbox"] = record.get("final_bbox")
        record["usable_box_polygon"] = record.get("final_box_polygon")
    return record


def aggregate_predictions(predictions: Sequence[PosePrediction], source: Path) -> PosePrediction | None:
    if not predictions:
        return None
    weights = [prediction_weight(prediction) for prediction in predictions]
    image_size = predictions[0].image_size
    bbox = tuple(
        weighted_median([prediction.bbox[index] for prediction in predictions], weights)
        for index in range(4)
    )
    keypoints = []
    for keypoint_index in range(len(predictions[0].keypoints)):
        x = weighted_median([prediction.keypoints[keypoint_index][0] for prediction in predictions], weights)
        y = weighted_median([prediction.keypoints[keypoint_index][1] for prediction in predictions], weights)
        conf = weighted_median([prediction.keypoints[keypoint_index][2] for prediction in predictions], weights)
        keypoints.append((x, y, conf))
    return PosePrediction(
        bbox=bbox,
        bbox_conf=max(prediction.bbox_conf for prediction in predictions),
        keypoints=tuple(keypoints),
        image_size=image_size,
        source=str(source),
    )


def predict_with_tta(
    model: Any,
    image_path: Path,
    *,
    conf: float,
    device: str | None,
    imgsz: int | None,
    degrees: Sequence[float],
    scales: Sequence[float],
) -> tuple[PosePrediction | None, int]:
    import cv2
    import numpy as np
    from PIL import Image

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    predictions: list[PosePrediction] = []
    for degree in degrees:
        for scale in scales:
            matrix = cv2.getRotationMatrix2D(center, float(degree), float(scale))
            inverse = cv2.invertAffineTransform(matrix).tolist()
            transformed = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            predict_kwargs: dict[str, Any] = {"source": transformed, "conf": conf, "verbose": False}
            if device is not None:
                predict_kwargs["device"] = device
            if imgsz is not None:
                predict_kwargs["imgsz"] = imgsz
            results = model.predict(**predict_kwargs)
            for result in results:
                prediction = result_to_prediction(result, source=str(image_path), inverse_affine=inverse)
                if prediction is not None:
                    predictions.append(prediction)
    return aggregate_predictions(predictions, image_path), len(predictions)


def metadata_for_source(metadata: dict[str, dict[str, Any]], source: str | Path) -> dict[str, Any]:
    path = Path(str(source))
    return metadata.get(str(path.resolve()), {"source": str(source), "original_source": str(source), "preprocess": {"type": "none"}})


def attach_prediction_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    record["algorithm_version"] = ALGORITHM_VERSION
    record["source"] = metadata["source"]
    record["original_source"] = metadata["original_source"]
    record["preprocess"] = metadata["preprocess"]
    return record


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install project requirements first.") from exc

    cfg = load_postprocess_config(load_yaml(args.postprocess_config))
    model = YOLO(str(args.weights))
    images, source_metadata = prepare_inference_images(
        args.source,
        out_path=args.out,
        blackpad_enabled=not args.no_blackpad,
        blackpad_input_dir=args.blackpad_input_dir,
        blackpad_fraction=args.blackpad_fraction,
        blackpad_min_padding=args.blackpad_min_padding,
    )
    predict_source: str | list[str]
    if args.source.is_file():
        predict_source = str(images[0]) if images else str(args.source)
    elif not args.no_blackpad:
        predict_source = str((args.blackpad_input_dir or default_blackpad_input_dir(args.out)))
    else:
        predict_source = str(args.source)
    predict_kwargs: dict[str, Any] = {"source": predict_source, "conf": args.conf, "stream": True}
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        if args.tta:
            degrees = parse_float_list(args.tta_degrees)
            scales = parse_float_list(args.tta_scales)
            result_iter = []
            for image_path in images:
                prediction, vote_count = predict_with_tta(
                    model,
                    image_path,
                    conf=args.conf,
                    device=args.device,
                    imgsz=args.imgsz,
                    degrees=degrees,
                    scales=scales,
                )
                result_iter.append((str(image_path), prediction, vote_count))
        else:
            result_iter = [
                (
                    str(getattr(result, "path", "")),
                    result_to_prediction(result, source=str(getattr(result, "path", ""))),
                    1,
                )
                for result in model.predict(**predict_kwargs)
            ]

        for source, prediction, vote_count in result_iter:
            metadata = metadata_for_source(source_metadata, prediction.source if prediction is not None else source)
            if prediction is None:
                record = {
                    "action": "reject_or_relabel",
                    "final_confidence": 0.0,
                    "flags": ["no_valid_pose_prediction"],
                }
            else:
                record = fuse_prediction(prediction, cfg)
                record = apply_image_quality_gates(record, Path(metadata["source"]), cfg)
                record["keypoints"] = [list(kp) for kp in prediction.keypoints]
                record["image_size"] = {
                    "width": prediction.image_size.width,
                    "height": prediction.image_size.height,
                }
                if args.tta:
                    record["tta_votes"] = vote_count
                    record["tta_degrees"] = parse_float_list(args.tta_degrees)
                    record["tta_scales"] = parse_float_list(args.tta_scales)
            record = attach_prediction_metadata(record, metadata)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} prediction record(s) to {args.out}")


if __name__ == "__main__":
    main()
