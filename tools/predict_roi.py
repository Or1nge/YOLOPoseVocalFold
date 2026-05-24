#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.postprocess import (
    PosePrediction,
    PostprocessConfig,
    decide_action,
    fuse_prediction,
    load_postprocess_config,
)
from yoloposevf.preprocess import IMAGE_EXTENSIONS, crop_black_border_then_blackpad_image_file


ALGORITHM_VERSION = "V1.2"


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
    parser.add_argument("--blackpad-fraction", type=float, default=0.30)
    parser.add_argument("--blackpad-min-padding", type=int, default=80)
    parser.add_argument("--blackpad-input-dir", type=Path, help="Directory for generated black-padded inference inputs.")
    parser.add_argument("--cropped-input-dir", type=Path, help="Directory for generated no-existing-black-border inputs.")
    parser.add_argument("--black-border-luma-floor", type=float, default=8.0)
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


def default_cropped_input_dir(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_cropped_inputs"


def relative_image_path(image_path: Path, source: Path) -> Path:
    if source.is_file():
        return Path(image_path.name)
    return image_path.relative_to(source)


def prepare_inference_images(
    source: Path,
    *,
    out_path: Path,
    blackpad_input_dir: Path | None,
    cropped_input_dir: Path | None,
    blackpad_fraction: float,
    blackpad_min_padding: int,
    black_border_luma_floor: float,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    images = iter_images(source)
    metadata: dict[str, dict[str, Any]] = {}

    pad_root = (blackpad_input_dir or default_blackpad_input_dir(out_path)).resolve()
    crop_root = (cropped_input_dir or default_cropped_input_dir(out_path)).resolve()
    model_input_images: list[Path] = []
    for image_path in images:
        rel = relative_image_path(image_path, source)
        destination = pad_root / rel
        cropped_destination = crop_root / rel
        info = crop_black_border_then_blackpad_image_file(
            image_path,
            destination,
            cropped_destination=cropped_destination,
            fraction=blackpad_fraction,
            min_padding=blackpad_min_padding,
            black_luma_floor=black_border_luma_floor,
        )
        model_input_images.append(destination)
        metadata[str(destination.resolve())] = {
            "source": str(destination.resolve()),
            "original_source": str(image_path.resolve()),
            "cropped_source": str(cropped_destination.resolve()),
            "preprocess": info.to_dict(),
        }
    return model_input_images, metadata


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


def _relative_dark_threshold(
    image_array: "np.ndarray",
    *,
    mode: str,
    absolute_threshold: float,
    relative_ratio: float,
    foreground_luma_floor: float,
) -> tuple[float, float | None]:
    import numpy as np

    mode = mode.lower()
    if mode == "absolute":
        return float(absolute_threshold), None
    if mode not in {"relative", "relative_foreground_median"}:
        raise ValueError(f"Unsupported roi_dark_mode: {mode}")

    pixels = image_array.reshape(-1)
    foreground = pixels[pixels > float(foreground_luma_floor)]
    reference_pixels = foreground if foreground.size else pixels
    reference_luma = float(np.median(reference_pixels))
    threshold = max(0.0, min(255.0, reference_luma * max(float(relative_ratio), 0.0)))
    return threshold, reference_luma


def polygon_dark_fraction(
    image_path: Path,
    polygon: Sequence[Sequence[float]],
    luma_threshold: float,
    *,
    mode: str = "absolute",
    relative_luma_ratio: float = 0.80,
    foreground_luma_floor: float = 8.0,
    analysis_bbox: Sequence[float] | None = None,
) -> tuple[float | None, float | None, float | None]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return None, None, None

    if not image_path.exists() or not polygon:
        return None, None, None
    image = Image.open(image_path).convert("L")
    mask = Image.new("1", image.size, 0)
    points = [(float(point[0]), float(point[1])) for point in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool).copy()
    threshold_array = image_array
    if analysis_bbox is not None:
        image_height, image_width = mask_array.shape
        x1, y1, x2, y2 = [float(value) for value in analysis_bbox]
        left = max(0, min(image_width, int(math.floor(min(x1, x2)))))
        top = max(0, min(image_height, int(math.floor(min(y1, y2)))))
        right = max(0, min(image_width, int(math.ceil(max(x1, x2)))))
        bottom = max(0, min(image_height, int(math.ceil(max(y1, y2)))))
        if left >= right or top >= bottom:
            return 0.0, None, None
        valid_region = np.zeros_like(mask_array, dtype=bool)
        valid_region[top:bottom, left:right] = True
        mask_array &= valid_region
        threshold_array = image_array[top:bottom, left:right]
    if not mask_array.any():
        return 0.0, None, None
    effective_threshold, reference_luma = _relative_dark_threshold(
        threshold_array,
        mode=mode,
        absolute_threshold=luma_threshold,
        relative_ratio=relative_luma_ratio,
        foreground_luma_floor=foreground_luma_floor,
    )
    return float((image_array[mask_array] < effective_threshold).mean()), effective_threshold, reference_luma


def foreground_bbox(
    image_path: Path,
    *,
    foreground_luma_floor: float = 8.0,
) -> tuple[float, float, float, float] | None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    if not image_path.exists():
        return None
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image)
    mask = image_array > float(foreground_luma_floor)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def effective_area_from_metadata(
    metadata: dict[str, Any],
    cfg: PostprocessConfig,
) -> tuple[float | None, tuple[float, float, float, float] | None, str]:
    preprocess = metadata.get("preprocess") if isinstance(metadata.get("preprocess"), dict) else {}
    preprocess_type = str(preprocess.get("type", ""))
    no_black_bbox = preprocess.get("no_black_bbox_in_model_input")
    if isinstance(no_black_bbox, (list, tuple)) and len(no_black_bbox) == 4:
        x1, y1, x2, y2 = [float(value) for value in no_black_bbox]
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        if area > 0.0:
            return area, (x1, y1, x2, y2), "preprocess_no_black_bbox"

    if preprocess_type in {"crop_black_border_then_blackpad", "crop_black_border", "blackpad"}:
        padding = float(preprocess.get("padding_px", 0.0) or 0.0)
        if preprocess_type in {"crop_black_border_then_blackpad", "crop_black_border"}:
            width = float(preprocess.get("cropped_width", 0.0) or 0.0)
            height = float(preprocess.get("cropped_height", 0.0) or 0.0)
            mode = "preprocess_cropped_region"
        else:
            width = float(preprocess.get("original_width", 0.0) or 0.0)
            height = float(preprocess.get("original_height", 0.0) or 0.0)
            mode = "preprocess_original_region"
        if width > 0.0 and height > 0.0:
            bbox = (padding, padding, padding + width, padding + height)
            return width * height, bbox, mode

    foreground_floor = float(cfg.roi_dark_foreground_luma_floor)
    source = Path(str(metadata.get("source") or ""))
    bbox = foreground_bbox(source, foreground_luma_floor=foreground_floor)
    if bbox is None:
        return None, None, "full_image"
    x1, y1, x2, y2 = bbox
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    return (area if area > 0.0 else None), bbox, "input_foreground_bbox"


def prediction_with_effective_area(
    prediction: PosePrediction,
    metadata: dict[str, Any],
    cfg: PostprocessConfig,
) -> PosePrediction:
    area, bbox, mode = effective_area_from_metadata(metadata, cfg)
    if area is None:
        return prediction
    return replace(
        prediction,
        effective_image_area=area,
        effective_image_bbox=bbox,
        effective_image_area_mode=mode,
    )


def apply_image_quality_gates(record: dict[str, Any], image_path: Path, cfg: PostprocessConfig) -> dict[str, Any]:
    dark_mode = str(cfg.roi_dark_mode).lower()
    dark_gate_enabled = cfg.good_roi_dark_fraction > 0.0 and (
        dark_mode != "absolute" or cfg.roi_dark_luma_threshold > 0.0
    )
    if not dark_gate_enabled:
        return record

    polygon = record.get("final_box_polygon") or record.get("roi_polygon")
    dark_fraction, effective_threshold, reference_luma = (
        polygon_dark_fraction(
            image_path,
            polygon,
            cfg.roi_dark_luma_threshold,
            mode=dark_mode,
            relative_luma_ratio=cfg.roi_dark_relative_luma_ratio,
            foreground_luma_floor=cfg.roi_dark_foreground_luma_floor,
            analysis_bbox=record.get("effective_image_bbox"),
        )
        if polygon
        else (None, None, None)
    )
    dark_factor = lower_bound_factor(dark_fraction, cfg.min_roi_dark_fraction, cfg.good_roi_dark_fraction)
    record["roi_dark_fraction"] = dark_fraction
    record["roi_dark_factor"] = dark_factor
    record["roi_dark_mode"] = dark_mode
    record["roi_dark_effective_luma_threshold"] = effective_threshold
    record["roi_dark_reference_luma"] = reference_luma
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
    if "cropped_source" in metadata:
        record["cropped_source"] = metadata["cropped_source"]
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
        blackpad_input_dir=args.blackpad_input_dir,
        cropped_input_dir=args.cropped_input_dir,
        blackpad_fraction=args.blackpad_fraction,
        blackpad_min_padding=args.blackpad_min_padding,
        black_border_luma_floor=args.black_border_luma_floor,
    )
    predict_kwargs: dict[str, Any] = {"conf": args.conf, "stream": True, "verbose": False}
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
            result_iter = (
                (str(image_path), result_to_prediction(result, source=str(image_path)), 1)
                for image_path in images
                for result in model.predict(source=str(image_path.resolve()), **predict_kwargs)
            )

        for source, prediction, vote_count in result_iter:
            metadata = metadata_for_source(source_metadata, prediction.source if prediction is not None else source)
            if prediction is None:
                record = {
                    "action": "reject_or_relabel",
                    "final_confidence": 0.0,
                    "flags": ["no_valid_pose_prediction"],
                }
            else:
                prediction = prediction_with_effective_area(prediction, metadata, cfg)
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
