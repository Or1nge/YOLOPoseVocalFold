#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.postprocess import PosePrediction, PostprocessConfig, decide_action, fuse_prediction, load_postprocess_config
from yoloposevf.preprocess import IMAGE_EXTENSIONS, blackpad_image_file
from yoloposevf.screen_photo_crop import classify_screen_photo, crop_screen_photo_window


ALGORITHM_VERSION = "V1.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-Pose inference and anatomy-constrained ROI postprocessing.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Image file or directory. Not required when --manifest is used.")
    parser.add_argument("--manifest", type=Path, help="JSONL manifest with image paths in original_source/source_key/source.")
    parser.add_argument("--postprocess-config", type=Path, default=Path("configs/postprocess.yaml"))
    parser.add_argument("--out", type=Path, default=Path("Results/predictions/predictions.jsonl"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0", help="Inference device. Defaults to GPU 0; use cpu to force CPU.")
    parser.add_argument("--imgsz", type=int, help="Inference image size. Use the training size for best keypoint precision.")
    parser.add_argument("--tta", action="store_true", help="Use rotation/scale test-time augmentation without flips.")
    parser.add_argument("--tta-degrees", default="-6,0,6", help="Comma-separated rotation degrees for --tta.")
    parser.add_argument("--tta-scales", default="0.95,1.0,1.05", help="Comma-separated scale factors for --tta.")
    parser.add_argument("--blackpad-fraction", type=float, default=0.30)
    parser.add_argument("--blackpad-min-padding", type=int, default=80)
    parser.add_argument("--blackpad-input-dir", type=Path, help="Directory for generated black-padded inference inputs.")
    parser.add_argument("--cropped-input-dir", type=Path, help="Directory for generated cropped/no-black DINO inputs.")
    parser.add_argument("--precrop-input-dir", type=Path, help="Directory for screen-photo pre-cropped intermediate images.")
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


def read_manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def manifest_image_path(record: dict[str, Any]) -> Path:
    for field in ("original_source", "source_key", "source"):
        value = record.get(field)
        if value:
            return Path(str(value))
    raise ValueError("manifest record has no original_source/source_key/source path")


def unique_manifest_relative_path(record: dict[str, Any], image_path: Path, index: int, used: set[Path]) -> Path:
    class_name = str(record.get("class_name") or "manifest")
    candidate = Path(class_name) / image_path.name
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = image_path.stem
    suffix = image_path.suffix
    for counter in range(1, 10000):
        candidate = Path(class_name) / f"{stem}_{index:05d}_{counter}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError(f"could not make unique manifest path for {image_path}")


def default_blackpad_input_dir(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_blackpad_inputs"


def default_cropped_input_dir(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_cropped_inputs"


def default_precrop_input_dir(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_precrop_inputs"


def relative_image_path(image_path: Path, source: Path) -> Path:
    if source.is_file():
        return Path(image_path.name)
    return image_path.relative_to(source)


def _build_pre_crop_info(
    *,
    triggered: bool,
    mode: str,
    reason: list[str] | None,
    box_xyxy: tuple[int, int, int, int] | None,
    signals: dict[str, Any] | None,
    original_width: int,
    original_height: int,
    cropped_width: int | None,
    cropped_height: int | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "triggered": triggered,
        "mode": mode,
        "reason": reason,
        "box_xyxy": list(box_xyxy) if box_xyxy is not None else None,
        "signals": signals,
        "original_width": original_width,
        "original_height": original_height,
        "cropped_width": cropped_width,
        "cropped_height": cropped_height,
    }
    return info


def prepare_inference_images(
    source: Path | None,
    *,
    manifest: Path | None,
    out_path: Path,
    blackpad_input_dir: Path | None,
    cropped_input_dir: Path | None,
    precrop_input_dir: Path | None = None,
    blackpad_fraction: float = 0.30,
    blackpad_min_padding: int = 80,
    black_border_luma_floor: float = 8.0,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    manifest_records = read_manifest_records(manifest) if manifest is not None else []
    if manifest_records:
        image_rows = [(manifest_image_path(record), record) for record in manifest_records]
    else:
        if source is None:
            raise ValueError("--source is required when --manifest is not provided")
        image_rows = [(image_path, {}) for image_path in iter_images(source)]
    metadata: dict[str, dict[str, Any]] = {}
    pad_root = (blackpad_input_dir or default_blackpad_input_dir(out_path)).resolve()
    crop_root = (cropped_input_dir or default_cropped_input_dir(out_path)).resolve()
    precrop_root = (precrop_input_dir or default_precrop_input_dir(out_path)).resolve()
    model_input_images: list[Path] = []
    used_rel_paths: set[Path] = set()
    for index, (image_path, record) in enumerate(image_rows):
        if manifest_records:
            rel = unique_manifest_relative_path(record, image_path, index, used_rel_paths)
        else:
            assert source is not None
            rel = relative_image_path(image_path, source)

        # Step 0: Screen-photo pre-crop classification and optional cropping.
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            triggered, signals = classify_screen_photo(img)
            if triggered:
                pre_cropped, pre_crop_box = crop_screen_photo_window(img)
                precrop_dest = precrop_root / rel
                precrop_dest.parent.mkdir(parents=True, exist_ok=True)
                pre_cropped.save(precrop_dest, quality=95, subsampling=0)
                working_source = precrop_dest
                pre_crop_info = _build_pre_crop_info(
                    triggered=True,
                    mode="screen_photo_precrop",
                    reason=signals.get("reason", []),
                    box_xyxy=pre_crop_box,
                    signals={k: v for k, v in signals.items() if k != "reason"},
                    original_width=img.width,
                    original_height=img.height,
                    cropped_width=pre_cropped.width,
                    cropped_height=pre_cropped.height,
                )
            else:
                working_source = image_path
                pre_crop_info = _build_pre_crop_info(
                    triggered=False,
                    mode="none",
                    reason=signals.get("reason", ["none"]),
                    box_xyxy=None,
                    signals={k: v for k, v in signals.items() if k != "reason"},
                    original_width=img.width,
                    original_height=img.height,
                    cropped_width=None,
                    cropped_height=None,
                )

        # Step 1: Black-border crop + blackpad (existing V1.2 pipeline).
        cropped_destination = crop_root / rel
        destination = pad_root / rel
        info = blackpad_image_file(
            working_source,
            destination,
            fraction=blackpad_fraction,
            min_padding=blackpad_min_padding,
            black_border_luma_floor=black_border_luma_floor,
            cropped_destination=cropped_destination,
        )
        model_input_images.append(destination)
        cropped_source = str(cropped_destination.resolve())
        preprocess_info = info.to_dict()
        preprocess_info["pre_crop"] = pre_crop_info
        metadata[str(destination.resolve())] = {
            "source": str(destination.resolve()),
            "original_source": str(image_path.resolve()),
            "dinov3_source": cropped_source,
            "cropped_source": cropped_source,
            "preprocess": preprocess_info,
            "pre_crop": pre_crop_info,
            "manifest_index": index,
            "class_name": record.get("class_name"),
            "source_key": record.get("source_key"),
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
    mask_array = np.asarray(mask, dtype=bool)
    if not mask_array.any():
        return 0.0, None, None
    effective_threshold, reference_luma = _relative_dark_threshold(
        image_array,
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
    foreground_floor = float(cfg.roi_dark_foreground_luma_floor)
    preprocess = metadata.get("preprocess") or {}
    preprocess_type = str(preprocess.get("type", "")).lower()
    if preprocess_type in {"crop_black_border_then_blackpad", "crop_black_borders_then_blackpad", "crop_black_border"}:
        no_black_bbox = preprocess.get("no_black_bbox_in_model_input")
        cropped_width = float(preprocess.get("no_black_width") or preprocess.get("cropped_width") or 0.0)
        cropped_height = float(preprocess.get("no_black_height") or preprocess.get("cropped_height") or 0.0)
        if no_black_bbox and len(no_black_bbox) == 4:
            bbox = tuple(float(value) for value in no_black_bbox)
            x1, y1, x2, y2 = bbox
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            return (area if area > 0.0 else None), bbox, "preprocess_no_black_content"
        padding = float(preprocess.get("padding_px") or 0.0)
        if cropped_width > 0.0 and cropped_height > 0.0:
            bbox = (padding, padding, padding + cropped_width, padding + cropped_height)
            return cropped_width * cropped_height, bbox, "preprocess_no_black_content"
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
    return metadata.get(
        str(path.resolve()),
        {
            "source": str(source),
            "original_source": str(source),
            "dinov3_source": str(source),
            "cropped_source": str(source),
            "preprocess": {"type": "none"},
        },
    )


def attach_prediction_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    record["algorithm_version"] = ALGORITHM_VERSION
    record["source"] = metadata["source"]
    record["original_source"] = metadata["original_source"]
    if metadata.get("dinov3_source") is not None:
        record["dinov3_source"] = metadata["dinov3_source"]
    if metadata.get("cropped_source") is not None:
        record["cropped_source"] = metadata["cropped_source"]
    record["preprocess"] = metadata["preprocess"]
    if metadata.get("pre_crop") is not None:
        record["pre_crop"] = metadata["pre_crop"]
    for field in ("manifest_index", "class_name", "source_key"):
        if metadata.get(field) is not None:
            record[field] = metadata[field]
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
        manifest=args.manifest,
        out_path=args.out,
        blackpad_input_dir=args.blackpad_input_dir,
        cropped_input_dir=args.cropped_input_dir,
        precrop_input_dir=args.precrop_input_dir,
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
