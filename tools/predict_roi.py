#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import ImageSize
from yoloposevf.postprocess import PosePrediction, fuse_prediction, load_postprocess_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-Pose inference and anatomy-constrained ROI postprocessing.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path, default=Path("configs/postprocess.yaml"))
    parser.add_argument("--out", type=Path, default=Path("Results/predictions/predictions.jsonl"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def result_to_prediction(result: Any) -> PosePrediction | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    if result.keypoints is None or result.keypoints.data is None:
        return None
    keypoints = result.keypoints.data.cpu().numpy()[best]
    height, width = result.orig_shape[:2]
    return PosePrediction(
        bbox=tuple(float(v) for v in boxes[best]),
        bbox_conf=float(confs[best]),
        keypoints=tuple(tuple(float(v) for v in kp[:3]) for kp in keypoints),
        image_size=ImageSize(width=int(width), height=int(height)),
        source=str(result.path),
    )


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install project requirements first.") from exc

    cfg = load_postprocess_config(load_yaml(args.postprocess_config))
    model = YOLO(str(args.weights))
    predict_kwargs: dict[str, Any] = {"source": str(args.source), "conf": args.conf, "stream": True}
    if args.device is not None:
        predict_kwargs["device"] = args.device

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for result in model.predict(**predict_kwargs):
            prediction = result_to_prediction(result)
            if prediction is None:
                record = {
                    "source": str(getattr(result, "path", "")),
                    "action": "reject_or_relabel",
                    "final_confidence": 0.0,
                    "flags": ["no_valid_pose_prediction"],
                }
            else:
                record = fuse_prediction(prediction, cfg)
                record["keypoints"] = [list(kp) for kp in prediction.keypoints]
                record["image_size"] = {
                    "width": prediction.image_size.width,
                    "height": prediction.image_size.height,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} prediction record(s) to {args.out}")


if __name__ == "__main__":
    main()
