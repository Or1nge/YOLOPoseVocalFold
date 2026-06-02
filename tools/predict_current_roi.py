#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_MODEL_DIR = Path("Results/models/vf_roi_current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current main ROI model: YOLO-Pose followed by DINOv3 auxiliary gating."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Image file or directory.")
    source.add_argument("--manifest", type=Path, help="JSONL manifest with image paths.")
    parser.add_argument("--out", type=Path, required=True, help="Final DINO-gated prediction JSONL.")
    parser.add_argument(
        "--pose-out",
        type=Path,
        help="Intermediate YOLO-Pose prediction JSONL. Defaults beside --out.",
    )
    parser.add_argument("--pose-weights", type=Path, default=CURRENT_MODEL_DIR / "pose_best.pt")
    parser.add_argument("--aux-checkpoint", type=Path, default=CURRENT_MODEL_DIR / "aux_best.pt")
    parser.add_argument("--postprocess-config", type=Path, default=CURRENT_MODEL_DIR / "postprocess.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, help="YOLO-Pose inference image size.")
    parser.add_argument("--dinov3-imgsz", type=int, help="Override DINOv3 scoring image size.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--preprocess-workers", type=int, default=0)
    parser.add_argument("--blackpad-fraction", type=float, default=0.30)
    parser.add_argument("--blackpad-min-padding", type=int, default=80)
    parser.add_argument("--black-border-luma-floor", type=float, default=8.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-degrees", default="-6,0,6")
    parser.add_argument("--tta-scales", default="0.95,1.0,1.05")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Persist YOLO pre-crop/no-black/blackpad intermediate images. Defaults to metadata-only in-memory preprocessing.",
    )
    parser.add_argument(
        "--no-apply-confidence-gate",
        action="store_true",
        help="Attach DINOv3 scores without changing final_confidence/action.",
    )
    return parser.parse_args()


def default_pose_out(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_pose_raw.jsonl"


def run_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def append_optional_path(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend([flag, str(path)])


def append_optional_value(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> None:
    args = parse_args()
    out_path = args.out
    pose_out = args.pose_out or default_pose_out(out_path)
    if pose_out == out_path:
        raise SystemExit("--pose-out must be different from --out")

    pose_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predict_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/predict_roi.py"),
        "--weights",
        str(args.pose_weights),
        "--postprocess-config",
        str(args.postprocess_config),
        "--out",
        str(pose_out),
        "--conf",
        str(args.conf),
        "--device",
        str(args.device),
        "--batch",
        str(args.batch),
        "--preprocess-workers",
        str(args.preprocess_workers),
        "--blackpad-fraction",
        str(args.blackpad_fraction),
        "--blackpad-min-padding",
        str(args.blackpad_min_padding),
        "--black-border-luma-floor",
        str(args.black_border_luma_floor),
    ]
    append_optional_path(predict_cmd, "--source", args.source)
    append_optional_path(predict_cmd, "--manifest", args.manifest)
    append_optional_value(predict_cmd, "--imgsz", args.imgsz)
    if args.save_intermediates:
        predict_cmd.append("--save-intermediates")
    if args.tta:
        predict_cmd.extend(["--tta", "--tta-degrees", args.tta_degrees, "--tta-scales", args.tta_scales])

    score_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/score_predictions_with_dinov3_aux.py"),
        "--aux-checkpoint",
        str(args.aux_checkpoint),
        "--predictions",
        str(pose_out),
        "--out",
        str(out_path),
        "--postprocess-config",
        str(args.postprocess_config),
        "--device",
        str(args.device),
    ]
    append_optional_value(score_cmd, "--imgsz", args.dinov3_imgsz)
    if not args.no_apply_confidence_gate:
        score_cmd.append("--apply-confidence-gate")

    run_command(predict_cmd)
    run_command(score_cmd)


if __name__ == "__main__":
    main()
