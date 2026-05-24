#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_dinov3_keypoint_aux import letterbox_tensor  # noqa: E402
from yoloposevf.dinov3_aux import (  # noqa: E402
    DinoV3AuxConfig,
    DinoV3KeypointAuxHead,
    foreground_mask_from_images,
    load_dinov3_extractor,
    normalize_for_dinov3,
    score_aux_triplet,
)
from yoloposevf.postprocess import decide_action, load_postprocess_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach frozen-DINOv3 auxiliary scores to ROI prediction JSONL.")
    parser.add_argument("--aux-checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path, default=Path("configs/postprocess.yaml"))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, help="Override DINO input size from aux config.")
    parser.add_argument("--apply-confidence-gate", action="store_true")
    parser.add_argument("--min-point-prob", type=float, default=0.30)
    parser.add_argument("--min-triplet-prob", type=float, default=0.30)
    parser.add_argument("--max-image-reject-prob", type=float, default=0.70)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_aux_config_from_checkpoint(checkpoint: dict[str, Any]) -> DinoV3AuxConfig:
    cfg = checkpoint["config"]
    dinov3 = dict(cfg["dinov3"])
    defaults = DinoV3AuxConfig()
    dinov3.setdefault("include_valid_mask", False)
    gate_keys = (
        "confidence_gate_mode",
        "confidence_reject_threshold",
        "confidence_penalty_threshold",
        "confidence_reward_threshold",
        "confidence_direct_accept_threshold",
        "confidence_reward_multiplier",
    )
    if "confidence_gate_mode" not in dinov3:
        for key in gate_keys:
            dinov3[key] = getattr(defaults, key)
    else:
        for key in gate_keys:
            dinov3.setdefault(key, getattr(defaults, key))
    return DinoV3AuxConfig(**dinov3)


def device_from_arg(value: str) -> torch.device:
    if value.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def prediction_image_path(record: dict[str, Any]) -> Path | None:
    for field in ("source", "original_source"):
        value = record.get(field)
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def prediction_keypoints(record: dict[str, Any]) -> list[list[float]] | None:
    rows = record.get("keypoints")
    if not rows or len(rows) < 3:
        return None
    return [[float(value) for value in row[:3]] for row in rows[:3]]


def letterbox_keypoints(
    keypoints: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    imgsz: int,
) -> torch.Tensor:
    points = []
    for kp in keypoints[:3]:
        x = float(kp[0]) * scale + pad_x
        y = float(kp[1]) * scale + pad_y
        points.append([x / max(float(imgsz), 1.0), y / max(float(imgsz), 1.0)])
    return torch.tensor(points, dtype=torch.float32)


def attach_aux_score(
    record: dict[str, Any],
    *,
    image_path: Path,
    keypoints: Sequence[Sequence[float]],
    extractor: Any,
    head: DinoV3KeypointAuxHead,
    aux_cfg: DinoV3AuxConfig,
    device: torch.device,
    imgsz: int,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_tensor, scale, pad_x, pad_y = letterbox_tensor(image, imgsz)
    points = letterbox_keypoints(
        keypoints,
        width=width,
        height=height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        imgsz=imgsz,
    )
    with torch.no_grad():
        images = image_tensor[None].to(device)
        feature_map, global_feature = extractor.forward_dense(normalize_for_dinov3(images, aux_cfg))
        valid_mask_map = (
            foreground_mask_from_images(images, luma_floor=aux_cfg.valid_mask_luma_floor)
            if aux_cfg.include_valid_mask
            else None
        )
        scores = score_aux_triplet(
            head,
            feature_map,
            global_feature,
            points[None].to(device),
            patch_size_input=aux_cfg.oriented_patch_size_input,
            input_size=imgsz,
            score_mode=aux_cfg.score_mode,
            gate_mode=aux_cfg.confidence_gate_mode,
            reject_threshold=aux_cfg.confidence_reject_threshold,
            penalty_threshold=aux_cfg.confidence_penalty_threshold,
            reward_threshold=aux_cfg.confidence_reward_threshold,
            direct_accept_threshold=aux_cfg.confidence_direct_accept_threshold,
            reward_multiplier=aux_cfg.confidence_reward_multiplier,
            valid_mask_map=valid_mask_map,
        )
    point_probs = scores["point_expected_probs"][0].detach().cpu().tolist()
    point_region_score = float(scores["point_region_score"][0].detach().cpu())
    confidence_factor = float(scores["confidence_factor"][0].detach().cpu())
    direct_accept = bool(scores["direct_accept"][0].detach().cpu())
    hard_reject = bool(scores["hard_reject"][0].detach().cpu())
    valid_fraction = float(scores["valid_fraction"][0].detach().cpu())
    record["dinov3_aux"] = {
        "point_expected_probs": {
            "anterior": point_probs[0],
            "left_posterior": point_probs[1],
            "right_posterior": point_probs[2],
        },
        "min_point_expected_prob": float(min(point_probs)),
        "point_region_score": point_region_score,
        "confidence_factor": confidence_factor,
        "direct_accept": direct_accept,
        "hard_reject": hard_reject,
        "valid_fraction": valid_fraction,
        "gate_mode": aux_cfg.confidence_gate_mode,
        "reject_threshold": aux_cfg.confidence_reject_threshold,
        "penalty_threshold": aux_cfg.confidence_penalty_threshold,
        "reward_threshold": aux_cfg.confidence_reward_threshold,
        "direct_accept_threshold": aux_cfg.confidence_direct_accept_threshold,
        "reward_multiplier": aux_cfg.confidence_reward_multiplier,
        "image_source": str(image_path),
    }
    return record


def maybe_apply_gate(
    record: dict[str, Any],
    *,
    postprocess_cfg: Any,
    min_point_prob: float,
    min_triplet_prob: float,
    max_image_reject_prob: float,
) -> dict[str, Any]:
    aux = record.get("dinov3_aux") or {}
    if not aux:
        return record
    old_conf = float(record.get("final_confidence", 0.0))
    factor = float(aux.get("confidence_factor", 1.0))
    direct_accept = bool(aux.get("direct_accept", False))
    hard_reject = bool(aux.get("hard_reject", False))
    record["pre_dinov3_aux_confidence"] = old_conf
    record["final_confidence"] = max(0.0, min(1.0, old_conf * factor))
    if hard_reject:
        record["final_confidence"] = 0.0
        record["action"] = "reject_or_relabel"
        record["dinov3_aux_gate_action"] = "hard_reject"
        record.setdefault("flags", []).append("dinov3_very_low_point_probability")
    elif direct_accept:
        record["dinov3_aux_gate_action"] = "direct_accept"
        record["final_confidence"] = max(float(record["final_confidence"]), float(postprocess_cfg.auto_accept_threshold))
        record["action"] = "auto_accept"
    else:
        record["dinov3_aux_gate_action"] = "reward" if factor > 1.0 else "none"
        record["action"] = decide_action(float(record["final_confidence"]), postprocess_cfg)
    if record["action"] == "reject_or_relabel":
        record["usable_bbox"] = None
        record["usable_box_polygon"] = None
    else:
        record["usable_bbox"] = record.get("final_bbox")
        record["usable_box_polygon"] = record.get("final_box_polygon")
    return record


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.aux_checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    aux_cfg = load_aux_config_from_checkpoint(checkpoint)
    imgsz = int(args.imgsz or cfg.get("imgsz", 960))
    extractor = load_dinov3_extractor(aux_cfg, device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    head = DinoV3KeypointAuxHead(
        int(checkpoint["feature_dim"]),
        patch_output_size=aux_cfg.oriented_patch_output_size,
        point_hidden_dim=aux_cfg.point_hidden_dim,
        include_coordinates=aux_cfg.include_point_coordinates,
        include_valid_mask=aux_cfg.include_valid_mask,
    ).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    postprocess_cfg = load_postprocess_config(load_yaml(args.postprocess_config))

    records = read_jsonl(args.predictions)
    written = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            image_path = prediction_image_path(record)
            keypoints = prediction_keypoints(record)
            if image_path is None or keypoints is None:
                skipped += 1
            else:
                record = attach_aux_score(
                    record,
                    image_path=image_path,
                    keypoints=keypoints,
                    extractor=extractor,
                    head=head,
                    aux_cfg=aux_cfg,
                    device=device,
                    imgsz=imgsz,
                )
                if args.apply_confidence_gate:
                    record = maybe_apply_gate(
                        record,
                        postprocess_cfg=postprocess_cfg,
                        min_point_prob=args.min_point_prob,
                        min_triplet_prob=args.min_triplet_prob,
                        max_image_reject_prob=args.max_image_reject_prob,
                    )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "skipped": skipped, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
