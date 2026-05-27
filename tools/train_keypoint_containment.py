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

from yoloposevf.containment_loss import containment_penalty_numpy  # noqa: E402
from yoloposevf.keypoint_contrast import (  # noqa: E402
    KeypointContrastConfig,
    KeypointProjectionHead,
    first_keypoints_per_image,
    keypoint_local_contrast_loss,
    make_light_augmented_view,
)
from yoloposevf.run_archive import write_run_metadata  # noqa: E402

try:
    from ultralytics.models.yolo.pose import PoseTrainer as _UltralyticsPoseTrainer
    from ultralytics.utils.loss import v8PoseLoss as _UltralyticsPoseLoss
except ImportError:
    _UltralyticsPoseTrainer = None
    _UltralyticsPoseLoss = object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental YOLO-Pose training entry for keypoint-containment "
            "and local contrast losses."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_containment_lambda_sweep.yaml"),
    )
    parser.add_argument(
        "--lambda-containment",
        type=float,
        help="Run one lambda value from the sweep.",
    )
    parser.add_argument(
        "--lambda-contrast",
        type=float,
        help="Run one keypoint contrast lambda value from the sweep.",
    )
    parser.add_argument("--data", type=Path, help="Override dataset YAML path.")
    parser.add_argument("--model", type=str, help="Override YOLO pose checkpoint.")
    parser.add_argument("--name", type=str, help="Override run name.")
    parser.add_argument("--project", type=Path, help="Override output project directory.")
    parser.add_argument("--device", type=str, help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--patience", type=int, help="Override early stopping patience.")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run folder.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print effective run configs.")
    parser.add_argument(
        "--smoke-loss",
        action="store_true",
        help="Run a synthetic containment-loss check.",
    )
    parser.add_argument(
        "--smoke-contrast",
        action="store_true",
        help="Run a synthetic keypoint-contrast-loss check.",
    )
    parser.add_argument(
        "--enable-unstable-loss-hook",
        action="store_true",
        help=(
            "Attempt the Ultralytics trainer hook. Requires local verification "
            "against the installed version."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return values or {}


def build_run_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = load_config(args.config)
    base = dict(raw.get("base", {}))
    loss_cfg = dict(raw.get("containment_loss", {}))
    contrast_cfg = dict(raw.get("keypoint_contrast", {}))
    containment_lambdas = [args.lambda_containment]
    if args.lambda_containment is None:
        containment_lambdas = list(raw.get("sweep", {}).get("lambda_containment", [0.0]))
    contrast_lambdas = [args.lambda_contrast]
    if args.lambda_contrast is None:
        contrast_lambdas = list(
            raw.get("sweep", {}).get(
                "lambda_contrast",
                [float(base.get("lambda_contrast", 0.0))],
            )
        )

    if args.data is not None:
        base["data"] = str(args.data)
    if args.model is not None:
        base["model"] = args.model
    if args.project is not None:
        base["project"] = str(args.project)
    if args.device is not None:
        base["device"] = args.device
    for key in ("epochs", "batch", "workers", "imgsz", "patience"):
        value = getattr(args, key)
        if value is not None:
            base[key] = value
    if args.exist_ok:
        base["exist_ok"] = True

    base.setdefault("model", "yolo11n-pose.pt")
    base.setdefault("data", "data/yolo_pose/vocal_fold_pose.yaml")
    base.setdefault("project", "Results/containment_loss")
    base.setdefault("imgsz", 640)
    base.setdefault("epochs", 150)
    base.setdefault("batch", 8)
    base.setdefault("workers", 4)
    base.setdefault("patience", 40)
    base.setdefault("seed", 42)
    base.setdefault("exist_ok", False)

    run_configs = []
    for lambda_containment in containment_lambdas:
        for lambda_contrast in contrast_lambdas:
            cfg = dict(base)
            cfg["lambda_containment"] = float(lambda_containment)
            cfg["lambda_contrast"] = float(lambda_contrast)
            cfg["containment_loss"] = {
                "margin": float(loss_cfg.get("margin", 0.0)),
                "normalize_by_box_size": bool(loss_cfg.get("normalize_by_box_size", True)),
                "reduction": str(loss_cfg.get("reduction", "mean")),
            }
            cfg["keypoint_contrast"] = {
                "feature_index": int(contrast_cfg.get("feature_index", 0)),
                "projection_dim": int(contrast_cfg.get("projection_dim", 64)),
                "projection_hidden_dim": int(contrast_cfg.get("projection_hidden_dim", 128)),
                "temperature": float(contrast_cfg.get("temperature", 0.10)),
                "patch_radius": int(contrast_cfg.get("patch_radius", 1)),
                "negative_count": int(contrast_cfg.get("negative_count", 12)),
                "negative_min_distance": float(contrast_cfg.get("negative_min_distance", 0.12)),
                "augment_degrees": float(contrast_cfg.get("augment_degrees", 5.0)),
                "augment_scale": float(contrast_cfg.get("augment_scale", 0.08)),
                "augment_translate": float(contrast_cfg.get("augment_translate", 0.03)),
                "augment_brightness": float(contrast_cfg.get("augment_brightness", 0.08)),
                "augment_contrast": float(contrast_cfg.get("augment_contrast", 0.12)),
            }
            default_name = (
                f"containment_l{float(lambda_containment):g}_contrast_l{float(lambda_contrast):g}"
            ).replace(".", "p")
            cfg["name"] = args.name or str(base.get("name", default_name))
            if args.name is None and "lambda_contrast" in raw.get("sweep", {}):
                cfg["name"] = (
                    f"{cfg['name']}_contrast_l{float(lambda_contrast):g}".replace(".", "p")
                )
            run_configs.append(cfg)
    return run_configs


def synthetic_loss_smoke() -> dict[str, float]:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    keypoints = [
        [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]],
        [[-2.0, 5.0], [5.0, 12.0], [15.0, 5.0]],
    ]
    per_sample = containment_penalty_numpy(boxes, keypoints, reduction="none")
    return {"inside_sample_loss": float(per_sample[0]), "outside_sample_loss": float(per_sample[1])}


def synthetic_contrast_smoke() -> dict[str, float]:
    import torch

    from yoloposevf.keypoint_contrast import keypoint_local_contrast_loss

    torch.manual_seed(42)
    features1 = torch.randn(2, 8, 12, 12)
    features2 = features1 + torch.randn_like(features1) * 0.01
    keypoints = torch.tensor(
        [
            [[0.30, 0.30], [0.60, 0.35], [0.50, 0.70]],
            [[0.25, 0.45], [0.70, 0.45], [0.48, 0.74]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((2, 3), dtype=torch.bool)
    head = KeypointProjectionHead(8, hidden_dim=16, out_dim=8)
    matched = keypoint_local_contrast_loss(
        features1,
        features2,
        keypoints,
        keypoints,
        mask,
        head,
        temperature=0.10,
        patch_radius=1,
        negative_count=4,
    )
    shifted = keypoint_local_contrast_loss(
        features1,
        torch.roll(features2, shifts=3, dims=-1),
        keypoints,
        keypoints,
        mask,
        head,
        temperature=0.10,
        patch_radius=1,
        negative_count=4,
    )
    return {"matched_loss": float(matched.detach()), "shifted_loss": float(shifted.detach())}


class ContainmentPoseLoss(_UltralyticsPoseLoss):
    """YOLO-Pose loss with an added predicted-bbox/keypoint containment term."""

    lambda_containment: float = 0.0
    containment_cfg: dict[str, Any] = {}
    lambda_contrast: float = 0.0
    contrast_cfg: dict[str, Any] = {}

    def __init__(  # type: ignore[no-untyped-def]
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int = 10,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.model = model
        self.lambda_containment = float(self.__class__.lambda_containment)
        self.containment_cfg = dict(self.__class__.containment_cfg)
        self.lambda_contrast = float(self.__class__.lambda_contrast)
        self.contrast_cfg = dict(self.__class__.contrast_cfg)
        self.last_containment_loss = None
        self.last_contrast_loss = None

    def loss(self, preds, batch):  # type: ignore[no-untyped-def]
        import torch

        from yoloposevf.containment_loss import containment_penalty_torch

        pred_kpts_raw = preds["kpts"].permute(0, 2, 1).contiguous()
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts_raw.shape[0]
        imgsz = torch.tensor(
            preds["feats"][0].shape[2:],
            device=self.device,
            dtype=pred_kpts_raw.dtype,
        ) * self.stride[0]

        pred_kpts = self.kpts_decode(
            anchor_points,
            pred_kpts_raw.view(batch_size, -1, *self.kpt_shape),
        )
        containment_loss = torch.zeros((), device=self.device, dtype=pred_kpts.dtype)
        contrast_loss = torch.zeros((), device=self.device, dtype=pred_kpts.dtype)

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

            if self.lambda_containment > 0:
                pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
                selected_keypoints = self._select_target_keypoints(
                    keypoints,
                    batch["batch_idx"].view(-1, 1),
                    target_gt_idx,
                    fg_mask,
                )
                selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
                visibility = (
                    selected_keypoints[fg_mask][..., 2] != 0
                    if selected_keypoints.shape[-1] == 3
                    else None
                )
                containment_loss = containment_penalty_torch(
                    pred_bboxes[fg_mask],
                    pred_kpts[fg_mask][..., :2],
                    visibility=visibility,
                    margin=float(self.containment_cfg.get("margin", 0.0)),
                    reduction=str(self.containment_cfg.get("reduction", "mean")),
                    normalize_by_box_size=bool(
                        self.containment_cfg.get("normalize_by_box_size", True)
                    ),
                )

        if self.lambda_contrast > 0 and self.model.training:
            contrast_loss = self._calculate_keypoint_contrast_loss(preds, batch, batch_size)

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[1] += self.lambda_containment * containment_loss
        loss[1] += self.lambda_contrast * contrast_loss
        self.last_containment_loss = containment_loss.detach()
        self.last_contrast_loss = contrast_loss.detach()
        return loss * batch_size, loss.detach()

    def _calculate_keypoint_contrast_loss(  # type: ignore[no-untyped-def]
        self,
        preds,
        batch,
        batch_size: int,
    ):
        keypoint_count = int(self.kpt_shape[0])
        keypoints01, keypoint_mask = first_keypoints_per_image(
            batch["keypoints"].to(self.device).float(),
            batch["batch_idx"].to(self.device),
            batch_size,
            keypoint_count,
        )
        if not keypoint_mask.any():
            return preds["feats"][0].new_zeros(())

        contrast_cfg = KeypointContrastConfig(**self.contrast_cfg)
        feature_index = int(contrast_cfg.feature_index)
        if feature_index < 0 or feature_index >= len(preds["feats"]):
            raise ValueError(
                f"feature_index={feature_index} is outside available feature maps: "
                f"{len(preds['feats'])}"
            )

        images2, keypoints2, mask2 = make_light_augmented_view(
            batch["img"].to(self.device).float(),
            keypoints01,
            keypoint_mask,
            degrees=contrast_cfg.augment_degrees,
            scale=contrast_cfg.augment_scale,
            translate=contrast_cfg.augment_translate,
            brightness=contrast_cfg.augment_brightness,
            contrast=contrast_cfg.augment_contrast,
        )
        valid_mask = keypoint_mask & mask2
        if not valid_mask.any():
            return preds["feats"][feature_index].new_zeros(())

        preds2 = self.parse_output(self.model.forward(images2))
        projection_head = getattr(self.model, "keypoint_contrast_head", None)
        if projection_head is None:
            raise RuntimeError("keypoint_contrast_head is missing from the YOLO pose model.")
        return keypoint_local_contrast_loss(
            preds["feats"][feature_index],
            preds2["feats"][feature_index],
            keypoints01,
            keypoints2,
            valid_mask,
            projection_head,
            temperature=contrast_cfg.temperature,
            patch_radius=contrast_cfg.patch_radius,
            negative_count=contrast_cfg.negative_count,
            negative_min_distance=contrast_cfg.negative_min_distance,
        )


def init_containment_criterion(pose_model):  # type: ignore[no-untyped-def]
    return ContainmentPoseLoss(pose_model)


_BaseContainmentPoseTrainer = (
    _UltralyticsPoseTrainer if _UltralyticsPoseTrainer is not None else object
)


class ContainmentPoseTrainer(_BaseContainmentPoseTrainer):
    """PoseTrainer subclass that installs the containment-loss criterion."""

    containment_lambda: float = 0.0
    containment_loss_config: dict[str, Any] = {}
    contrast_lambda: float = 0.0
    contrast_loss_config: dict[str, Any] = {}

    def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore[no-untyped-def]
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.containment_lambda = self.containment_lambda
        model.containment_loss_config = self.containment_loss_config
        model.contrast_lambda = self.contrast_lambda
        model.contrast_loss_config = self.contrast_loss_config
        if self.contrast_lambda > 0:
            contrast_cfg = KeypointContrastConfig(**self.contrast_loss_config)
            in_channels = _feature_channels_for_contrast(model, contrast_cfg.feature_index)
            model.keypoint_contrast_head = KeypointProjectionHead(
                in_channels,
                hidden_dim=contrast_cfg.projection_hidden_dim,
                out_dim=contrast_cfg.projection_dim,
            )
        model.__class__.init_criterion = init_containment_criterion
        return model

    def save_model(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        criterion = getattr(self.model, "criterion", None)
        self.model.criterion = None
        try:
            return super().save_model(*args, **kwargs)
        finally:
            self.model.criterion = criterion


def _feature_channels_for_contrast(  # type: ignore[no-untyped-def]
    model,
    feature_index: int,
) -> int:
    pose_head = model.model[-1]
    if not hasattr(pose_head, "cv4") or feature_index < 0 or feature_index >= len(pose_head.cv4):
        raise ValueError(
            f"Cannot infer contrast feature channels for feature_index={feature_index}."
        )
    first_block = pose_head.cv4[feature_index][0]
    conv = getattr(first_block, "conv", None)
    if conv is None or not hasattr(conv, "in_channels"):
        raise ValueError("Cannot infer contrast feature channels from the pose head.")
    return int(conv.in_channels)


def build_experimental_trainer(
    lambda_containment: float,
    containment_cfg: dict[str, Any],
    lambda_contrast: float,
    contrast_cfg: dict[str, Any],
):
    if _UltralyticsPoseTrainer is None:
        raise SystemExit(
            "ultralytics is not installed; use --dry-run or --smoke-loss on this host."
        )
    ContainmentPoseLoss.lambda_containment = float(lambda_containment)
    ContainmentPoseLoss.containment_cfg = dict(containment_cfg)
    ContainmentPoseLoss.lambda_contrast = float(lambda_contrast)
    ContainmentPoseLoss.contrast_cfg = dict(contrast_cfg)
    ContainmentPoseTrainer.containment_lambda = float(lambda_containment)
    ContainmentPoseTrainer.containment_loss_config = dict(containment_cfg)
    ContainmentPoseTrainer.contrast_lambda = float(lambda_contrast)
    ContainmentPoseTrainer.contrast_loss_config = dict(contrast_cfg)
    return ContainmentPoseTrainer


def train_one(cfg: dict[str, Any], *, enable_unstable_loss_hook: bool) -> None:
    if not enable_unstable_loss_hook:
        raise SystemExit(
            "Refusing full training without --enable-unstable-loss-hook. "
            "The containment criterion is wired for the Ultralytics 8.4.x v8PoseLoss "
            "tensor flow, but it is still "
            "an experimental branch and should be launched deliberately."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install project requirements first."
        ) from exc

    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    train_kwargs = dict(cfg)
    model_name = train_kwargs.pop("model")
    lambda_containment = float(train_kwargs.pop("lambda_containment"))
    lambda_contrast = float(train_kwargs.pop("lambda_contrast"))
    containment_cfg = dict(train_kwargs.pop("containment_loss"))
    contrast_cfg = dict(train_kwargs.pop("keypoint_contrast"))
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str((PROJECT_ROOT / train_kwargs["project"]).resolve())

    trainer_cls = build_experimental_trainer(
        lambda_containment,
        containment_cfg,
        lambda_contrast,
        contrast_cfg,
    )
    model = YOLO(model_name)
    results = model.train(trainer=trainer_cls, **train_kwargs)
    save_dir = Path(getattr(results, "save_dir", train_kwargs["project"]))
    write_run_metadata(
        run_dir=save_dir,
        project_root=PROJECT_ROOT,
        command=sys.argv,
        config={
            "model": model_name,
            **train_kwargs,
            "lambda_containment": lambda_containment,
            "lambda_contrast": lambda_contrast,
            "containment_loss": containment_cfg,
            "keypoint_contrast": contrast_cfg,
            "loss_hook_status": (
                "ContainmentPoseLoss subclass installed via PoseModel.init_criterion"
            ),
        },
    )
    print(f"Training finished. Run folder: {save_dir}")


def main() -> None:
    args = parse_args()
    run_configs = build_run_configs(args)
    if args.smoke_loss:
        print(json.dumps(synthetic_loss_smoke(), indent=2, ensure_ascii=False))
    if args.smoke_contrast:
        print(json.dumps(synthetic_contrast_smoke(), indent=2, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(run_configs, indent=2, ensure_ascii=False))
    if args.smoke_loss or args.smoke_contrast or args.dry_run:
        return
    for cfg in run_configs:
        train_one(cfg, enable_unstable_loss_hook=args.enable_unstable_loss_hook)


if __name__ == "__main__":
    main()
