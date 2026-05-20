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
from yoloposevf.run_archive import write_run_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental YOLO-Pose training entry for keypoint-containment loss."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_containment_lambda_sweep.yaml"))
    parser.add_argument("--lambda-containment", type=float, help="Run one lambda value from the sweep.")
    parser.add_argument("--data", type=Path, help="Override dataset YAML path.")
    parser.add_argument("--model", type=str, help="Override YOLO pose checkpoint.")
    parser.add_argument("--name", type=str, help="Override run name.")
    parser.add_argument("--device", type=str, help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Print effective run configs.")
    parser.add_argument("--smoke-loss", action="store_true", help="Run a synthetic containment-loss check.")
    parser.add_argument(
        "--enable-unstable-loss-hook",
        action="store_true",
        help="Attempt the Ultralytics trainer hook. Requires local verification against the installed version.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return values or {}


def build_run_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = load_config(args.config)
    base = dict(raw.get("base", {}))
    loss_cfg = dict(raw.get("containment_loss", {}))
    lambdas = [args.lambda_containment]
    if args.lambda_containment is None:
        lambdas = list(raw.get("sweep", {}).get("lambda_containment", [0.0]))

    if args.data is not None:
        base["data"] = str(args.data)
    if args.model is not None:
        base["model"] = args.model
    if args.device is not None:
        base["device"] = args.device

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
    for lambda_value in lambdas:
        cfg = dict(base)
        cfg["lambda_containment"] = float(lambda_value)
        cfg["containment_loss"] = {
            "margin": float(loss_cfg.get("margin", 0.0)),
            "normalize_by_box_size": bool(loss_cfg.get("normalize_by_box_size", True)),
            "reduction": str(loss_cfg.get("reduction", "mean")),
        }
        cfg["name"] = args.name or f"containment_lambda_{float(lambda_value):g}".replace(".", "p")
        run_configs.append(cfg)
    return run_configs


def synthetic_loss_smoke() -> dict[str, float]:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    keypoints = [
        [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0], [10.0, 10.0]],
        [[-2.0, 5.0], [5.0, 12.0], [15.0, 5.0], [5.0, -3.0]],
    ]
    per_sample = containment_penalty_numpy(boxes, keypoints, reduction="none")
    return {"inside_sample_loss": float(per_sample[0]), "outside_sample_loss": float(per_sample[1])}


def build_experimental_trainer(lambda_containment: float, containment_cfg: dict[str, Any]):
    try:
        from ultralytics.models.yolo.pose import PoseTrainer
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed; use --dry-run or --smoke-loss on this host.") from exc

    class ContainmentPoseTrainer(PoseTrainer):
        """PoseTrainer subclass placeholder for the experimental loss branch.

        Ultralytics does not expose a stable public callback for adding a term
        to pose loss across versions. This subclass is intentionally explicit:
        it records the intended loss settings and refuses full training until
        the installed Ultralytics prediction/loss tensors are inspected and
        wired in this branch.
        """

        containment_lambda = lambda_containment
        containment_loss_config = containment_cfg

        def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore[no-untyped-def]
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model.containment_lambda = self.containment_lambda
            model.containment_loss_config = self.containment_loss_config
            return model

    return ContainmentPoseTrainer


def train_one(cfg: dict[str, Any], *, enable_unstable_loss_hook: bool) -> None:
    if not enable_unstable_loss_hook:
        raise SystemExit(
            "Refusing full training without --enable-unstable-loss-hook. "
            "The reusable loss and sweep entry are dry-run/smoke verified; the Ultralytics loss tensor hook "
            "must be wired against the installed Ultralytics version before real training."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install project requirements first.") from exc

    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    train_kwargs = dict(cfg)
    model_name = train_kwargs.pop("model")
    lambda_containment = float(train_kwargs.pop("lambda_containment"))
    containment_cfg = dict(train_kwargs.pop("containment_loss"))
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str((PROJECT_ROOT / train_kwargs["project"]).resolve())

    trainer_cls = build_experimental_trainer(lambda_containment, containment_cfg)
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
            "containment_loss": containment_cfg,
            "loss_hook_status": "trainer subclass created; prediction tensor addition requires version check",
        },
    )
    print(f"Training finished. Run folder: {save_dir}")


def main() -> None:
    args = parse_args()
    run_configs = build_run_configs(args)
    if args.smoke_loss:
        print(json.dumps(synthetic_loss_smoke(), indent=2, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(run_configs, indent=2, ensure_ascii=False))
    if args.smoke_loss or args.dry_run:
        return
    for cfg in run_configs:
        train_one(cfg, enable_unstable_loss_hook=args.enable_unstable_loss_hook)


if __name__ == "__main__":
    main()
