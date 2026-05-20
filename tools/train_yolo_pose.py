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

from yoloposevf.run_archive import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standard YOLO-Pose vocal fold ROI baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_baseline.yaml"))
    parser.add_argument("--data", type=Path, help="Override dataset YAML path.")
    parser.add_argument("--model", type=str, help="Override YOLO pose checkpoint, e.g. yolo11n-pose.pt.")
    parser.add_argument("--name", type=str, help="Override run name.")
    parser.add_argument("--device", type=str, help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without starting training.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return values or {}


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.data is not None:
        cfg["data"] = str(args.data)
    if args.model is not None:
        cfg["model"] = args.model
    if args.name is not None:
        cfg["name"] = args.name
    if args.device is not None:
        cfg["device"] = args.device
    cfg.setdefault("model", "yolo11n-pose.pt")
    cfg.setdefault("data", "data/yolo_pose/vocal_fold_pose.yaml")
    cfg.setdefault("project", "Results/baseline")
    cfg.setdefault("name", "yolo_pose_baseline")
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("epochs", 150)
    cfg.setdefault("batch", 8)
    cfg.setdefault("workers", 4)
    cfg.setdefault("patience", 40)
    cfg.setdefault("seed", 42)
    cfg.setdefault("exist_ok", False)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = effective_config(args)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

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
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str((PROJECT_ROOT / train_kwargs["project"]).resolve())

    model = YOLO(model_name)
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", train_kwargs["project"]))
    write_run_metadata(
        run_dir=save_dir,
        project_root=PROJECT_ROOT,
        command=sys.argv,
        config={"model": model_name, **train_kwargs},
    )
    print(f"Training finished. Run folder: {save_dir}")


if __name__ == "__main__":
    main()
