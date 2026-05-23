#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual oriented keypoint contrast pretraining, then YOLO-Pose training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline_manual_oriented_contrast_then_pose_y11m.yaml"),
    )
    parser.add_argument("--device", type=str, help="Override GPU/CPU device for both stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pretrain_checkpoint(pretrain_config: Path) -> Path:
    cfg = load_yaml(pretrain_config)
    project = resolve(cfg.get("project", "Results/oriented_keypoint_contrast_pretrain"))
    name = cfg.get("name", "yolo11m_manual_oriented_kp_contrast")
    return project / name / "weights" / "best.pt"


def run_command(command: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    pretrain_config = resolve(cfg["pretrain_config"])
    final_train_config = resolve(cfg["final_train_config"])
    device = str(args.device if args.device is not None else cfg.get("device", "0"))
    pretrain_log = PROJECT_ROOT / "Results" / "manual_oriented_contrast_pipeline" / "pretrain.log"
    final_log = PROJECT_ROOT / "Results" / "manual_oriented_contrast_pipeline" / "final_train.log"

    pretrain_cmd = [
        sys.executable,
        "tools/pretrain_oriented_contrast.py",
        "--config",
        str(pretrain_config),
        "--device",
        device,
    ]
    best_checkpoint = pretrain_checkpoint(pretrain_config)
    train_cmd = [
        sys.executable,
        "tools/train_keypoint_containment.py",
        "--config",
        str(final_train_config),
        "--model",
        str(best_checkpoint),
        "--lambda-containment",
        str(cfg.get("lambda_containment", 0.05)),
        "--device",
        device,
    ]
    if bool(cfg.get("enable_unstable_loss_hook", True)):
        train_cmd.append("--enable-unstable-loss-hook")

    if args.dry_run:
        print(json.dumps({"pretrain": pretrain_cmd, "final_train": train_cmd}, indent=2, ensure_ascii=False))
        return

    run_command(pretrain_cmd, pretrain_log)
    if not best_checkpoint.exists():
        raise SystemExit(f"Pretrain finished but best checkpoint is missing: {best_checkpoint}")
    run_command(train_cmd, final_log)
    print(f"Pipeline finished. Logs: {pretrain_log}, {final_log}")


if __name__ == "__main__":
    main()
