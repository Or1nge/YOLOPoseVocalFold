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
        description="Run oriented keypoint contrast, bulk mixed hard-negative training, then final YOLO-Pose fine-tune."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline_three_stage_oriented_mixed_reject_y11m.yaml"),
    )
    parser.add_argument("--device", type=str, help="Override device for all stages.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_checkpoint(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    base = cfg.get("base", cfg)
    project = resolve(base["project"])
    name = base["name"]
    return project / name / "weights" / "best.pt"


def command_for_train(
    config_path: Path,
    *,
    model: Path,
    lambda_containment: float,
    device: str,
    enable_unstable_loss_hook: bool,
) -> list[str]:
    command = [
        sys.executable,
        "tools/train_keypoint_containment.py",
        "--config",
        str(config_path),
        "--model",
        str(model),
        "--lambda-containment",
        str(lambda_containment),
        "--device",
        device,
    ]
    if enable_unstable_loss_hook:
        command.append("--enable-unstable-loss-hook")
    return command


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
    stage2_config = resolve(cfg["stage2_config"])
    stage3_config = resolve(cfg["stage3_config"])
    device = str(args.device if args.device is not None else cfg.get("device", "0"))
    lambda_containment = float(cfg.get("lambda_containment", 0.05))
    enable_hook = bool(cfg.get("enable_unstable_loss_hook", True))
    log_root = PROJECT_ROOT / "Results" / "three_stage_oriented_contrast" / "pipeline_logs"

    pretrain_cmd = [
        sys.executable,
        "tools/pretrain_oriented_contrast.py",
        "--config",
        str(pretrain_config),
        "--device",
        device,
    ]
    pretrain_best = run_checkpoint(pretrain_config)
    stage2_best = run_checkpoint(stage2_config)
    stage2_cmd = command_for_train(
        stage2_config,
        model=pretrain_best,
        lambda_containment=lambda_containment,
        device=device,
        enable_unstable_loss_hook=enable_hook,
    )
    stage3_cmd = command_for_train(
        stage3_config,
        model=stage2_best,
        lambda_containment=lambda_containment,
        device=device,
        enable_unstable_loss_hook=enable_hook,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "pretrain": pretrain_cmd,
                    "stage2_mixed_hard_negative": stage2_cmd,
                    "stage3_final_pose": stage3_cmd,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    run_command(pretrain_cmd, log_root / "stage1_pretrain.log")
    if not pretrain_best.exists():
        raise SystemExit(f"Missing stage 1 checkpoint: {pretrain_best}")
    run_command(stage2_cmd, log_root / "stage2_mixed_hard_negative.log")
    if not stage2_best.exists():
        raise SystemExit(f"Missing stage 2 checkpoint: {stage2_best}")
    run_command(stage3_cmd, log_root / "stage3_final_pose.log")
    print(f"Three-stage pipeline finished. Logs: {log_root}")


if __name__ == "__main__":
    main()
