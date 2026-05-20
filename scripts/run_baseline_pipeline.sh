#!/usr/bin/env bash
set -euo pipefail

python tools/convert_labelme_to_yolo_pose.py \
  --labelme-dir data/labelme \
  --image-dir data/images \
  --out-dir data/yolo_pose

python tools/validate_dataset.py --dataset-dir data/yolo_pose
python tools/train_yolo_pose.py --config configs/train_baseline.yaml

