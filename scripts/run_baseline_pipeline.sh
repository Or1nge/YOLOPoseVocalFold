#!/usr/bin/env bash
set -euo pipefail

python tools/convert_labelme_to_yolo_pose.py \
  --labelme-dir data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images \
  --image-dir data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images \
  --out-dir data/yolo_pose \
  --split-source-dir data/splits/glottic_roi_rectangle_annotation_200_20260520_blackpad

python tools/validate_dataset.py --dataset-dir data/yolo_pose
python tools/tune_geometry_roi.py \
  --dataset-dir data/yolo_pose \
  --split train \
  --postprocess-out Results/geometry_tuning/glottic_three_point/postprocess_tuned.yaml
python tools/train_yolo_pose.py --config configs/train_baseline.yaml
