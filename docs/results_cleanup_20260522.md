# Results Cleanup Notes - 2026-05-22

## Background

`Results/` had accumulated several short-lived experiment families: early three-point YOLO-Pose baselines, containment-loss smoke runs, geometry tuning, 200-image LDP probes, V1/V1.1 black-border inference, ROI crop exports, square224 exports, and external-folder checks. The cleanup keeps the final V1.1 ROI/data-generation version and records why the older artifacts were removed.

The retained ROI pipeline is:

```text
original image
-> V1.1 black-border input for ROI inference
-> 3 keypoints + angle-bisector rotated final_box_polygon
-> polygon ROI crop when possible
-> fallback to original_source when crop fails
-> resize every output to 224x224 square
```

## Retained Version

Retained ROI model alias:

- `Results/models/vf_roi_v1/`
- `best.pt` points to `Results/containment_loss/yolo_pose_glottic_three_point_y11m_img960_pose24_containment_l0p05_ldp_pseudo_mixedpenalty_copy/weights/best.pt`
- `postprocess.yaml` points to the current V1.1 postprocessing config.

Retained source training run:

- `Results/containment_loss/yolo_pose_glottic_three_point_y11m_img960_pose24_containment_l0p05_ldp_pseudo_mixedpenalty_copy/`
- Created at `2026-05-22T09:40:30+08:00`.
- Command: `tools/train_keypoint_containment.py --config configs/train_containment_ldp_pseudo_mixedpenalty_copy_y11m.yaml --lambda-containment 0.05 --enable-unstable-loss-hook`.
- Training completed 60 epochs and includes `args.yaml`, `results.csv`, `run_metadata.json`, `weights/best.pt`, and `weights/last.pt`.

Retained final LDP V1.1 square224 data output:

- `Results/ldp_8class_v1_1_nonvoc_rule_20260522_201632/`
- `prediction_action_summary.json`: full LDP action summary.
- `predictions.jsonl`: V1.1 ROI predictions with black-border preprocessing and `original_source`.
- `crop_square224_all_actions.log`: command summary for final square export.
- `cropped_square224_box_or_unpad_original_all_actions/`: full square224 crop-or-original export.
- `subset_square224_all_voc_boxes_all_nonvoc/`: final classification-training input used by the downstream completed run.

The final training input subset contains `13612` images. It keeps all `混杂图片` and keeps non-`混杂图片` classes only when ROI crop succeeded. `subset_summary.json` reported `bad_size_examples: []`, so every retained training image is `224x224`.

The downstream completed classifier run lives outside this repo:

```text
/home/or1ngelinux/CVProjects/Larynx/laryngeal_4class/Results/v1_1_square224_all_voc_boxes_all_nonvoc_swin_base224_stage2stage3_1block_two_stage_20260522_215431
```

That run consumed:

```text
Results/ldp_8class_v1_1_nonvoc_rule_20260522_201632/subset_square224_all_voc_boxes_all_nonvoc
```

## Removed Result Groups

Removed old or superseded ROI/data-generation artifacts:

- Early baseline and geometry output: `angle_bisector_roi_pilot/`, `glottic_three_point/`, `geometry_tuning/`, `evaluation/`, `predictions/`.
- Containment smoke and old training runs that were not the final LDP pseudo mixed-penalty copy run.
- 200-image LDP probes used to tune thresholds, TTA, confidence curves, dark-region gates, orientation checks, and mixed-negative behavior.
- V1 full LDP/external exports without integrated V1.1 black-border preprocessing.
- V1 blackpad exports where black-border images were materialized as a separate input tree and could leak into fallback/copy-original outputs.
- Non-square or non-final crop exports inside the retained V1.1 directories, including large blackpad input copies and old auto/manual/non-vocal-rule crop trees.

Removed stale `latest_*` pointer files that referenced deleted result directories.

## Lessons

- Black-border enhancement belongs inside `tools/predict_roi.py`, not as an opaque preprocessed image tree. Integrated V1.1 inference preserves `preprocess` metadata and `original_source`.
- Black-border images are only for ROI localization. Downstream fallback images should come from `original_source`, so the classifier does not learn artificial borders.
- Final downstream data should be uniformly square. Mixed-size crop/original exports are useful for review but not as the final training input.
- `混杂图片` negatives and hard negatives are important; earlier models without mixed-negative pressure produced too many usable boxes on non-vocal-fold images.
- Do not gate anterior/posterior points by a fixed image-y ordering. Patient position and camera angle can flip the visual orientation.
- The 87% ROI target was more useful than older oversized coverage targets for downstream classification.
- Keep source LDP data read-only. Use local derived datasets under `data/` or `Results/` for training and inference products.

## Future Result Hygiene

- Keep a small model alias such as `Results/models/vf_roi_v1/` for the current recommended model.
- Keep only the final training run, final predictions, final square export, final subset, and summary/manifest files.
- Name future output directories with algorithm version, source dataset, whether `original_source` fallback was used, square size, and date.
- Prefer deleting large regenerated materialized inputs, overlays, and superseded crop trees once summaries and manifests have been saved.
