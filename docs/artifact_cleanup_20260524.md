# Artifact Cleanup Notes - 2026-05-24

## Retained Data

The local checkout now treats these as the important data surfaces:

- `data/glottic_roi_rectangle_annotation_200_20260520_blackpad/`: original blackpad LabelMe annotation bundle.
- `data/splits/glottic_roi_rectangle_annotation_200_20260520_blackpad/`: split mirror used to recreate the exact train/val/test assignment.
- `data/yolo_pose/`: converted YOLO-Pose GT train/val/test set.
- `data/yolo_pose_mixed_negative_120_blackpad/`: current GT split plus 120 holdout-excluded LDP mixed-image empty-label negatives with blackpad preprocessing.
- `data/yolo_pose_mixed_negative_60/`: GT split plus 60 LDP mixed-image empty-label negatives, retained for the older no-DINO baseline run.
- `data/yolo_pose_mixed_negative_60_blackpad/`: blackpadded 60-negative version used by the latest V1.2/DINO auxiliary pose run.
- `data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded/`: 200 blackpadded LDP mixed negatives, excluding the then-fixed holdout, used for DINO auxiliary hard-negative training.
- `data/yolo_pose_ldp_pseudo_mixedpenalty_copy/`: exact derived training copy for the retained V1.1 LDP-pseudo containment ROI model.
- `data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl`: fixed 800-image LDP holdout manifest retained at cleanup time; it was later replaced by the refreshed 2026-05-25 holdout documented in `docs/ldp_holdout_refresh_20260525.md`.

`data/yolo_pose_stage2_ldp_conf040_holdout100/` was removed. It belonged to the older LDP pseudo-positive contrast branch and is no longer part of the recommended V1.2/DINO auxiliary direction.

## Retained Weights

Stable aliases live under `Results/models/`:

- `Results/models/vf_roi_v1/best.pt`: retained V1.1/V1.2 recommended YOLO-Pose ROI model.
- `Results/models/vf_roi_v12_dinov3_aux/pose_best.pt`: latest V1.2 manual + blackpad mixed-negative YOLO-Pose checkpoint from 2026-05-24.
- `Results/models/vf_roi_v12_dinov3_aux/aux_best.pt`: latest DINOv3 point-region auxiliary head from 2026-05-24.
- `Results/models/vf_roi_no_dino_mixedneg60/best.pt`: retained no-DINO mixed-negative YOLO-Pose checkpoint.

For retained runs, `best` checkpoints are kept and `last` checkpoint duplicates are removed.

## Removed Result Groups

Removed or pruned artifacts:

- Superseded three-stage oriented contrast result trees. Their conclusions remain in `docs/three_stage_oriented_mixed_reject_*` and `docs/contrast_experiment_lessons_20260523.md`.
- Superseded DINO auxiliary heads and the older non-v12 DINO full-pipeline output.
- Large evaluation folders containing regenerated blackpad input images or superseded ablations. The compact latest DINO reward-only evaluation and V1.1 smoke summaries are kept.
- Review/visual-check image grids that can be regenerated from retained weights and manifests.

The goal is to keep the source data, fixed evaluation holdout, current best checkpoints, and enough run metadata to understand how the retained models were produced, while removing large derived experiment payloads that are no longer the active path.
