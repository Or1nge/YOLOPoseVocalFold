# Three-stage oriented contrast rectangular patch evaluation, 2026-05-24

This run repeats the retained three-stage 200-negative workflow, changing Stage 1 local contrast sampling from a square `48x48` input-pixel footprint to keypoint-specific rectangular footprints:

```text
A/anterior: 48 canonical x by 72 canonical y
L/posterior: 72 canonical x by 48 canonical y
R/posterior: 72 canonical x by 48 canonical y
```

Canonical y follows the anterior-to-posterior angle-bisector direction. Canonical x is perpendicular to that bisector. Thus A is longer along the bisector, while L/R are wider perpendicular to it.

## Run artifacts

| Item | Path |
|---|---|
| Pipeline config | `configs/pipeline_three_stage_oriented_mixed_reject_rect48x72_y11m.yaml` |
| Stage 1 pretrain | `Results/three_stage_oriented_contrast/pretrain/yolo11m_manual_only_oriented_kp_contrast_rect48x72` |
| Stage 2 hard-negative train | `Results/three_stage_oriented_contrast/stage2_mixed_hard_negative/yolo11m_stage2_ldp200_blackpad_hardneg_from_oriented_pretrain_rect48x72` |
| Stage 3 final model | `Results/three_stage_oriented_contrast/stage3_final_pose/yolo11m_stage3_manual_pose_after_oriented_and_ldp200_blackpad_rect48x72` |
| Native test eval | `Results/evaluation/three_stage_oriented_mixed_reject_rect48x72_native_test/native_test_metrics.json` |
| Manual postprocess eval | `Results/evaluation/three_stage_oriented_mixed_reject_rect48x72_manual_test/test_summary.json` |
| LDP holdout eval | `Results/evaluation/three_stage_oriented_mixed_reject_rect48x72_ldp_holdout/ldp_holdout_summary.json` |

Stage 1 best validation contrast loss was `0.0208` at epoch 23. Stage 2 ran 40 epochs; best training-CSV validation pose mAP50-95 was `0.9454` at epoch 33. Stage 3 stopped early at epoch 96; best training-CSV validation pose mAP50-95 was `0.9605` at epoch 37, and the best checkpoint validation pose mAP50-95 was `0.959`.

## Manual test: native YOLO metrics

Same 20-image manual test split, `imgsz=960`.

| Model | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|---|---:|---:|---:|---:|
| Three-stage, 48px square patch | 0.886 | 0.362 | 0.964 | 0.910 |
| Three-stage, 96px square patch | 0.886 | 0.362 | 0.964 | 0.910 |
| Three-stage, rect48x72 patch | 0.886 | 0.362 | 0.964 | 0.910 |

Native YOLO-Pose test metrics are unchanged versus the 48px and 96px runs.

## Manual test: ROI postprocessing

Same 20-image manual test split, using `configs/postprocess.yaml`.

| Model | Usable | Rejected | Mean bbox IoU | Mean normalized keypoint error | Mean PCK |
|---|---:|---:|---:|---:|---:|
| Three-stage, 48px square patch | 11/20 | 9/20 | 0.581 | 0.0706 | 0.193 |
| Three-stage, 96px square patch | 8/20 | 12/20 | 0.0867 | 0.4233 | 0.000 |
| Three-stage, rect48x72 patch | 8/20 | 12/20 | 0.0867 | 0.4233 | 0.000 |

The rectangular footprint did not recover the ROI geometry degradation seen with the 96px run.

## LDP holdout

Holdout: `data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl`, 800 images, 100 per class including `混杂图片`.

| Model | Auto accept | Manual review | Reject | Mixed false positive | Mixed auto accept | Final conf mean | no_valid_pose_prediction | low_roi_area | low_roi_dark_fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1.1 retained ROI model | 43.1% | 14.5% | 42.4% | 0.0% | 0.0% | 0.336 | 16.1% | 4.6% | 22.6% |
| Three-stage, 48px square patch | 47.5% | 14.4% | 38.1% | 8.0% | 6.0% | 0.375 | 11.9% | 6.0% | 19.0% |
| Three-stage, 96px square patch | 47.5% | 14.4% | 38.1% | 8.0% | 6.0% | 0.375 | 11.9% | 6.0% | 19.0% |
| Three-stage, rect48x72 patch | 47.5% | 14.4% | 38.1% | 8.0% | 6.0% | 0.375 | 11.9% | 6.0% | 19.0% |

The LDP holdout profile is effectively unchanged. Mixed-image false positives remain 8%, worse than the retained V1.1 ROI model.

## Conclusion

The keypoint-specific rectangular footprint changes Stage 1 contrast learning and produces an independent checkpoint, but it does not improve native test metrics, ROI postprocessing, or LDP mixed-image rejection. This run should not replace V1.1 or the 48px experimental checkpoint.
