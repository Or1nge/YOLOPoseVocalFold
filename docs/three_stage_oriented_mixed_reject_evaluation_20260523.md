# Three-stage oriented contrast evaluation, 2026-05-23

This run tested the revised three-stage workflow after reducing Stage 2 mixed-image negatives from all available LDP mixed images to a random 200-image subset. Added mixed negatives were materialized with the same V1.1 black-border preprocessing used at inference time.

```text
Stage 1: manual A/L/R oriented local contrast pretrain
Stage 2: manual positives + 200 blackpadded LDP mixed-image empty-label hard negatives
Stage 3: manual positives + 60 blackpadded mixed-image negatives, YOLO-Pose + containment fine-tune
```

## Run artifacts

| Item | Path |
|---|---|
| Pipeline config | `configs/pipeline_three_stage_oriented_mixed_reject_y11m.yaml` |
| Stage 2 data | `data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded` |
| Stage 3 data | `data/yolo_pose_mixed_negative_60_blackpad` |
| Stage 1 pretrain | `Results/three_stage_oriented_contrast/pretrain/yolo11m_manual_only_oriented_kp_contrast_48px` |
| Stage 2 hard-negative train | `Results/three_stage_oriented_contrast/stage2_mixed_hard_negative/yolo11m_stage2_ldp200_blackpad_hardneg_from_oriented_pretrain` |
| Stage 3 final model | `Results/three_stage_oriented_contrast/stage3_final_pose/yolo11m_stage3_manual_pose_after_oriented_and_ldp200_blackpad` |
| Native test eval | `Results/evaluation/three_stage_oriented_mixed_reject_native_test/native_test_metrics.json` |
| Manual postprocess eval | `Results/evaluation/three_stage_oriented_mixed_reject_manual_test/test_summary.json` |
| LDP holdout eval | `Results/evaluation/three_stage_oriented_mixed_reject_ldp_holdout/ldp_holdout_summary.json` |

Stage 1 best validation contrast loss was `0.0283` at epoch 23. Stage 2 ran 40 epochs; best validation pose mAP50 was `0.9696`. Stage 3 stopped early at epoch 96; best validation pose mAP50 was `0.9854`, and the final best checkpoint validation pose mAP50-95 was `0.959`.

## Manual test: native YOLO metrics

Same 20-image manual test split, `imgsz=960`.

| Model | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|---|---:|---:|---:|---:|
| No-contrast stage1 baseline | 0.874 | 0.294 | 0.904 | 0.856 |
| Three-stage, 200 blackpad negatives | 0.886 | 0.362 | 0.964 | 0.910 |

Native YOLO-Pose test accuracy improved versus the no-contrast baseline.

## Manual test: ROI postprocessing

Same 20-image manual test split, using `configs/postprocess.yaml`.

| Model | Usable | Rejected | Mean bbox IoU | Mean normalized keypoint error | Mean PCK | Mean final confidence |
|---|---:|---:|---:|---:|---:|---:|
| No-contrast stage1 baseline | 10/20 | 10/20 | 0.581 | 0.0390 | 0.333 | 0.412 |
| Three-stage, 200 blackpad negatives | 11/20 | 9/20 | 0.581 | 0.0706 | 0.193 | 0.404 |

Postprocessed usability slightly improved, but the keypoint-error metrics worsened. This suggests the native mAP gain does not fully translate into the current ROI geometry score.

## LDP holdout

Holdout: `data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl`, 800 images, 100 per class including `混杂图片`.

| Model | Auto accept | Manual review | Reject | Mixed false positive | Mixed auto accept | Final conf mean | no_valid_pose_prediction | low_roi_area | low_roi_dark_fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1.1 retained ROI model | 43.1% | 14.5% | 42.4% | 0.0% | 0.0% | 0.336 | 16.1% | 4.6% | 22.6% |
| No-contrast stage1 baseline | 48.4% | 13.3% | 38.4% | 2.0% | 1.0% | 0.369 | 14.0% | 7.1% | 23.6% |
| Three-stage, 200 blackpad negatives | 47.5% | 14.4% | 38.1% | 8.0% | 6.0% | 0.375 | 11.9% | 6.0% | 19.0% |

The 200-negative run avoids the over-rejection failure from the deleted 6k-negative run and improves native pose metrics, but it does not improve mixed-image rejection. On LDP holdout, mixed false positives rise to 8%, worse than both the no-contrast stage1 baseline and the retained V1.1 ROI model.

## Conclusion

This version is useful as evidence that the negative ratio was the main cause of the deleted 6k-negative failure. However, it is not yet a better ROI model than V1.1 because mixed-image rejection worsened.

Next experiments should keep the 200-negative scale or use balanced sampling, but add a separate reject objective or harder mixed-negative mining that does not degrade keypoint/ROI geometry.
