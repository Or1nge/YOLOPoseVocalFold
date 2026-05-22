# Changelog

## Unreleased

- Added the short `vf_roi_v1` alias for the current recommended LDP-pseudo containment ROI model.
- Merged the former `exp/keypoint-containment-loss` workflow into `main`; future containment-loss and LDP pseudo fine-tuning work should continue on `main`.
- Added configurable confidence sharpening via `confidence_gamma` for postprocessed ROI decisions.
- Added a configurable `tanh` confidence curve for postprocessed ROI decisions while keeping the previous power/gamma curve available.
- Added ROI-area and keypoint image-boundary confidence gates to reduce usable boxes on obvious non-vocal-fold LDP images.
- Added an anterior-vs-posterior point orientation gate for inverted three-point predictions.
- Added a configurable predicted-ROI dark-region gate to downgrade bright/highlight-only regions without using manual annotations.
- Added tooling and a YOLO11m containment recipe for training with 60 empty-label `混杂图片` negative samples.
- Added an LDP pseudo-label fine-tuning recipe that copies accepted/review non-`混杂图片` predictions as YOLO-Pose positives, uses all `混杂图片` as empty-label negatives, and repeats mixed false accepts as hard negatives.
- Updated overlay rendering so rejected predictions are drawn in red and predictions can be rendered directly from their `source` paths outside a YOLO dataset split.
- Added `tools/crop_rois_from_predictions.py` to preserve class-folder structure while cropping accepted rotated ROI predictions.

## 0.1.1 - 2026-05-21

- Added a keypoint-precision containment training recipe using YOLO11m-pose, `imgsz=960`, no flips, no mosaic/erasing, and higher pose-loss weight.
- Added rotation/scale-only TTA inference support for more stable keypoint predictions without changing left/right point semantics.
- Updated the baseline annotation contract from 4 generic keypoints to 3 glottic keypoints.
- Added angle-bisector rotated ROI postprocessing, confidence-gated `usable_bbox`, duplicate-aware conversion metadata, ROI polygon evaluation, and geometry tuning.
- Added prediction overlay rendering for quick visual review of manual ROI, predicted ROI, keypoints, and confidence.
- Changed the glottic ROI tuning target from 95% to 87% containment to reduce oversized boxes.
- Changed the final usable ROI from an axis-aligned outer bbox to a four-point rotated `final_box_polygon`; the legacy xyxy bbox remains only as a compatibility envelope.
- Changed the angle-bisector ROI width from symmetric half-widths to separate left/right lateral extents so the final rotated box is not forced to be centered on the bisector.
- Reduced the default asymmetric 87% ROI side margin from `1.00` to `0.70` after retuning the new geometry for less lateral overreach.
- Added a glottic three-point angle plausibility gate: angles below `20°` or above `130°` now force low geometry confidence, with `20°-35°` treated as a low-confidence transition band.
- Aligned the containment-loss sweep training augmentation defaults with the main baseline recipe.
- Added flexible YOLO-Pose label parsing for variable keypoint counts.
- Renamed the baseline training run output for the glottic three-point setup.
- Kept local data directories ignored before publishing the private GitHub remote.

## 0.1.0 - 2026-05-20

- Created the `YOLOPoseVocalFold` project on the `main` branch.
- Added LabelMe to YOLO-Pose conversion with bbox/keypoint consistency correction.
- Added standard YOLO-Pose training, prediction postprocessing, dataset validation, and evaluation tools.
- Documented the baseline protocol and the separate containment-loss experiment branch.
