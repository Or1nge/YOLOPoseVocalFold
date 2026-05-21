# Changelog

## 0.1.1 - 2026-05-21

- Updated the baseline annotation contract from 4 generic keypoints to 3 glottic keypoints.
- Added duplicate-aware conversion metadata, ROI polygon evaluation, and geometry tuning.
- Added flexible YOLO-Pose label parsing for variable keypoint counts.
- Renamed the baseline training run output for the glottic three-point setup.
- Kept local data directories ignored before publishing the private GitHub remote.

## 0.1.0 - 2026-05-20

- Created the `YOLOPoseVocalFold` project on the `main` branch.
- Added LabelMe to YOLO-Pose conversion with bbox/keypoint consistency correction.
- Added standard YOLO-Pose training, prediction postprocessing, dataset validation, and evaluation tools.
- Documented the baseline protocol and the separate containment-loss experiment branch.
