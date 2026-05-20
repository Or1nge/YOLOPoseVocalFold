# Changelog

## Unreleased

- Merged baseline commit `08e70f0` into the containment-loss experiment branch.
- Added reusable keypoint-containment loss functions with pure NumPy tests.
- Added lambda sweep configuration and an experimental training entry with dry-run/smoke validation.
- Documented the current Ultralytics hook boundary so this branch does not claim full training support before version-specific tensor wiring is verified.

## 0.1.0 - 2026-05-20

- Created the `YOLOPoseVocalFold` project on the `main` branch.
- Added LabelMe to YOLO-Pose conversion with bbox/keypoint consistency correction.
- Added standard YOLO-Pose training, prediction postprocessing, dataset validation, and evaluation tools.
- Documented the baseline protocol and the separate containment-loss experiment branch.
