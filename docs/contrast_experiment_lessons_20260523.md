# Contrast experiment lessons, 2026-05-23

This note keeps the lessons from the superseded contrast runs after their raw `Results/` outputs were removed.

## LDP pseudo-positive contrast

The LDP pseudo-positive stage improved LDP holdout mixed-image behavior, but it used model-generated keypoints as positive labels. That creates a feedback risk: if the current ROI model places A/L/R incorrectly, keypoint contrast can make the same wrong local regions more stable rather than adding new anatomical information.

Lesson: do not use LDP pseudo-positive keypoints as the main source for keypoint-local contrast unless a separate noise-control mechanism is added.

## Manual oriented contrast

Manual-only oriented contrast is a cleaner test of local anatomy. It samples A/L/R patches in a canonical anterior-to-posterior frame and avoids pseudo-label feedback. On the 20-image manual test split it showed small positive signals versus the no-contrast stage1 baseline, such as fewer invalid predictions and slightly higher PCK.

However, the manual-only run had weak mixed-image rejection. It used only 60 mixed negatives and a low-weight image-level reject branch, and LDP holdout mixed false positives rose to 6%.

Lesson: oriented A/L/R local contrast helps keypoint representation, but broad mixed-image rejection needs much stronger negative exposure.

## Current replacement design

The replacement run used three stages:

```text
Stage 1: manual A/L/R oriented local contrast pretrain
Stage 2: LDP mixed-image empty-label hard-negative training
Stage 3: standard YOLO-Pose + containment fine-tune on manual labels plus 60 mixed negatives
```

The key change is that LDP contributes only mixed-image negatives in Stage 2. Non-mixed LDP pseudo-positive keypoints are not used.

Deleted result: using all 6,247 available LDP mixed negatives made the ROI model too conservative. It reduced LDP holdout mixed-image false positives to 1%, but manual test postprocessing rejected all 20 images and native pose mAP50 fell from 0.904 to 0.742.

Replacement result: capping Stage 2 to 200 random blackpadded mixed negatives restored localization. Native manual-test pose mAP50 improved to 0.964 and postprocessed usable predictions were 11/20. However, LDP holdout mixed-image false positives rose to 8%, worse than the no-contrast stage1 baseline and V1.1.

Lesson: bulk empty-label mixed negatives cannot dominate manual positive pose supervision, but simply capping negatives is not enough for rejection. A safer follow-up would keep balanced sampling and add a separate reject objective or targeted hard-negative mining.
