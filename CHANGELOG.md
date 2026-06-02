# Changelog

## Unreleased

- Changed downstream ROI crop export so fallback/retained images default to
  `preprocessed_source`: use the saved no-black/cropped image
  (`cropped_source`/`dinov3_source`) when available, then fall back to
  `original_source`, and only use the black-padded `source` if no cleaner input
  exists. This keeps screen-photo pre-crops in the classifier path when ROI
  geometry is rejected or fails.
- Treat `dinov3_keypoints_outside_cropped_image` as an unusable ROI signal
  during DINOv3 gating: predictions with keypoints outside the no-black/cropped
  image are downgraded to `reject_or_relabel` instead of being auto-cropped.
- Replaced the edge white-corner special case with a general edge-artifact crop
  in the default ROI preprocessing path. The new detector triggers from large
  near-black regions connected to the image edge, then removes edge-connected
  black border/background and neutral bright corner/UI pixels before the usual
  no-black crop + blackpad step. Prediction JSONL metadata now records the
  edge-artifact crop box and trigger diagnostics.
- Promoted the DINOv3-assisted ROI stack to the project main model via the
  local `Results/models/vf_roi_current/` alias and the new
  `tools/predict_current_roi.py` wrapper, which runs YOLO-Pose first and then
  applies DINOv3 auxiliary confidence gating.
- Batched the default non-TTA ROI prediction path in `tools/predict_roi.py`:
  `--batch` now sends prepared images to YOLO in GPU batches, while
  `--preprocess-workers` can parallelize screen-photo pre-crop, black-border
  crop, and blackpad generation. Intermediate pre-crop/no-black/blackpad images
  now stay in memory by default and are only written when `--save-intermediates`
  is passed or explicit intermediate directories are provided. Single-image
  `--source` still uses the same entrypoint; `--tta` remains the slower
  per-image aggregation path.
- Added screen-photo pre-crop to the ROI prediction pipeline: `tools/predict_roi.py` now classifies each image as a clean frame or a messy phone-photo-of-screen (via fixed-color stripe and blue-region signals), and when triggered, crops out the laryngoscope window before the existing black-border crop + blackpad step. The reusable detection/cropping logic lives in `yoloposevf/screen_photo_crop.py`; the `scripts/crop_527_xianlin.py` standalone pre-cropper imports from it. JSONL outputs carry a new `pre_crop` audit section, mirrored under `preprocess.pre_crop`, while preserving the existing `source`/`original_source`/`cropped_source`/`dinov3_source` contract; `tools/crop_rois_from_predictions.py` can now use `--copy-original-source cropped_source` for screen-photo fallback images.

- Made `tools/predict_roi.py` default to GPU 0 for YOLO-Pose inference; pass
  `--device cpu` only when CPU inference is explicitly needed.
- Fixed manual split ROI evaluation for `predict_roi.py` outputs by mapping GT
  bbox/keypoints/manual ROI polygons through the saved crop+blackpad preprocess
  metadata before computing metrics.
- Trained and evaluated the 120 blackpad mixed-negative containment run; saved
  `best.pt` under `Results/containment_loss/yolo_pose_glottic_three_point_y11m_img960_pose24_containment_l0p05_mixedneg120_blackpad/weights/`
  and documented the validation/holdout tradeoff in
  `docs/mixedneg120_blackpad_training_20260525.md`.
- Rebuilt the LabelMe-derived YOLO-Pose GT split from 396 valid blackpad samples (`277/79/40`) and added `data/yolo_pose_mixed_negative_120_blackpad` with 120 holdout-excluded LDP mixed-image negatives plus a matching containment training config.
- Cleaned local data/results artifacts: retained the source LabelMe bundle, GT splits, fixed LDP holdout, current V1.1/V1.2 and DINO/no-DINO model aliases, and removed superseded contrast/DINO/evaluation payloads documented in `docs/artifact_cleanup_20260524.md`.
- Changed ROI prediction preprocessing to required `V1.2`: crop existing black borders first, then always add the uniform black border before YOLO-Pose; prediction JSONL now records the shared preprocess metadata contract including `crop_bbox_xyxy`, `crop_was_applied`, `cropped_source`, `padding_fraction`, `model_input_width/height`, and `no_black_bbox_in_model_input`.
- Updated DINOv3 auxiliary scoring to read no-black/cropped images before padded `source`, transform YOLO padded keypoints back into cropped coordinates, and keep the auxiliary score available for final confidence/action gating.
- Changed DINOv3 valid-mask sampling so out-of-image oriented patch locations are not clamped to edge pixels; incomplete 48x48 point patches now carry invalid mask cells and zeroed features.
- Started `exp/dinov3-keypoint-aux` as a replacement direction for keypoint-local contrast.
- Narrowed the frozen-DINOv3 auxiliary scorer to oriented point-region `background/A/L/R` classification with mined hard negatives and a reward-only confidence gate: `0.30-0.60` linearly rewards up to `1.5x`, very high scores can directly pass, and low DINO scores no longer directly reject.
- Added valid-mask support to DINOv3 oriented point-region patches so invalid local cells are zeroed and exposed to the point head as a mask channel.
- Added training and prediction-scoring entrypoints: `tools/train_dinov3_keypoint_aux.py` and `tools/score_predictions_with_dinov3_aux.py`.
- Added DINOv3 auxiliary configs, design notes, and unit tests.
- Completed the three-stage oriented contrast run with 200 random blackpadded
  LDP mixed-image negatives and documented it in
  `docs/three_stage_oriented_mixed_reject_evaluation_20260523.md`: native
  manual-test pose mAP50 improved versus the no-contrast baseline, but LDP
  mixed false positives rose to 8%, so it is not recommended as the current
  ROI model.
- Replaced the previous contrast run plan with a three-stage recipe:
  manual-only oriented A/L/R contrast pretraining, bulk LDP mixed-image
  empty-label hard-negative training, and final YOLO-Pose/containment fine-tune.
- Extended `tools/add_negative_images_to_yolo_pose.py` with `--all`,
  `--exclude-manifest`, and `--blackpad-negatives`, then built blackpadded
  mixed-negative datasets for the current three-stage run while excluding the
  fixed LDP holdout.
- Removed superseded contrast `Results/` outputs and kept their lessons in
  `docs/contrast_experiment_lessons_20260523.md`.
- Reworked the contrast experiment into manual-label oriented pretraining:
  `tools/pretrain_oriented_contrast.py` samples anatomy-aligned A/L/R local
  patches and feeds the resulting checkpoint into later YOLO-Pose training.
- Added LDP holdout manifest inference/evaluation support for the two-stage contrast experiment:
  `tools/predict_roi.py --manifest` and `tools/summarize_ldp_holdout_predictions.py`.
- Started the `exp/keypoint-local-contrast` experiment branch with a keypoint-level local contrast loss, synchronized second-view augmentation, projection head, LDP pseudo hard-negative recipe, and unit tests.
- Added a two-stage ROI/contrast experiment layout: stage 1 trains on manual labels plus 60 mixed negatives, stage 2 fine-tunes with LDP pseudo labels requiring `final_confidence > 0.4`, and an 800-image LDP holdout is excluded from training.
- Cleaned `Results/` to retain only the final 2026-05-22 V1.1 square224 ROI/data-generation version, its source ROI training run, and the `vf_roi_v1` model alias; documented removed experiment groups in `docs/results_cleanup_20260522.md`.
- Promoted ROI localization to `V1.1`: `tools/predict_roi.py` now applies black-border pre-enhancement to every input image by default before YOLO-Pose inference and postprocessing.
- Changed directory/list ROI prediction to stream one image at a time so full LDP inference does not batch thousands of paths into GPU memory.
- Added `--copy-original-classes` to ROI crop export, allowing rules such as cropping accepted/review images while keeping only `混杂图片` rejects as originals and skipping other rejects.
- Added square-output ROI crop export with `--output-size` and `--copy-original-source original_source`, so classifier datasets can use cropped boxes when present and unpadded originals when no box is found.
- Fixed external 4-class checkpoint evaluation to apply the same ImageNet normalization used by the training/test evaluation path.
- Added the short `vf_roi_v1` alias for the current recommended LDP-pseudo containment ROI model.
- Added `--copy-original-actions` and `--fallback-original-on-crop-failure` to ROI crop export so rejected predictions can be retained as original images for downstream classification datasets.
- Added `tools/blackpad_image_tree.py` to copy image folders while adding the black border used by the V1 ROI training setup before prediction and crop export.
- Added `tools/evaluate_four_class_checkpoint.py` for evaluating a laryngeal 4-class checkpoint on folder-structured external datasets.
- Merged the former `exp/keypoint-containment-loss` workflow into `main`; future containment-loss and LDP pseudo fine-tuning work should continue on `main`.
- Added configurable confidence sharpening via `confidence_gamma` for postprocessed ROI decisions.
- Added a configurable `tanh` confidence curve for postprocessed ROI decisions while keeping the previous power/gamma curve available.
- Added ROI-area and keypoint image-boundary confidence gates to reduce usable boxes on obvious non-vocal-fold LDP images.
- Removed the anterior-vs-posterior image-y ordering gate; anterior/posterior keypoints are no longer rejected solely because patient position flips their vertical order.
- Added a configurable predicted-ROI dark-region gate to downgrade bright/highlight-only regions without using manual annotations, now defaulting to exposure-relative foreground median thresholding so synthetic black borders do not dominate the threshold.
- Changed ROI relative-area scoring to divide by the non-black foreground bounding-rectangle area, so large input black borders no longer make clear vocal-fold ROIs look artificially too small.
- Softened the three-point glottic angle gate so angles outside `20°-130°` are penalized instead of forcing `geometry_score=0`.
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
