# DINOv3 point-region auxiliary judgement, 2026-05-24

This branch replaces the keypoint-local contrast experiment with a frozen-DINOv3
auxiliary scorer. The current version is intentionally point-only: it judges
whether the oriented local region around one candidate point looks like
background, anterior commissure, left posterior midpoint, or right posterior
midpoint. It no longer trains triplet plausibility or image reject heads.

## Design

The auxiliary scorer uses DINOv3 as a dense feature encoder and trains only a
small point head on top:

```text
image -> frozen DINOv3 dense features
      -> oriented patch around candidate point
      -> foreground-valid mask for the same oriented patch
      -> point head: background / anterior / left posterior / right posterior
```

Positive point labels come only from manual YOLO-Pose labels. Background points
come from three sources: random locations away from visible keypoints,
near-miss locations around annotated keypoints, and mined high-confidence
YOLO-Pose candidate points on empty-label mixed images. L/R are separate
classes and are not treated as each other's positive samples.

The current implementation does not crop the full DINO input before encoding.
It uses the image path supplied by the training dataset or prediction JSONL,
then letterboxes that image to the DINO input size. At inference, the scorer
chooses `source` before `original_source`, so LDP holdout predictions scored
from V1.1 ROI outputs use the generated blackpad input image.

To reduce local border contamination without changing the full-image DINO
input, the point head is mask-aware. A foreground-valid mask is built from the
same letterboxed DINO input using a low luma floor. The oriented 48x48 point
region samples both DINO features and the mask; invalid black-border locations
are zeroed in the local feature patch, and the mask itself is appended as an
extra local channel for the head.

## Commands

The default config uses the public `timm/vit_small_patch16_dinov3.lvd1689m`
checkpoint, which still resolves through the Hugging Face cache. The official
`facebook/dinov3-vits16-pretrain-lvd1689m` repository is gated; after access is
approved, `dinov3.backend` can be switched to `transformers` with that official
model id.

Dry run:

```bash
python tools/train_dinov3_keypoint_aux.py \
  --config configs/train_dinov3_keypoint_aux_y11m.yaml \
  --dry-run
```

Train the auxiliary head:

```bash
python tools/train_dinov3_keypoint_aux.py \
  --config configs/train_dinov3_keypoint_aux_y11m.yaml
```

Attach scores to an existing prediction JSONL:

```bash
python tools/score_predictions_with_dinov3_aux.py \
  --aux-checkpoint Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200/weights/best_aux_head.pt \
  --predictions Results/predictions/predictions.jsonl \
  --out Results/predictions/predictions_dinov3_aux.jsonl
```

To let the auxiliary score modify `final_confidence`, add
`--apply-confidence-gate`. The default is score-only so the first run can be
audited without changing existing ROI decisions.

The active gate only uses DINOv3 as positive evidence:

```text
point_region_score < 0.30: confidence_factor = 1.0
0.30 <= point_region_score < 0.60: confidence_factor rises linearly from 1.0 to 1.5
point_region_score >= 0.60: direct auto-accept candidate
```

## Boundary

This is an auxiliary judgement layer, not a new ROI geometry definition. The
standard YOLO-Pose model still predicts A/L/R and the existing geometric
postprocess still creates the final rotated ROI. DINOv3 only adds semantic
evidence around the predicted point regions.
