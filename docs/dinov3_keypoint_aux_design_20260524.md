# DINOv3 three-point auxiliary judgement, 2026-05-24

This branch replaces the keypoint-local contrast experiment with a frozen-DINOv3
auxiliary scorer. The goal is not to make left/right posterior commissures look
similar. Diseased left and right sides may differ strongly, so L/R are never
used as each other's positive samples.

## Design

The auxiliary scorer uses DINOv3 as a dense feature encoder and trains only a
small head on top:

```text
image -> frozen DINOv3 dense features
      -> point head: background / anterior / left posterior / right posterior
      -> triplet head: ordered A/L/R triplet plausible or corrupted
      -> image head: usable vocal-fold image or mixed/reject image
```

Positive point labels come only from manual YOLO-Pose labels. Background points
are random locations away from visible keypoints. Triplet negatives are
synthetic corruptions of the same image's A/L/R triplet, such as point swaps,
jitter, or replacing one point with a random location. This tests whether the
three predicted points are anatomically coherent without contrastively pulling
L/R together.

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
  --aux-checkpoint Results/dinov3_keypoint_aux/dinov3_vits16_three_point_aux_448_mixedneg60/weights/best_aux_head.pt \
  --predictions Results/predictions/predictions.jsonl \
  --out Results/predictions/predictions_dinov3_aux.jsonl
```

To let the auxiliary score reduce `final_confidence`, add
`--apply-confidence-gate`. The default is score-only so the first run can be
audited without changing existing ROI decisions.

## Boundary

This is an auxiliary judgement layer, not a new ROI geometry definition. The
standard YOLO-Pose model still predicts A/L/R and the existing geometric
postprocess still creates the final rotated ROI. DINOv3 only adds semantic
evidence around the predicted points and the ordered three-point structure.
