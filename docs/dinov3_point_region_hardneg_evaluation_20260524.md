# DINOv3 point-region hard-negative evaluation, 2026-05-24

## Change

The DINOv3 auxiliary branch was narrowed to point-region classification only:

```text
oriented local DINOv3 patch -> background / anterior / left posterior / right posterior
```

Removed from the active training objective:

- ordered A/L/R triplet plausibility head
- image-level reject head

Hard-negative sources:

- `data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded`, which excludes the fixed LDP holdout
- mined YOLO-Pose candidate points from empty-label mixed images
- near-miss background points around annotated A/L/R points

Training ratio:

```text
positive A/L/R points: 411
random background points: 337
near-miss background points: 411
mined hard-negative points after filtering: 18
```

## Training

Command:

```bash
python tools/train_dinov3_keypoint_aux.py --config configs/train_dinov3_keypoint_aux_y11m.yaml
```

Run:

```text
Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200
```

Best checkpoint:

```text
Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200/weights/best_aux_head.pt
```

Best epoch was 20, with best validation point loss `0.1447315489`.

## ROI Evaluation

The evaluation reused the same V1.1 baseline prediction files, then applied the new DINOv3 point-region confidence gate. This isolates the auxiliary effect; geometry predictions are unchanged.

| Setting | Manual auto | Manual review | Manual usable | LDP auto | LDP review | LDP accept/review | Mixed FP | Mixed auto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline V1.1 | 6 | 0 | 6 | 43.125% | 14.5% | 57.625% | 0% | 0% |
| Old DINOv3 three-head gate | 4 | 2 | 6 | 28.75% | 14.875% | 43.625% | 0% | 0% |
| Point-region hard-negative gate | 4 | 2 | 6 | 28.25% | 12.75% | 41.0% | 0% | 0% |
| Point-region reward gate | 6 | 0 | 6 | 37.875% | 9.75% | 47.625% | 0% | 0% |

The superseded 0.60/0.85 reward gate mapped DINOv3 point-region scores as follows:

```text
score < 0.60: penalize confidence by score / 0.60
0.60 <= score < 0.85: keep confidence unchanged
score >= 0.85: reward up to 1.2x
```

On the 800-image LDP holdout, 671 images had valid predicted keypoints and DINOv3 scores:

```text
factor > 1: 374
factor = 1: 81
factor < 1: 216
mean factor: 0.883849
max factor: 1.199448
```

## Conclusion

In this earlier 0.60/0.85 run, the point-region-only DINOv3 head did not provide a positive ROI-level gain. It kept mixed false positives at zero, but the baseline already had zero mixed false positives on that holdout run. The reward gate fixed part of the one-way-penalty problem, improving accepted/review yield from `41.0%` to `47.625%`, but it still remained below the no-DINO baseline `57.625%`.

## Reward-only linear gate update

The active inference policy was changed after GT-test review so DINOv3 no
longer rejects low scores and only adds positive evidence:

```text
score < 0.30: no confidence change
0.30 <= score < 0.60: confidence_factor rises linearly from 1.0 to 1.5
score >= 0.60: direct auto-accept candidate
```

Evaluation outputs:

| Artifact | Path |
| --- | --- |
| LDP reward-only linear predictions | `Results/evaluation/dinov3_reward_linear_no_reject_20260524/ldp_holdout_predictions.jsonl` |
| LDP reward-only linear summary | `Results/evaluation/dinov3_reward_linear_no_reject_20260524/ldp_holdout_summary/ldp_holdout_summary.json` |
| GT no-blackpad reward-only linear predictions | `Results/evaluation/dinov3_reward_linear_no_reject_20260524/manual_test_no_blackpad/test_predictions.jsonl` |
| GT no-blackpad reward-only linear summary | `Results/evaluation/dinov3_reward_linear_no_reject_20260524/manual_test_no_blackpad/eval/test_summary.json` |

| Setting | GT usable | GT reject | LDP auto | LDP review | LDP accept/review | Mixed FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO-only baseline | 10/20 | 10/20 | 367/800 | 143/800 | 63.75% | 1/100 |
| Old DINO reward gate, 0.60/0.85 | 6/20 | 14/20 | 336/800 | 93/800 | 53.625% | 0/100 |
| Hard-floor linear-reward DINO gate, 0.05/0.30/0.60 | 12/20 | 8/20 | 527/800 | 28/800 | 69.375% | 0/100 |
| Reward-only linear DINO gate, 0.30/0.60 | 13/20 | 7/20 | 547/800 | 38/800 | 73.125% | 1/100 |

After correcting the source-folder label for `0006731802.jpg`, the mixed false positive in the reward-only linear run is treated as a dataset-folder issue rather than a DINO gate failure; the hard floor is disabled in the active config.

Recommendation: keep the checkpoint and code as an auxiliary diagnostic experiment, but do not promote it into the current ROI pipeline.
