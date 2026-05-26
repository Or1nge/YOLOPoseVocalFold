# LDP holdout refresh, 2026-05-25

The previous holdout manifest was built on 2026-05-23 and was removed after the
LDP dataset refresh. The replacement manifest samples the current LDP tree
directly, without depending on old prediction outputs:

```text
data/ldp_holdout/ldp_holdout_100_per_class_seed20260525.jsonl
```

Sampled classes are the same eight-class evaluation set as before:

```text
喉癌, 声带任克水肿, 声带囊肿, 声带息肉, 声带白斑, 声带肉芽肿, 正常, 混杂图片
```

Each class has 100 images. The source availability at sampling time was:

```text
喉癌 996
声带任克水肿 408
声带囊肿 667
声带息肉 1907
声带白斑 736
声带肉芽肿 505
正常 2404
混杂图片 5939
```

The DINOv3 auxiliary training dataset was rebuilt with the refreshed holdout as
its exclusion manifest:

```text
data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded_gt396_seed20260525
```

It contains `477/79/40` train/val/test images, including 200 train-only mixed
negative images, and has zero overlap with the refreshed holdout. Hard-negative
mining predictions were regenerated at:

```text
Results/dinov3_keypoint_aux/hard_negative_mining_20260525/train_gt396_ldp200_seed20260525_predictions.jsonl
```

The refreshed DINOv3 head is:

```text
Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200_gt396_seed20260525/weights/best_aux_head.pt
```

It reached best epoch 12 with validation loss `0.2840102077`.

## Evaluation

The refreshed LDP holdout was evaluated with the mixed-negative-120 blackpad
YOLO-Pose checkpoint, then rescored with the refreshed DINOv3 reward-only gate.

| Setting | Auto accept | Manual review | Reject | Accept/review | Mixed FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO-only, refreshed holdout | 625/800 | 9/800 | 166/800 | 79.25% | 0/100 |
| DINO reward gate, refreshed holdout | 631/800 | 8/800 | 161/800 | 79.875% | 0/100 |

Manual GT test behavior was unchanged in usable count: YOLO-only had
`36 auto_accept + 1 manual_review + 3 reject`; DINO changed the single manual
review to auto-accept, leaving `37 usable / 40` and the same 87%-containment
rate of `0.85`.
