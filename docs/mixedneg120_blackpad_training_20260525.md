# 120 张 blackpad 混杂负样本训练记录

## 训练

- 时间：2026-05-25
- 配置：`configs/train_containment_mixed_negative_120_blackpad_y11m.yaml`
- 数据：`data/yolo_pose_mixed_negative_120_blackpad`
- 权重：`Results/containment_loss/yolo_pose_glottic_three_point_y11m_img960_pose24_containment_l0p05_mixedneg120_blackpad/weights/best.pt`
- 训练命令：`python3 tools/train_keypoint_containment.py --config configs/train_containment_mixed_negative_120_blackpad_y11m.yaml --enable-unstable-loss-hook`

YOLO native validation 在 120 epoch 正常完成。按 pose mAP50-95 选出的最好 epoch 是 106：

```text
box precision/recall: 0.97284 / 0.90693
box mAP50 / mAP50-95: 0.96633 / 0.53342
pose precision/recall: 0.99781 / 0.91139
pose mAP50 / mAP50-95: 0.97373 / 0.96375
```

最终 `best.pt` 复验结果：

```text
box precision/recall: 0.986 / 0.910
box mAP50 / mAP50-95: 0.959 / 0.566
pose precision/recall: 1.000 / 0.923
pose mAP50 / mAP50-95: 0.977 / 0.961
```

## 后处理评估

人工 test split：

```text
count: 40
actions: auto_accept=36, manual_review=1, reject_or_relabel=3
mean_bbox_iou: 0.6754
mean_containment_rate: 1.0000
mean_normalized_keypoint_error: 0.0378
mean_roi_polygon_containment_rate: 0.9506
roi_polygon_containment_ge_87_rate: 0.8500
```

注：最初直接用 `evaluate_predictions.py` 跑出的 ROI 覆盖率偏低，因为预测 JSONL 的坐标在 `predict_roi.py` 重新 crop+blackpad 后的输入图上，而 GT 仍在原 split 图坐标上。2026-05-25 已修正评估脚本，使其按 prediction `preprocess` 元数据把 GT bbox/keypoints/ROI 映射到同一坐标系后再计算指标。

LDP holdout：

```text
count: 800
overall actions: auto_accept=632, manual_review=13, reject_or_relabel=155
overall auto_accept_rate: 0.7900
overall reject_rate: 0.1938
混杂图片 auto_accept_rate: 0.0300
混杂图片 manual_review_rate: 0.0100
混杂图片 reject_rate: 0.9600
混杂图片 false_positive_rate: 0.0400
```

## 判断

这版 native keypoint 指标很好，人工 test 的同坐标系 ROI 覆盖也恢复到可用水平，LDP 普通类别通过率高。混杂图片拒绝能力优于 2026-05-24 DINO reward 线的普通类通过表现，但不如 V1.1 smoke holdout 的 0% mixed false positive 保守。

暂不直接替换当前推荐别名。若要推进这版，需要先调 ROI 后处理或单独比较 crop 视觉质量，再决定是否提升为新 alias。
