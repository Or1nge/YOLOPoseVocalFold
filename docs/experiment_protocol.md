# 实验协议

## 主分支 baseline

主分支只包含标准 YOLO-Pose 训练和推理后处理：

1. LabelMe 转 YOLO-Pose 标签。
2. 标准 YOLO-Pose 训练。
3. 推理得到 `bbox + 4 keypoints + confidence`。
4. 根据关键点生成 `bbox_keypoints`。
5. 融合 `bbox_yolo` 与 `bbox_keypoints`。
6. 计算 `final_confidence` 并输出 `auto_accept / manual_review / reject_or_relabel`。

主分支不改 YOLO loss，方便建立可解释 baseline。

## 分支实验

`exp/keypoint-containment-loss` 只测试 containment loss：

```text
loss_total = loss_yolo_pose + lambda * loss_containment
```

分支不得同时改后处理逻辑，否则无法判断收益来自 loss 还是来自规则变化。

## 比较指标

- `keypoints outside bbox rate`
- `bbox-keypoint consistency score`
- `bbox IoU`
- `keypoint PCK`
- `manual_review` 比例
- 失败样本类型

只有当 containment loss 降低点在框外比例，同时不损害 bbox/keypoint 精度，才考虑合并回主线。

