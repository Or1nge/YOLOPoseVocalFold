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

### 当前实验实现

- 可复用 loss：`yoloposevf/containment_loss.py`
- 实验入口：`tools/train_keypoint_containment.py`
- lambda sweep：`configs/train_containment_lambda_sweep.yaml`
- 本地可验证命令：

```bash
python3 tools/train_keypoint_containment.py --dry-run
python3 tools/train_keypoint_containment.py --smoke-loss
pytest tests/test_containment_loss.py
```

`loss_containment` 对每个预测关键点计算其落在预测 bbox 外的 hinge 距离，可按 bbox 尺度归一化，并支持 visibility mask。所有关键点在框内时 loss 为 0。

### Ultralytics hook 状态

本分支提供了 `PoseTrainer` subclass 接入点，但 Ultralytics 的 pose loss 预测张量不是稳定公开接口；不同版本可能需要不同的 decode 逻辑。当前代码默认拒绝直接启动真实训练，以避免误以为 loss 已经接进官方训练图。真实训练前需要在安装了 Ultralytics 和真实数据的环境里完成以下检查：

1. 确认 `PoseTrainer` 和当前 Ultralytics 版本的 loss/prediction tensor 形状。
2. 将预测 `bbox_xyxy` 和 `keypoints_xy` 接到 `containment_penalty_torch`。
3. 用 1 个小 batch 验证 `loss_total` 中 containment 项非零且可反向传播。
4. 再按 lambda sweep 运行完整训练。

## 比较指标

- `keypoints outside bbox rate`
- `bbox-keypoint consistency score`
- `bbox IoU`
- `keypoint PCK`
- `manual_review` 比例
- 失败样本类型

只有当 containment loss 降低点在框外比例，同时不损害 bbox/keypoint 精度，才考虑合并回主线。
