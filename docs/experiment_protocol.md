# 实验协议

## 主分支 baseline

主分支只包含标准 YOLO-Pose 训练和推理后处理：

1. LabelMe 转 YOLO-Pose 标签。
2. 标准 YOLO-Pose 训练。
3. 推理得到 `bbox + 3 keypoints + confidence`。
4. 以前联合为顶点，连接左右后方中点并取夹角角平分线。
5. 沿角平分线反方向回退一小段到 A 点，以垂直于角平分线的线段为底，生成旋转 ROI。
6. 输出 `final_box_polygon` 作为最终四点旋转框：一条边平行于角平分线，左右宽度按两侧后方点到角平分线的投影分别扩张；`final_bbox_xyxy` / `final_bbox` 只作为传统 bbox 评估的外接矩形兼容字段。
7. 计算 `final_confidence` 并输出 `auto_accept / manual_review / reject_or_relabel`；低置信度样本保留置信度和原因，但 `usable_box_polygon` 置空，不作为自动 ROI。当前配置会用 `confidence_gamma` 拉开中低置信度差异，并用 ROI 面积、关键点是否越出图像边界等 gate 降低明显非声带区域的自动可用风险。

主分支不改 YOLO loss，方便建立可解释 baseline。

## 分支实验

`exp/keypoint-containment-loss` 只测试 containment loss：

```text
loss_total = loss_yolo_pose + lambda * loss_containment
```

本分支已同步 main 的 3 点角平分线流程，并固定使用 87% 几何调参结果。训练期 containment 项只约束 decoded predicted bbox 与 decoded predicted keypoints：3 个点由模型预测，训练时不会作为图像输入。

分支比较时应固定同一套 87% 后处理参数，否则无法判断收益来自 loss 还是来自规则变化。

### 当前实验实现

- 可复用 loss：`yoloposevf/containment_loss.py`
- 实验入口：`tools/train_keypoint_containment.py`
- lambda sweep：`configs/train_containment_lambda_sweep.yaml`
- 本地可验证命令：

```bash
python tools/train_keypoint_containment.py --dry-run
python tools/train_keypoint_containment.py --smoke-loss
pytest tests/test_containment_loss.py
```

`loss_containment` 对每个预测关键点计算其落在预测 bbox 外的 hinge 距离，可按 bbox 尺度归一化，并支持 visibility mask。所有预测关键点在预测 bbox 内时 loss 为 0。默认 sweep 采用 `0.0 / 0.01 / 0.05 / 0.1`。

## 比较指标

- `keypoints outside final rotated box rate`
- `bbox-keypoint consistency score`
- `bbox IoU`
- `keypoint PCK`
- `roi polygon containment >= 87%` 比例
- `roi area ratio to target`
- LDP 直接推理中的 `roi_area_too_small / low_roi_area / keypoints_outside_image` 触发比例
- `manual_review` 比例
- 失败样本类型

只有当 containment loss 降低点在框外比例，同时不损害 bbox/keypoint 精度，才考虑合并回主线。
