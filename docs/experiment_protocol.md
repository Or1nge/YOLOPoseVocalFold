# 实验协议

## 当前 main 流程

main 现在包含标准 YOLO-Pose 训练、推理后处理、keypoint-containment loss 训练入口，以及历史 LDP pseudo 微调工具：

1. LabelMe 转 YOLO-Pose 标签。
2. 标准 YOLO-Pose 训练。
3. 推理得到 `bbox + 3 keypoints + confidence`。
4. 以前联合为顶点，连接左右后方中点并取夹角角平分线。
5. 沿角平分线反方向回退一小段到 A 点，以垂直于角平分线的线段为底，生成旋转 ROI。
6. 输出 `final_box_polygon` 作为最终四点旋转框：一条边平行于角平分线，左右宽度按两侧后方点到角平分线的投影分别扩张；`final_bbox_xyxy` / `final_bbox` 只作为传统 bbox 评估的外接矩形兼容字段。
7. 计算 `final_confidence` 并输出 `auto_accept / manual_review / reject_or_relabel`；低置信度样本保留置信度和原因，但 `usable_box_polygon` 置空，不作为自动 ROI。当前配置会用 `confidence_curve=tanh` 拉开中低置信度差异，并用 ROI 面积、关键点是否越出图像边界等 gate 降低明显非声带区域的自动可用风险。

## Keypoint-Containment Loss

```text
loss_total = loss_yolo_pose + lambda * loss_containment
```

训练期 containment 项只约束 decoded predicted bbox 与 decoded predicted keypoints：3 个点由模型预测，训练时不会作为图像输入。实验比较时应固定同一套 87% 后处理参数，否则无法判断收益来自 loss 还是来自规则变化。

### 当前实现

- 可复用 loss：`yoloposevf/containment_loss.py`
- 实验入口：`tools/train_keypoint_containment.py`
- lambda sweep：`configs/train_containment_lambda_sweep.yaml`
- LDP pseudo 微调：`configs/train_containment_ldp_pseudo_mixedpenalty_copy_y11m.yaml`
- 本地可验证命令：

```bash
python tools/train_keypoint_containment.py --dry-run
python tools/train_keypoint_containment.py --smoke-loss
pytest tests/test_containment_loss.py
```

`loss_containment` 对每个预测关键点计算其落在预测 bbox 外的 hinge 距离，可按 bbox 尺度归一化，并支持 visibility mask。所有预测关键点在预测 bbox 内时 loss 为 0。默认 sweep 采用 `0.0 / 0.01 / 0.05 / 0.1`。

## Oriented Keypoint Contrast Pretraining

当前 contrast 分支不再把 LDP pseudo-positive 三点作为 contrast 训练输入。新的推荐流程是三阶段：

```text
Stage 1:
  data/yolo_pose
  人工三点标注图: A/L/R 有向局部 contrast

Stage 2:
  data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded
  人工三点标注图 + 200 张 LDP 混杂图片 blackpad 空标签负样本

Stage 3:
  data/yolo_pose_mixed_negative_60_blackpad
  常规 YOLO-Pose + containment fine-tune
```

当前实现：

- 可复用模块：`yoloposevf/keypoint_contrast.py`
- 有向 contrast pretrain 入口：`tools/pretrain_oriented_contrast.py`
- 串联入口：`tools/run_three_stage_oriented_mixed_reject_pipeline.py`
- pretrain 配置：`configs/pretrain_oriented_keypoint_contrast_manual_only_y11m.yaml`
- hard-negative 配置：`configs/train_stage2_mixed_hard_negative_y11m.yaml`
- final fine-tune 配置：`configs/train_stage3_pose_after_mixed_hard_negative_y11m.yaml`
- 本地可验证命令：

```bash
python tools/pretrain_oriented_contrast.py --config configs/pretrain_oriented_keypoint_contrast_manual_only_y11m.yaml --dry-run
pytest tests/test_keypoint_contrast.py
```

训练时从当前 batch 生成第二个轻增强视图，同步变换人工 A/L/R 三点坐标。对每张有三点的图像，先计算 A 到 L/R midpoint 的角平分线方向，再在每个关键点附近采样有向局部 patch。默认主配置是约 `48x48` 输入像素足迹；rect48x72 消融中，A 使用 canonical `48x72`，L/R 使用 canonical `72x48`；rect36x60 消融中，A 使用 canonical `36x60`，L/R 使用 canonical `60x36`。patch 的 canonical y 轴沿 anterior-to-posterior 方向，x 轴为其垂线，因此前联合附近的声带/声门区上下关系会被摆正后再进入 projection head。

正样本只定义为“同一图像、同一关键点、不同增强视图”：`A1-A2`、`L1-L2`、`R1-R2`。A/L/R 三个解剖点不被当作互相接近的目标。背景随机点、远离人工三点的区域提供负样本。Stage 1 不使用 LDP 图像；Stage 2 只使用 LDP 混杂图作为空标签负样本，不使用非混杂 LDP pseudo-positive 三点。

该 pretrain 会对每个 batch 额外 forward 一次轻增强视图，比较时需要记录 batch size、显存、训练耗时和是否因为显存调整了其他超参。

2026-05-23 三阶段 200 张 blackpad 混杂负样本实测结果记录在 `docs/three_stage_oriented_mixed_reject_evaluation_20260523.md`。结论：该版恢复并提升了 native manual-test pose mAP50，但 LDP holdout `混杂图片` false-positive 升到 8%，不推荐替代当前 V1.1 ROI 模型。已删除的 6,247 张负样本 run 说明负样本比例过大时会压制有效 pose 检出。

## DINOv3 Three-Point Auxiliary Judgement

`exp/dinov3-keypoint-aux` 分支把后续探索从 keypoint-local contrast 改为 DINOv3 辅助判定。核心约束：

- 不把同一图的左后方和右后方当作正样本；左右病变差异不能被镜像假设抹平。
- 冻结 DINOv3 dense encoder，只训练轻量 head。
- point-region head 预测有向局部区域属于 `background / anterior / left posterior / right posterior`。
- 当前 active 版本不训练 triplet head 或 image-level reject head。
- hard negative 来自排除 holdout 后的混杂图误检点，以及人工关键点附近的 near-miss background。
- DINO 输入图不做额外裁黑边；训练/推理使用当前数据集或 prediction JSONL 的图像路径，再 letterbox 到 DINO 输入尺寸。
- 点区域 head 采样同一有向 48x48 局部区域的有效像素 mask，黑边/无效位置在局部特征中被置零，mask 本身也作为局部输入提供给 head。

当前实现：

- 可复用模块：`yoloposevf/dinov3_aux.py`
- 训练入口：`tools/train_dinov3_keypoint_aux.py`
- JSONL 打分入口：`tools/score_predictions_with_dinov3_aux.py`
- 配置：`configs/train_dinov3_keypoint_aux_y11m.yaml`
- 设计记录：`docs/dinov3_keypoint_aux_design_20260524.md`

本地可验证命令：

```bash
python tools/train_dinov3_keypoint_aux.py --config configs/train_dinov3_keypoint_aux_y11m.yaml --dry-run
pytest tests/test_dinov3_keypoint_aux.py
```

默认打分入口只追加 `dinov3_aux` 字段，不改变已有 `final_confidence/action`。只有显式加 `--apply-confidence-gate` 时，才把 DINOv3 auxiliary factor 应用到 `final_confidence/action`。当前 gate：低于 `0.30` 不动，`0.30-0.60` 按比例从 `1.0x` 奖励到 `1.5x`，`>=0.60` 作为 DINO 极高分直接通过候选；DINO 低分不再直接拒绝。

### 历史 LDP pseudo 两阶段数据设计

以下为 2026-05-23 的历史消融流程，已不再作为当前推荐 contrast 训练方案。经验教训记录在 `docs/contrast_experiment_lessons_20260523.md`，旧 `Results/` 产物和本地 `data/yolo_pose_stage2_ldp_conf040_holdout100/` 物化数据集已清理：

1. Stage 1 用 `data/yolo_pose_mixed_negative_60` 训练/验证 ROI 定位能力。这套数据只包含人工三点标注 split 和 60 张混杂图片空标签负样本。
2. Stage 2 用 `data/yolo_pose_stage2_ldp_conf040_holdout100` 做 LDP 辅助微调。该数据集复制 Stage 1 数据，再追加 LDP pseudo-positive 和 LDP 混杂负样本。
3. 最终同时在人工 `test` split 和 LDP holdout 上评估。

LDP pseudo-positive 只接受 `final_confidence > 0.4` 的非混杂样本；`<= 0.4` 的非混杂样本不进入训练。LDP holdout 固定为 `data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl`，八类各 100 张，构造 stage-2 数据集时通过 `--exclude-manifest` 排除。

历史 stage-2 数据构造结果：

```text
holdout: 800 records, 100 per LDP class
stage2 train: 10378 images
stage2 train negatives: 6379 empty labels
LDP pseudo-positive: 3862
LDP skipped by confidence/action/holdout: skipped_holdout=800, skipped_not_training_sample=3225
```

## 比较指标

- `keypoints outside final rotated box rate`
- `bbox-keypoint consistency score`
- `bbox IoU`
- `keypoint PCK`
- `normalized keypoint error`
- `混杂图片` 误检率
- `roi polygon containment >= 87%` 比例
- `roi area ratio to target`
- LDP 直接推理中的 `roi_area_too_small / low_roi_area / keypoints_outside_image / roi_dark_fraction` 触发比例
- `manual_review` 比例
- 失败样本类型

只有当新增 loss 降低点在框外比例、混杂/反光/伪暗区误检和 manual review 比例，同时不损害 bbox/keypoint 精度，才考虑合并回主线。
