# YOLOPoseVocalFold

Anatomy-Constrained YOLO-Pose for Vocal Fold ROI Localization.

本项目把声门/喉镜图像自动定位任务实现为：

```text
image -> ROI bbox + 4 vocal-fold keypoints -> geometry fusion -> final_bbox + final_confidence
```

主分支只做稳定 baseline：标准 YOLO-Pose 训练 + 关键点几何后处理。实验性 `keypoint-containment loss` 放在独立 Git 分支 `exp/keypoint-containment-loss`。

## 目录

```text
configs/                  # 关键点、训练、后处理配置
data/                     # 数据占位；真实图片和 LabelMe JSON 不提交
docs/                     # 标注规范和实验协议
tools/                    # 转换、验证、训练、推理、评估入口
yoloposevf/               # 几何融合、指标、标签格式等核心代码
Results/                  # 训练和评估输出，默认不提交
```

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

先把真实数据放入：

```text
data/images/
data/labelme/
```

每张 LabelMe JSON 默认需要：

- 1 个框：`vocal_fold_roi`
- 4 个点：`kp1`, `kp2`, `kp3`, `kp4`

如果标签名不同，改 `configs/keypoints.yaml`。

转换为 YOLO-Pose：

```bash
python tools/convert_labelme_to_yolo_pose.py \
  --labelme-dir data/labelme \
  --image-dir data/images \
  --out-dir data/yolo_pose
```

检查数据：

```bash
python tools/validate_dataset.py --dataset-dir data/yolo_pose
```

## 训练 baseline

先确认有效配置：

```bash
python tools/train_yolo_pose.py --dry-run
```

开始训练：

```bash
python tools/train_yolo_pose.py --config configs/train_baseline.yaml
```

训练输出写入 `Results/baseline/`，并在每个 run 目录内记录 `run_metadata.json`，包含命令、配置和 Git 版本信息。

## 实验：keypoint-containment loss

本分支只测试训练期 bbox/keypoint containment 约束，不改 `configs/postprocess.yaml` 或推理后处理逻辑。

可先做无数据、无 Ultralytics 的配置和 loss smoke 检查：

```bash
python3 tools/train_keypoint_containment.py --dry-run
python3 tools/train_keypoint_containment.py --smoke-loss
```

lambda sweep 配置在：

```text
configs/train_containment_lambda_sweep.yaml
```

实验 loss 模块在 `yoloposevf/containment_loss.py`，核心形式为：

```text
loss_total = loss_yolo_pose + lambda_containment * loss_containment
```

注意：当前入口为 Ultralytics 8.4.x 的 `v8PoseLoss` 张量流实现了实验性 criterion subclass，会在 decoded predicted bbox 与 decoded predicted keypoints 上加入 containment 项。脚本默认拒绝直接训练，只有显式传入 `--enable-unstable-loss-hook` 才会启动该实验 hook；如果以后升级 Ultralytics，先用小 batch 检查 loss 张量形状。

## 推理

```bash
python tools/predict_roi.py \
  --weights Results/baseline/yolo_pose_baseline/weights/best.pt \
  --source data/yolo_pose/images/val \
  --out Results/predictions/val_predictions.jsonl
```

每条输出包含：

- `bbox_yolo`
- `bbox_keypoints`
- `final_bbox`
- `final_confidence`
- `action`: `auto_accept`, `manual_review`, `reject_or_relabel`
- `flags`: 低置信度或几何不一致原因

## 评估

```bash
python tools/evaluate_predictions.py \
  --predictions Results/predictions/val_predictions.jsonl \
  --dataset-dir data/yolo_pose \
  --split val \
  --out-dir Results/evaluation
```

评估会输出 bbox IoU、关键点 PCK、点是否落在框内、最终置信度和人工复核比例。

## Git 分支

```text
main
  标准 YOLO-Pose baseline + 几何融合后处理

exp/keypoint-containment-loss
  只测试 bbox 包含 keypoints 的训练约束，不改主分支后处理
```

分支比较应使用同一批数据、同一套 split、同一套增强策略和同一套评估脚本。
