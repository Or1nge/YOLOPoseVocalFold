# YOLOPoseVocalFold

Anatomy-Constrained YOLO-Pose for Vocal Fold ROI Localization.

本项目把声门/喉镜图像自动定位任务实现为：

```text
image -> 3 anatomy keypoints -> angle-bisector rotated final_box_polygon + final_confidence
```

主分支现在包含标准 YOLO-Pose 训练、关键点几何后处理、keypoint-containment loss 训练入口，以及 LDP pseudo 微调/裁剪流程。

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

通用放置方式：

```text
data/images/
data/labelme/
```

当前 2026-05-20 黑边增强标注集已经放在
`data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images/`，图片和 JSON 同目录。

每张 LabelMe JSON 默认需要：

- 1 个旋转框：`声门区域`
- 3 个点：`前联合`, `左后方中点`, `右后方中点`

如果标签名不同，改 `configs/keypoints.yaml`。

转换为 YOLO-Pose：

```bash
python tools/convert_labelme_to_yolo_pose.py \
  --labelme-dir data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images \
  --image-dir data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images \
  --out-dir data/yolo_pose \
  --split-source-dir data/splits/glottic_roi_rectangle_annotation_200_20260520_blackpad
```

转换脚本会按图片内容哈希把精确重复图片固定在同一 split，避免同一画面同时进入训练和验证/测试。

检查数据：

```bash
python tools/validate_dataset.py --dataset-dir data/yolo_pose
```

如果确认 `混杂图片` 都没有声带区域，可把它们作为 YOLO-Pose 负样本加入训练集。负样本图片复制到新数据集的 `images/train/`，对应 `labels/train/*.txt` 保持空文件，表示背景/无声门 ROI：

```bash
python tools/add_negative_images_to_yolo_pose.py \
  --base-dataset data/yolo_pose \
  --negative-source-dir /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed/混杂图片 \
  --out-dir data/yolo_pose_mixed_negative_60 \
  --count 60 \
  --seed 20260522
```

带 60 张混杂负样本的训练配置在 `configs/train_containment_mixed_negative_y11m.yaml`。

如果要让 LDP 参与微调，同时强惩罚 `混杂图片` 误检，可先用上一轮 LDP 八分类预测结果构造 pseudo 数据集。该流程只读取 LDP 或已有 `input_links`，把训练用图片复制到项目本地数据目录，不修改 LDP 原始目录：

```bash
python tools/build_ldp_pseudo_pose_dataset.py \
  --base-dataset data/yolo_pose_mixed_negative_60 \
  --predictions Results/ldp_8class_roi_crop_mixedneg60_auto043_20260522_073221/predictions.jsonl \
  --out-dir data/yolo_pose_ldp_pseudo_mixedpenalty_copy \
  --positive-actions auto_accept manual_review \
  --min-positive-confidence 0.30 \
  --negative-repeat 1 \
  --hard-negative-repeat 8 \
  --copy-mode copy \
  --overwrite
```

对应微调配置为 `configs/train_containment_ldp_pseudo_mixedpenalty_copy_y11m.yaml`。当前约定是：非 `混杂图片` 的 `auto_accept`/`manual_review` 且 `final_confidence >= 0.30` 作为 pseudo-positive；全部 `混杂图片` 作为空标签负样本；`混杂图片` 中曾被 `auto_accept` 的样本额外重复 8 次作为 hard negative。

## 训练 baseline

先确认有效配置：

```bash
python tools/train_yolo_pose.py --dry-run
```

开始训练：

```bash
python tools/train_yolo_pose.py --config configs/train_baseline.yaml
```

训练输出写入 `Results/glottic_three_point/`，并在每个 run 目录内记录 `run_metadata.json`，包含命令、配置和 Git 版本信息。

## Keypoint-Containment Loss

当前 3 点声门 ROI 流程可额外加入训练期 containment loss：

```text
loss_total = loss_yolo_pose + lambda_containment * loss_containment
```

`loss_containment` 作用在 decoded predicted bbox 与 decoded predicted keypoints 上：当预测出来的 3 个关键点落到预测 bbox 外时产生 hinge penalty；关键点本身仍由模型预测，不作为输入喂给模型。

当前后处理使用 87% 调参结果：

```text
roi_base_backtrack_fraction = 0.05
roi_posterior_margin_fraction = 0.20
roi_side_margin_fraction = 0.70
confidence_curve = tanh
confidence_gamma = 1.0
confidence_tanh_midpoint = 0.65
confidence_tanh_steepness = 6.0
review_threshold = 0.30
auto_accept_threshold = 0.43
min_roi_area_ratio = 0.03
good_roi_area_ratio = 0.08
keypoint_image_bounds_tolerance_px = 5.0
roi_dark_luma_threshold = 75.0
min_roi_dark_fraction = 0.10
good_roi_dark_fraction = 0.20
```

`confidence_curve` 会同时作用于检测框置信度和关键点置信度；当前 `tanh` 曲线在 0.65 附近拉开中低置信度和高置信度，同时避免简单平方把 0.9 这类高置信度也压低太多。旧行为可用 `confidence_curve=power` 且 `confidence_gamma=2.0` 复现。ROI 面积和关键点图像边界 gate 用于降低明显非声带区域的“可用框”风险。
后处理不要求前联合 A 在图像 y 方向上低于 L/R 后方点；体位和镜头方向可能改变上下关系，三点只通过夹角、几何一致性、ROI 面积、图像边界和暗区比例等规则判断。
启用 `roi_dark_fraction` 后，推理会额外检查预测旋转 ROI 内是否存在足够暗的声门样区域；该 gate 只使用预测框和原图像素，不依赖人工标注。

本机如果数据放在 main checkout，可这样启动单个 lambda：

```bash
python tools/train_keypoint_containment.py \
  --config configs/train_containment_lambda_sweep.yaml \
  --lambda-containment 0.05 \
  --data data/yolo_pose/vocal_fold_pose.yaml \
  --device 0 \
  --enable-unstable-loss-hook
```

## 推理

当前推荐模型短名为 `vf_roi_v1`：

```text
Results/models/vf_roi_v1/best.pt
```

```bash
python tools/predict_roi.py \
  --weights Results/models/vf_roi_v1/best.pt \
  --source data/yolo_pose/images/val \
  --postprocess-config Results/models/vf_roi_v1/postprocess.yaml \
  --out Results/predictions/val_predictions.jsonl
```

每条输出包含：

- `bbox_yolo`
- `bbox_keypoints`
- `roi_polygon`: 由 3 点生成的角平分线旋转 ROI
- `final_box_polygon`: 最终四点旋转框；一条边平行于前联合夹角的角平分线，左右宽度按两侧后方点到角平分线的投影分别扩张
- `final_bbox_xyxy` / `final_bbox`: 最终旋转框的 axis-aligned 外接矩形，仅用于兼容传统 bbox 评估
- `usable_box_polygon`: 置信度过线时可用；低置信度时为 `null`
- `usable_bbox`: `usable_box_polygon` 的 axis-aligned 外接矩形兼容字段
- `glottic_angle_degrees`
- `geometry_score`
- `consistency_score`
- `roi_area_ratio` / `roi_area_factor`
- `max_keypoint_outside_image_px` / `image_bounds_factor`
- `roi_dark_fraction` / `roi_dark_factor`
- `final_confidence`
- `action`: `auto_accept`, `manual_review`, `reject_or_relabel`
- `flags`: 低置信度或几何不一致原因

批量生成裁剪图时，可先用 `tools/predict_roi.py` 写出 JSONL，再按原始类别目录裁剪 `auto_accept` 的旋转 ROI：

```bash
python tools/crop_rois_from_predictions.py \
  --predictions Results/predictions/ldp_8class_predictions.jsonl \
  --source-root Results/predictions/ldp_8class_input_links \
  --out-dir Results/predictions/ldp_8class_roi_crops \
  --crop-actions auto_accept \
  --crop-mode polygon
```

该脚本会保留源目录结构，并输出 `roi_crop_manifest.csv` 与 `roi_crop_summary.json`。

## 评估

```bash
python tools/evaluate_predictions.py \
  --predictions Results/predictions/val_predictions.jsonl \
  --dataset-dir data/yolo_pose \
  --split val \
  --out-dir Results/evaluation
```

评估会输出 bbox IoU、关键点 PCK、点是否落在框内、旋转 ROI 对人工 `声门区域` 的覆盖率、面积比例、最终置信度和人工复核比例。

## 几何调参

转换脚本会在 `data/yolo_pose/roi_polygons/` 写出人工 ROI 多边形元数据，可用于调三点角平分线 ROI 的边距：

```bash
python tools/tune_geometry_roi.py \
  --dataset-dir data/yolo_pose \
  --split train \
  --postprocess-out Results/geometry_tuning/glottic_three_point/postprocess_tuned.yaml
```

## Git 说明

```text
main
  三点 YOLO-Pose + 角平分线旋转 ROI + containment loss + LDP pseudo 微调流程
```

旧的 `exp/keypoint-containment-loss` 实验分支已经合入 main；后续直接在 main 上维护。
