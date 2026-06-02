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

如果确认 `混杂图片` 都没有声带区域，可把它们作为 YOLO-Pose 负样本加入训练集。当前重建版本加入 120 张黑边增强混杂负样本，并排除固定 LDP holdout，负样本图片复制到新数据集的 `images/train/`，对应 `labels/train/*.txt` 保持空文件，表示背景/无声门 ROI：

```bash
python tools/add_negative_images_to_yolo_pose.py \
  --base-dataset data/yolo_pose \
  --negative-source-dir /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed/混杂图片 \
  --out-dir data/yolo_pose_mixed_negative_120_blackpad \
  --count 120 \
  --seed 20260525 \
  --exclude-manifest data/ldp_holdout/ldp_holdout_100_per_class_seed20260525.jsonl \
  --prefix mixed_negative120_blackpad \
  --blackpad-negatives \
  --overwrite
```

带 120 张黑边混杂负样本的训练配置在 `configs/train_containment_mixed_negative_120_blackpad_y11m.yaml`。

如果要让 LDP 参与微调，同时强惩罚 `混杂图片` 误检，可先用上一轮 LDP 八分类预测结果构造 pseudo 数据集。该流程只读取 LDP 或已有 `input_links`，把训练用图片复制到项目本地数据目录，不修改 LDP 原始目录：

```bash
python tools/build_ldp_pseudo_pose_dataset.py \
  --base-dataset data/yolo_pose_mixed_negative_60 \
  --predictions Results/ldp_8class_roi_crop_mixedneg60_auto043_20260522_073221/predictions.jsonl \
  --out-dir data/yolo_pose_ldp_pseudo_mixedpenalty_copy \
  --positive-actions auto_accept manual_review \
  --min-positive-confidence 0.40 \
  --negative-repeat 1 \
  --hard-negative-repeat 8 \
  --copy-mode copy \
  --overwrite
```

对应微调配置为 `configs/train_containment_ldp_pseudo_mixedpenalty_copy_y11m.yaml`。当前约定是：非 `混杂图片` 的 `auto_accept`/`manual_review` 且 `final_confidence > 0.40` 才作为 pseudo-positive；`<= 0.40` 的非混杂图不参与 pseudo 训练；全部 `混杂图片` 作为空标签负样本；`混杂图片` 中曾被 `auto_accept` 的样本额外重复 8 次作为 hard negative。

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
roi_dark_mode = relative_foreground_median
roi_dark_luma_threshold = 75.0
roi_dark_relative_luma_ratio = 0.80
roi_dark_foreground_luma_floor = 8.0
min_roi_dark_fraction = 0.10
good_roi_dark_fraction = 0.20
```

`confidence_curve` 会同时作用于检测框置信度和关键点置信度；当前 `tanh` 曲线在 0.65 附近拉开中低置信度和高置信度，同时避免简单平方把 0.9 这类高置信度也压低太多。旧行为可用 `confidence_curve=power` 且 `confidence_gamma=2.0` 复现。ROI 面积和关键点图像边界 gate 用于降低明显非声带区域的“可用框”风险。ROI 相对面积的分母使用模型实际输入图裁掉黑边后的有效图像面积，不区分黑边是原图自带还是 V1.1 推理时额外加入。
后处理不要求前联合 A 在图像 y 方向上低于 L/R 后方点；体位和镜头方向可能改变上下关系，三点只通过夹角、几何一致性、ROI 面积、图像边界和暗区比例等规则判断。
三点夹角低于 `20°` 或高于 `130°` 时只作为温和几何扣分，不再直接把 `geometry_score` 置零；单独由夹角造成的最大惩罚约为乘以 `0.6`。
启用 `roi_dark_fraction` 后，推理会额外检查预测旋转 ROI 内是否存在足够暗的声门样区域；该 gate 只使用预测框和原图像素，不依赖人工标注。当前默认使用相对亮度：先排除接近纯黑的人工黑边，再取整图前景中位灰度作为参考，像素灰度低于 `reference_luma * 0.80` 才计入暗区；`roi_dark_luma_threshold` 仅用于复现旧的 absolute 模式。

本机如果数据放在 main checkout，可这样启动单个 lambda：

```bash
python tools/train_keypoint_containment.py \
  --config configs/train_containment_lambda_sweep.yaml \
  --lambda-containment 0.05 \
  --data data/yolo_pose/vocal_fold_pose.yaml \
  --device 0 \
  --enable-unstable-loss-hook
```

## Keypoint-Level Local Contrast

实验分支 `exp/keypoint-local-contrast` 改为三阶段流程。新的主流程不再把 LDP pseudo-positive 三点拉入 contrast 训练；LDP 只以混杂图片负样本形式进入 hard-negative 阶段：

```text
Stage 1: 人工三点有向局部 contrast pretrain
Stage 2: 随机 200 张 LDP 混杂图片 blackpad 空标签 hard-negative 训练
Stage 3: 常规 YOLO-Pose + containment fine-tune
```

第一阶段在 YOLO neck/head 输入特征图上，根据人工 A/L/R 三点计算前联合到左右后方中点 midpoint 的角平分线方向，然后在每个点附近采样约 `48x48` 输入像素足迹的有向局部 patch。patch 会按解剖方向摆正后进入 projection head，只拉近“同一图像、同一解剖点、两种轻增强视图”的 embedding：

```text
A_view1 <-> A_view2
L_view1 <-> L_view2
R_view1 <-> R_view2
```

A/L/R 三个点不会互相当作正样本。第二阶段使用 `data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded`，即人工三点训练集加上排除 LDP holdout 后随机抽取的 200 张 LDP `混杂图片` blackpad 空标签负样本；非混杂 LDP pseudo-positive 不进入训练。第三阶段使用 `data/yolo_pose_mixed_negative_60_blackpad`。

先做快速检查：

```bash
python tools/pretrain_oriented_contrast.py \
  --config configs/pretrain_oriented_keypoint_contrast_manual_only_y11m.yaml \
  --dry-run
```

构造混杂负样本数据集：

```bash
python tools/add_negative_images_to_yolo_pose.py \
  --base-dataset data/yolo_pose \
  --negative-source-dir /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed/混杂图片 \
  --out-dir data/yolo_pose_mixed_negative_ldp200_blackpad_holdout_excluded \
  --count 200 \
  --seed 20260523 \
  --exclude-manifest data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl \
  --prefix ldp_mixed_negative_blackpad \
  --blackpad-negatives \
  --overwrite
```

完整跑法：

```bash
python tools/run_three_stage_oriented_mixed_reject_pipeline.py \
  --config configs/pipeline_three_stage_oriented_mixed_reject_y11m.yaml
```

旧 contrast 结果已清理，经验教训保留在 `docs/contrast_experiment_lessons_20260523.md`。历史 LDP pseudo contrast 不再作为当前推荐实验流程。

2026-05-23 三阶段 200 张 blackpad 混杂负样本实测结果见 `docs/three_stage_oriented_mixed_reject_evaluation_20260523.md`。结论：该版 native pose test 指标高于 no-contrast baseline，但 LDP holdout `混杂图片` false-positive 升到 8%，不应替代当前 V1.1 ROI 模型。

2026-05-24 额外测试 `96x96` 有向局部 patch，记录见 `docs/three_stage_oriented_mixed_reject_96px_evaluation_20260524.md`。结论：contrast pretrain loss 下降，但 native test 和 LDP holdout 基本不变，manual ROI postprocess 明显变差，不推荐替代 48px 实验或 V1.1。

2026-05-24 额外测试关键点特异长方形 patch：A 为 `48x72`，L/R 为 `72x48`，记录见 `docs/three_stage_oriented_mixed_reject_rect48x72_evaluation_20260524.md`。结论：训练验证指标与 96px 接近，但 manual ROI postprocess 和 LDP 混杂误接收没有改善。

2026-05-24 继续测试更小的长方形 patch：A 为 `36x60`，L/R 为 `60x36`，记录见 `docs/three_stage_oriented_mixed_reject_rect36x60_evaluation_20260524.md`。结论：Stage 1 contrast loss 更弱，后续 native/manual/LDP 结果仍与 48x72 版重合，不推荐继续沿单纯框尺寸方向试。

历史注记：LDP pseudo-positive contrast 流程已废弃，旧结果目录已清理；经验教训统一保留在 `docs/contrast_experiment_lessons_20260523.md`。当时的 `0.4` 只用于筛选哪些 LDP 非混杂图片可进入 pseudo-positive 数据集，不是最终 ROI 算法阈值；最终 reject/manual_review 阈值仍来自 `configs/postprocess.yaml`，当前 `review_threshold: 0.30`。

`data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl` 从 LDP 八类各固定抽 100 张，只用于最终评估，不进入 pseudo-positive、mixed negative 或 hard-negative 训练。

## DINOv3 Three-Point Auxiliary Judgement

实验分支 `exp/dinov3-keypoint-aux` 用 DINOv3 辅助判定三点，替代继续加码 keypoint-local contrast。该设计冻结 DINOv3 dense encoder，只训练一个小 head：

```text
image -> frozen DINOv3 dense features
      -> oriented point-region head: background / anterior / left posterior / right posterior
```

左后方和右后方不被当作彼此的正样本；病变造成的不对称性应被保留。DINOv3 只提供高维局部语义特征，三点监督仍来自人工 YOLO-Pose 标签。当前 active 版本只训练点区域四分类，不再训练 triplet head 或 image reject head；hard negative 来自排除 holdout 后的混杂图误检点和关键点邻域 near-miss 背景。DINO 训练/打分改为看裁掉已有黑边后的 no-black/cropped 图像；prediction JSONL 会优先提供 `dinov3_source`/`cropped_source`，DINO 评分时把 YOLO padded 坐标减去 `preprocess.padding_px` 后再采样。点区域 head 会同时采样一张有效像素 mask，让 48x48 有向局部 patch 中的 letterbox padding、超出真实图像边界的位置，以及可选 luma-floor 黑区不作为正常组织特征参与判断。

默认配置用 `timm/vit_small_patch16_dinov3.lvd1689m` 这份公开 DINOv3 ViT-S/16 权重。官方 `facebook/dinov3-vits16-pretrain-lvd1689m` 仓库是 gated；如果已经获得访问权限，也可以把 `dinov3.backend` 改成 `transformers` 并使用官方模型 id。

快速检查：

```bash
python tools/train_dinov3_keypoint_aux.py \
  --config configs/train_dinov3_keypoint_aux_y11m.yaml \
  --dry-run
```

训练 auxiliary head：

```bash
python tools/train_dinov3_keypoint_aux.py \
  --config configs/train_dinov3_keypoint_aux_y11m.yaml
```

给已有推理结果追加 DINOv3 分数：

```bash
python tools/score_predictions_with_dinov3_aux.py \
  --aux-checkpoint Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200/weights/best_aux_head.pt \
  --predictions Results/predictions/predictions.jsonl \
  --out Results/predictions/predictions_dinov3_aux.jsonl
```

默认只追加 `dinov3_aux` 字段，不改变原来的 `final_confidence/action`。需要让 auxiliary score 修改置信度时，再加 `--apply-confidence-gate`。设计说明见 `docs/dinov3_keypoint_aux_design_20260524.md`。
当前 gate 会把极低 DINO 分作为拒绝信号：`point_region_score < 0.10` 标为 `reject_or_relabel`，`0.10-0.30` 不改变原置信度，`0.30-0.60` 按比例从 `1.0x` 奖励到 `1.5x`，`>=0.60` 作为极高分 DINO 直接通过候选。2026-05-24 评估见 `docs/dinov3_point_region_hardneg_evaluation_20260524.md`。

## 推理

当前项目主 ROI 模型为 `vf_roi_current`：先用当前 YOLO-Pose 权重生成三点/旋转 ROI，再用 DINOv3 point-region auxiliary head 对三点做语义打分并默认应用 confidence gate。这个别名当前指向 2026-05-25 refreshed-holdout 组合：`mixedneg120_blackpad` YOLO-Pose checkpoint + `gt396_seed20260525` DINOv3 auxiliary head。

```text
Results/models/vf_roi_current/pose_best.pt
Results/models/vf_roi_current/aux_best.pt
Results/models/vf_roi_current/postprocess.yaml
```

```bash
python tools/predict_current_roi.py \
  --source data/yolo_pose/images/val \
  --out Results/predictions/val_predictions_dinov3.jsonl \
  --batch 32 \
  --preprocess-workers 8
```

`tools/predict_current_roi.py` 会保留一份中间 YOLO-Pose JSONL（默认 `<out_stem>_pose_raw.jsonl`），再调用 `tools/score_predictions_with_dinov3_aux.py --apply-confidence-gate` 写出最终 DINO-gated JSONL。该入口默认不保存 pre-crop/no-black/blackpad 中间图；DINO scorer 会根据 JSONL 中的 `pre_crop` 和 `preprocess.crop_bbox_xyxy` 元数据从原图现场重建 no-black/cropped 输入，手机翻拍屏幕图也会先重放 screen-photo pre-crop。只有显式传入 `--save-intermediates` 时，才把这些中间图写入磁盘。

V1.2 的预测前处理会先检测外沿是否存在大面积近黑色干扰区。若触发，会从图像边缘连通区域中同时移除近黑色边框/背景和灰白色角落/UI 区域，再按低亮度前景框裁掉残余黑边，最后把裁后图按长边 30%、最少 80 px 加四周黑边；这个 black pad 是 YOLO-Pose 的必经输入步骤。默认情况下，这些 pre-crop/no-black/blackpad 中间图只在内存中生成并直接送入 YOLO，避免大量磁盘写图。JSONL 中 `source` 为 `memory://...` 引用，`original_source` 保留原图路径，`preprocess.type` 固定为 `crop_black_border_then_blackpad`，并记录原图尺寸、最终 `crop_bbox_xyxy`、是否实际裁剪、裁后尺寸、`padding_px`、padding 规则、`model_input_width/height`、`no_black_bbox_in_model_input`，以及 `edge_artifact_crop_*` 外沿黑边检测诊断字段。没有大面积外沿黑边的图会跳过该步，只保留原来的低亮度去黑边行为。DINOv3 scorer 会用这些元数据重建裁后图；如果后续要审查中间图、复现旧落盘流程，或强制让 scorer 读取已保存文件，可加 `--save-intermediates`。这时黑边图写到 `<out_stem>_blackpad_inputs/`，裁后 no-black 图写到 `<out_stem>_cropped_inputs/`，预裁切图写到 `<out_stem>_precrop_inputs/`，JSONL 中 `source`/`dinov3_source`/`cropped_source` 会恢复为实际文件路径。

从 V1.2 开始，`tools/predict_roi.py` 还会在外沿黑边/残余黑边裁剪 + blackpad 之前自动检测并对手机翻拍屏幕图片做预裁剪（screen-photo pre-crop）。检测逻辑通过固定色条纹（stripe_col/stripe_row > 0.24）和蓝色区域（blue_col/blue_row > 0.06）判断是否为翻拍图；若触发则先裁出喉镜窗口区域（tissue bbox → window frame → trim UI edges），再把裁后图送进黑边裁剪和加黑边流程。每条 JSONL 输出的 `pre_crop` 字段记录触发状态（`triggered`）、模式（`mode`）、触发原因（`reason`）、裁剪框（`box_xyxy`）、信号值（`signals`）、原始尺寸和预裁后尺寸；同一份信息也写入 `preprocess.pre_crop`，方便连同 V1.2 前处理一起审查。未触发的图片 `pre_crop.triggered` 为 `false`、`mode` 为 `"none"`。

可复用检测和裁剪逻辑位于 `yoloposevf/screen_photo_crop.py`，提供 `classify_screen_photo()` 和 `crop_screen_photo_window()` 两个公共接口。独立批量预裁剪预览（含 contact sheet）仍可通过 `scripts/crop_527_xianlin.py` 使用。
目录或列表输入默认以 `--batch 16` 分块送入 YOLO 批量推理，避免逐张调用模型导致 GPU 空转；单张 `--source` 会自动走同一入口并退化为 1 张。`--preprocess-workers N` 可并行执行 screen-photo pre-crop、去黑边和 blackpad 前处理，`0` 或 `1` 表示串行。需要完全复现旧逐张落盘路径时可传 `--batch 1 --preprocess-workers 0 --save-intermediates`。`--tta` 仍为逐张多增强聚合路径。YOLO-Pose 推理默认使用 GPU 0；只有明确需要 CPU 时才传 `--device cpu`。

固定 JSONL holdout/manifest 推理可用 `--manifest`，脚本会优先读取每行的 `original_source`、`source_key`、`source`：

```bash
python tools/predict_current_roi.py \
  --manifest data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl \
  --out Results/evaluation/vf_roi_current_ldp_holdout/predictions_dinov3.jsonl \
  --device 0 \
  --imgsz 960

python tools/summarize_ldp_holdout_predictions.py \
  --manifest data/ldp_holdout/ldp_holdout_100_per_class_seed20260523.jsonl \
  --predictions Results/evaluation/vf_roi_current_ldp_holdout/predictions_dinov3.jsonl \
  --out-dir Results/evaluation/vf_roi_current_ldp_holdout
```

每条输出包含：

- `bbox_yolo`
- `bbox_keypoints`
- `original_source`
- `dinov3_source` / `cropped_source`
- `preprocess`
- `roi_polygon`: 由 3 点生成的角平分线旋转 ROI
- `final_box_polygon`: 最终四点旋转框；一条边平行于前联合夹角的角平分线，左右宽度按两侧后方点到角平分线的投影分别扩张
- `final_bbox_xyxy` / `final_bbox`: 最终旋转框的 axis-aligned 外接矩形，仅用于兼容传统 bbox 评估
- `usable_box_polygon`: 置信度过线时可用；低置信度时为 `null`
- `usable_bbox`: `usable_box_polygon` 的 axis-aligned 外接矩形兼容字段
- `glottic_angle_degrees`
- `geometry_score`
- `consistency_score`
- `roi_area_denominator` / `roi_area_denominator_mode` / `effective_image_bbox`
- `roi_area_ratio` / `roi_area_factor`
- `dinov3_aux`: DINOv3 point-region auxiliary scores and gate metadata
- `pre_dinov3_aux_confidence` / `dinov3_aux_gate_action`: DINO gate applied to the YOLO-Pose confidence/action
- `max_keypoint_outside_image_px` / `image_bounds_factor`
- `roi_dark_fraction` / `roi_dark_factor`
- `final_confidence`
- `action`: `auto_accept`, `manual_review`, `reject_or_relabel`
- `flags`: 低置信度或几何不一致原因

如果需要复现 V1 训练时的输入分布，可先把独立复制出的图片树加黑边，再对加黑边后的图片跑
`tools/predict_roi.py` 和裁剪。默认规则是按图片长边的 30% 加四周黑边，且不少于 80 px：

```bash
python tools/blackpad_image_tree.py \
  --source-root Results/ldp_8class_v1_blackpad_crop/input_copy_original \
  --out-dir Results/ldp_8class_v1_blackpad_crop/input_blackpad \
  --fraction 0.30 \
  --min-padding 80
```

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
如果需要生成完整的下游分类数据集，可同时裁剪 `auto_accept`/`manual_review`，并把 `reject_or_relabel`
保留为原图：

```bash
python tools/crop_rois_from_predictions.py \
  --predictions Results/predictions/ldp_8class_predictions.jsonl \
  --source-root Results/predictions/ldp_8class_input_links \
  --out-dir Results/predictions/ldp_8class_crop_or_original \
  --crop-actions auto_accept manual_review \
  --copy-original-actions reject_or_relabel \
  --fallback-original-on-crop-failure \
  --crop-mode polygon
```

如果只想让特定类别的 `reject_or_relabel` 保留原图，例如八分类训练集中仅保留 `混杂图片` 的 reject，
可加 `--copy-original-classes`，其他类别的 reject 会跳过：

```bash
python tools/crop_rois_from_predictions.py \
  --predictions Results/predictions/ldp_8class_predictions.jsonl \
  --source-root Results/predictions/ldp_8class_blackpad_inputs \
  --out-dir Results/predictions/ldp_8class_nonvoc_reject_original \
  --crop-actions auto_accept manual_review \
  --copy-original-actions reject_or_relabel \
  --copy-original-classes 混杂图片 \
  --fallback-original-on-crop-failure \
  --crop-mode polygon
```

如果分类模型需要统一看到正方形输入，可把所有 action 都尝试按预测框裁剪，并把裁剪图或无框 fallback
输入拉伸到固定正方形。黑边图只用于 ROI 定位；默认 fallback 使用 `preprocessed_source`，即优先返回
`cropped_source`/`dinov3_source` 对应的预裁剪、去黑边喉镜窗口，缺失时再退到 `original_source`，最后才退到
黑边 `source`。这样手机翻拍屏幕图在 ROI 裁剪失败时也不会把完整手机照片送入下游分类器：

```bash
python tools/crop_rois_from_predictions.py \
  --predictions Results/predictions/ldp_8class_predictions.jsonl \
  --source-root Results/predictions/ldp_8class_blackpad_inputs \
  --out-dir Results/predictions/ldp_8class_square224_box_or_original \
  --crop-actions auto_accept manual_review reject_or_relabel \
  --fallback-original-on-crop-failure \
  --output-size 224 \
  --crop-mode polygon
```

如果需要复现旧的“ROI 失败后回到未加黑边原图”口径，可显式传入 `--copy-original-source original_source`。
DINOv3 打分阶段还会把 `dinov3_keypoints_outside_cropped_image` 作为不可用 ROI 信号：一旦关键点越出
预处理后的 no-black/cropped 图，最终 action 会降级为 `reject_or_relabel`，交由上述 fallback 输入处理。

训练完 4-class 分类模型后，可用下列工具在外部文件夹数据集上评估单个 checkpoint：

```bash
python tools/evaluate_four_class_checkpoint.py \
  --shared-py /path/to/model_variants/main/图像识别/shared.py \
  --config /path/to/model_variants/main/图像识别/config_phase2.json \
  --checkpoint /path/to/phase2_best_model.pth \
  --dataset-root Results/external_larynxdata_data_v1_crop/cropped_auto_manual_reject_original \
  --out-dir Results/external_eval
```

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
