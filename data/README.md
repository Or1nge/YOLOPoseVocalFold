# 数据目录

当前仓库不包含真实图片或标注。

通用放置方式：

```text
data/
  images/      # 原始喉镜图片
  labelme/     # 与图片对应的 LabelMe JSON
  yolo_pose/   # 转换脚本生成，默认不提交
  splits/      # 可选：按 split 镜像原始图片/JSON，默认不提交
```

当前黑边增强标注集放在
`data/glottic_roi_rectangle_annotation_200_20260520_blackpad/images/`，图片和 LabelMe JSON 同目录。实际可用配对样本为 196 个；`039`, `057`, `100`, `124` 在 manifest 中有记录但当前目录没有图片/JSON。

转换后 YOLO-Pose 数据集结构：

```text
data/yolo_pose/
  images/{train,val,test}/
  labels/{train,val,test}/
  roi_polygons/{train,val,test}.jsonl
  vocal_fold_pose.yaml
  conversion_manifest.json
  validation_report.json
```

`roi_polygons` 保存人工 `声门区域` 旋转框四角，用于评价三点角平分线生成 ROI 的 87% 覆盖率与面积比例。

负样本数据集约定：

```text
data/yolo_pose_mixed_negative_60/
  images/train/mixed_negative_*.jpg
  labels/train/mixed_negative_*.txt  # 空文件，表示背景/无声门 ROI
  negative_samples_manifest.{csv,json}
```

这些负样本只从 LDP `混杂图片` 只读复制，不修改 LDP 原目录。

LDP pseudo 微调数据集约定：

```text
data/yolo_pose_ldp_pseudo_mixedpenalty_copy/
  images/train/ldp_pos_*.jpg        # 非混杂 auto_accept/manual_review pseudo-positive
  labels/train/ldp_pos_*.txt        # 由预测三点和 ROI envelope 生成的 YOLO-Pose 标签
  images/train/ldp_neg_*.jpg        # 全部混杂图片负样本
  labels/train/ldp_neg_*.txt        # 空文件
  ldp_pseudo_manifest.{csv,json}
```

`copy` 版本会把 LDP/input-links 图片复制到本项目目录并设为用户可写，避免 Ultralytics 检查 JPEG 时尝试修复只读源文件。
