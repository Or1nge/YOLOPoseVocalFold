# 数据目录

当前仓库不包含真实图片或标注。

预期放置方式：

```text
data/
  images/      # 原始喉镜图片
  labelme/     # 与图片对应的 LabelMe JSON
  yolo_pose/   # 转换脚本生成，默认不提交
```

转换后 YOLO-Pose 数据集结构：

```text
data/yolo_pose/
  images/{train,val,test}/
  labels/{train,val,test}/
  vocal_fold_pose.yaml
  conversion_manifest.json
  validation_report.json
```

