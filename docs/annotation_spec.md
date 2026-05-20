# 标注规范

本项目假设已经有约 200 张 LabelMe 标注图，每张图包含：

- 1 个矩形框：标签名默认 `vocal_fold_roi`
- 4 个点：标签名默认 `kp1`, `kp2`, `kp3`, `kp4`

默认点位顺序：

| 点 | 含义 |
| --- | --- |
| `kp1` | 左侧声带 anterior point |
| `kp2` | 左侧声带 posterior point |
| `kp3` | 右侧声带 anterior point |
| `kp4` | 右侧声带 posterior point |

如果实际标注采用不同名字或顺序，只改 `configs/keypoints.yaml`，不要混用多套规则。

## 训练前自动修正

转换脚本会检查人工框是否包含 4 个关键点。若关键点在框外，会用：

```text
union(manual_bbox, keypoint_bbox_with_margin)
```

生成训练框，避免 YOLO-Pose 同时收到“框在这里”和“点在框外”的矛盾信号。

## 翻转增强

主分支默认关闭水平/垂直翻转，因为翻转会改变左右或前后点位含义。只有在明确写好 `flip_idx` 和点位重排规则后，才建议打开。

