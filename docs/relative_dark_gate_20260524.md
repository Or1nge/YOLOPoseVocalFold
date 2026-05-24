# Relative ROI dark-region gate, 2026-05-24

The ROI dark-region gate now supports exposure-relative thresholding through:

```yaml
roi_dark_mode: relative_foreground_median
roi_dark_relative_luma_ratio: 0.80
roi_dark_foreground_luma_floor: 8.0
```

The old absolute threshold field is retained for reproducibility and configs
that explicitly set `roi_dark_mode: absolute`.

## Rationale

The V1.1 pipeline adds synthetic black borders before YOLO-Pose inference. A
fixed absolute luminance threshold can be skewed by image exposure and black
padding. In `relative_foreground_median` mode, pixels at or below
`roi_dark_foreground_luma_floor` are excluded when estimating the image
foreground reference luminance. The effective dark threshold is then:

```text
foreground_median_luma * roi_dark_relative_luma_ratio
```

Prediction JSONL rows include `roi_dark_mode`,
`roi_dark_effective_luma_threshold`, and `roi_dark_reference_luma` for audit.
