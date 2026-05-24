from yoloposevf.geometry import (
    ImageSize,
    angle_bisector_roi_from_three_points,
    bbox_iou,
    containment_rate,
    geometry_score,
    keypoint_bbox,
    included_angle_degrees,
    polygon_keypoint_containment_rate,
    polygon_containment_rate,
)
from yoloposevf.postprocess import PosePrediction, PostprocessConfig, fuse_prediction
from tools.predict_roi import effective_area_from_metadata, polygon_dark_fraction


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def test_keypoint_bbox_expands_and_clips() -> None:
    image_size = ImageSize(width=100, height=80)
    bbox = keypoint_bbox(
        [(20, 20, 2), (40, 20, 2), (20, 40, 2), (40, 40, 2)],
        margin_x=0.1,
        margin_y=0.2,
        image_size=image_size,
    )
    assert bbox == (18.0, 16.0, 42.0, 44.0)


def test_iou_and_containment() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert containment_rate((0, 0, 10, 10), [(1, 1), (11, 1)]) == 0.5


def test_fuse_prediction_outputs_action_and_final_bbox_contains_keypoints() -> None:
    prediction = PosePrediction(
        bbox=(10, 10, 50, 50),
        bbox_conf=0.95,
        keypoints=((30, 20, 0.9), (20, 60, 0.9), (50, 60, 0.9)),
        image_size=ImageSize(width=100, height=100),
        source="sample.png",
    )
    output = fuse_prediction(prediction, PostprocessConfig(auto_accept_threshold=0.1))
    assert output["action"] == "auto_accept"
    assert output["containment_rate"] == 1.0
    assert output["roi_polygon"] is not None
    assert output["final_bbox"][0] <= 20
    assert output["final_bbox"][2] >= 50


def test_angle_bisector_roi_contains_target_polygon() -> None:
    roi = angle_bisector_roi_from_three_points(
        [(50, 80, 2), (25, 20, 2), (75, 20, 2)],
        base_backtrack_fraction=0.10,
        posterior_margin_fraction=0.12,
        side_margin_fraction=0.12,
    )
    target = ((35, 75), (65, 75), (78, 22), (22, 22))
    assert polygon_containment_rate(target, roi.polygon) >= 0.95


def test_angle_bisector_roi_allows_asymmetric_lateral_widths() -> None:
    keypoints = [(50, 80, 2), (25, 20, 2), (100, 70, 2)]
    roi = angle_bisector_roi_from_three_points(
        keypoints,
        base_backtrack_fraction=0.10,
        posterior_margin_fraction=0.12,
        side_margin_fraction=0.0,
        min_side_margin_px=0.0,
    )

    left_width = -_dot(_sub(roi.polygon[0], roi.base_center), roi.normal)
    right_width = _dot(_sub(roi.polygon[1], roi.base_center), roi.normal)
    parallel_edge = _sub(roi.polygon[2], roi.polygon[1])

    assert abs(left_width - right_width) > 5.0
    assert abs(_cross(parallel_edge, roi.direction)) < 1e-6
    assert polygon_keypoint_containment_rate(roi.polygon, [point[:2] for point in keypoints]) == 1.0


def test_three_point_geometry_rejects_implausibly_small_angle() -> None:
    image_size = ImageSize(width=100, height=100)
    nearly_collinear = [(50, 80, 2), (52, 20, 2), (53, 10, 2)]
    plausible = [(50, 80, 2), (25, 20, 2), (75, 20, 2)]

    assert included_angle_degrees(nearly_collinear[0], nearly_collinear[1], nearly_collinear[2]) < 20.0
    assert geometry_score(nearly_collinear, (20, 10, 80, 90), image_size) == 0.0
    assert geometry_score(plausible, (20, 10, 80, 90), image_size) > 0.0


def test_confidence_power_curve_preserves_gamma_behavior() -> None:
    prediction = PosePrediction(
        bbox=(20, 20, 80, 80),
        bbox_conf=0.60,
        keypoints=((50, 70, 0.90), (30, 30, 0.90), (70, 30, 0.90)),
        image_size=ImageSize(width=100, height=100),
        source="sample.png",
    )

    linear = fuse_prediction(prediction, PostprocessConfig(confidence_curve="power", confidence_gamma=1.0))
    squared = fuse_prediction(prediction, PostprocessConfig(confidence_curve="power", confidence_gamma=2.0))

    assert squared["bbox_confidence_factor"] == prediction.bbox_conf**2
    assert squared["keypoint_confidence_factor"] < linear["keypoint_confidence_factor"]
    assert squared["final_confidence"] < linear["final_confidence"]


def test_confidence_tanh_curve_sharpens_without_squaring_high_confidence() -> None:
    low = PosePrediction(
        bbox=(20, 20, 80, 80),
        bbox_conf=0.40,
        keypoints=((50, 70, 0.40), (30, 30, 0.40), (70, 30, 0.40)),
        image_size=ImageSize(width=100, height=100),
        source="low.png",
    )
    high = PosePrediction(
        bbox=(20, 20, 80, 80),
        bbox_conf=0.90,
        keypoints=((50, 70, 0.90), (30, 30, 0.90), (70, 30, 0.90)),
        image_size=ImageSize(width=100, height=100),
        source="high.png",
    )
    cfg = PostprocessConfig(
        confidence_curve="tanh",
        confidence_tanh_midpoint=0.65,
        confidence_tanh_steepness=6.0,
    )

    low_out = fuse_prediction(low, cfg)
    high_out = fuse_prediction(high, cfg)

    assert low_out["bbox_confidence_factor"] < 0.40
    assert high_out["bbox_confidence_factor"] > 0.90
    assert low_out["final_confidence"] < high_out["final_confidence"]


def test_tiny_three_point_roi_can_force_rejection() -> None:
    prediction = PosePrediction(
        bbox=(450, 450, 550, 550),
        bbox_conf=0.95,
        keypoints=((500, 530, 0.99), (470, 470, 0.99), (530, 470, 0.99)),
        image_size=ImageSize(width=1000, height=1000),
        source="tiny.png",
    )

    output = fuse_prediction(
        prediction,
        PostprocessConfig(
            confidence_gamma=2.0,
            min_roi_area_ratio=0.03,
            good_roi_area_ratio=0.08,
        ),
    )

    assert output["roi_area_ratio"] < 0.03
    assert output["roi_area_factor"] == 0.0
    assert output["action"] == "reject_or_relabel"
    assert output["usable_box_polygon"] is None
    assert "roi_area_too_small" in output["flags"]


def test_roi_area_ratio_can_use_effective_image_area_denominator() -> None:
    prediction = PosePrediction(
        bbox=(450, 450, 550, 550),
        bbox_conf=0.95,
        keypoints=((500, 530, 0.99), (470, 470, 0.99), (530, 470, 0.99)),
        image_size=ImageSize(width=1000, height=1000),
        source="tiny.png",
        effective_image_area=10_000.0,
        effective_image_bbox=(450.0, 450.0, 550.0, 550.0),
        effective_image_area_mode="foreground_bbox",
    )

    output = fuse_prediction(
        prediction,
        PostprocessConfig(
            confidence_gamma=2.0,
            min_roi_area_ratio=0.03,
            good_roi_area_ratio=0.08,
        ),
    )

    assert output["roi_area_denominator"] == 10_000.0
    assert output["roi_area_denominator_mode"] == "foreground_bbox"
    assert output["effective_image_bbox"] == [450.0, 450.0, 550.0, 550.0]
    assert output["roi_area_ratio"] > 0.08
    assert output["roi_area_factor"] == 1.0
    assert "roi_area_too_small" not in output["flags"]


def test_keypoint_outside_image_bounds_forces_rejection() -> None:
    prediction = PosePrediction(
        bbox=(20, 20, 80, 80),
        bbox_conf=0.95,
        keypoints=((50, 70, 0.99), (30, 30, 0.99), (70, 106, 0.99)),
        image_size=ImageSize(width=100, height=100),
        source="outside.png",
    )

    output = fuse_prediction(
        prediction,
        PostprocessConfig(
            confidence_gamma=2.0,
            keypoint_image_bounds_tolerance_px=5.0,
        ),
    )

    assert output["max_keypoint_outside_image_px"] == 6.0
    assert output["image_bounds_factor"] == 0.0
    assert output["action"] == "reject_or_relabel"
    assert output["usable_box_polygon"] is None
    assert "keypoints_outside_image" in output["flags"]


def test_anterior_point_y_order_does_not_force_rejection() -> None:
    prediction = PosePrediction(
        bbox=(20, 20, 80, 80),
        bbox_conf=0.95,
        keypoints=((50, 30, 0.99), (30, 70, 0.99), (70, 70, 0.99)),
        image_size=ImageSize(width=100, height=100),
        source="inverted.png",
    )

    output = fuse_prediction(
        prediction,
        PostprocessConfig(
            confidence_gamma=2.0,
        ),
    )

    assert output["final_confidence"] > 0.0
    assert "anterior_point_not_below_posterior_points" not in output["flags"]
    assert "weak_anterior_posterior_orientation" not in output["flags"]


def test_relative_dark_gate_ignores_synthetic_black_border(tmp_path) -> None:
    from PIL import Image

    image = Image.new("L", (10, 10), 0)
    for y in range(2, 8):
        for x in range(2, 8):
            image.putpixel((x, y), 100)
    for y in range(3, 5):
        for x in range(3, 6):
            image.putpixel((x, y), 60)
    path = tmp_path / "blackpad_sample.png"
    image.save(path)

    dark_fraction, threshold, reference_luma = polygon_dark_fraction(
        path,
        [(2, 2), (7, 2), (7, 7), (2, 7)],
        75.0,
        mode="relative_foreground_median",
        relative_luma_ratio=0.70,
        foreground_luma_floor=8.0,
    )

    assert reference_luma == 100.0
    assert threshold == 70.0
    assert dark_fraction is not None
    assert 0.0 < dark_fraction < 1.0


def test_effective_area_from_blackpad_metadata_uses_original_foreground_bbox(tmp_path) -> None:
    from PIL import Image

    image = Image.new("L", (10, 10), 0)
    for y in range(3, 9):
        for x in range(2, 8):
            image.putpixel((x, y), 120)
    path = tmp_path / "black_border_sample.png"
    image.save(path)

    area, bbox, mode = effective_area_from_metadata(
        {
            "original_source": str(path),
            "preprocess": {
                "type": "blackpad",
                "padding_px": 5,
                "original_width": 10,
                "original_height": 10,
            },
        },
        PostprocessConfig(roi_dark_foreground_luma_floor=8.0),
    )

    assert area == 36.0
    assert bbox == (7.0, 8.0, 13.0, 14.0)
    assert mode == "blackpad_foreground_bbox"
