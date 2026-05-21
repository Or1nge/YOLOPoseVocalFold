from yoloposevf.geometry import (
    ImageSize,
    angle_bisector_roi_from_three_points,
    bbox_iou,
    containment_rate,
    keypoint_bbox,
    polygon_containment_rate,
)
from yoloposevf.postprocess import PosePrediction, PostprocessConfig, fuse_prediction


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
