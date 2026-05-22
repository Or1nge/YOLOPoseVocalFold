import numpy as np
import pytest

from yoloposevf.containment_loss import containment_penalty_numpy


def test_containment_loss_is_zero_when_all_keypoints_are_inside() -> None:
    loss = containment_penalty_numpy(
        boxes_xyxy=[[0, 0, 10, 10]],
        keypoints_xy=[[[1, 1], [5, 5], [9, 9], [10, 10]]],
    )
    assert loss == 0.0


def test_containment_loss_penalizes_only_visible_outside_keypoints() -> None:
    loss = containment_penalty_numpy(
        boxes_xyxy=[[0, 0, 10, 10]],
        keypoints_xy=[[[-2, 5], [5, 12], [15, 5], [5, -3]]],
        visibility=[[1, 1, 0, 1]],
        normalize_by_box_size=False,
    )
    assert loss == pytest.approx((2**2 + 2**2 + 3**2) / 3)


def test_containment_loss_margin_and_unsorted_boxes() -> None:
    loss = containment_penalty_numpy(
        boxes_xyxy=[[10, 10, 0, 0]],
        keypoints_xy=[[[-0.4, 5], [5, 10.3]]],
        margin=0.5,
        normalize_by_box_size=False,
    )
    assert loss == 0.0


def test_containment_loss_none_reduction_returns_per_sample_values() -> None:
    losses = containment_penalty_numpy(
        boxes_xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]],
        keypoints_xy=[
            [[1, 1], [2, 2]],
            [[-10, 5], [5, 20]],
        ],
        reduction="none",
    )
    assert isinstance(losses, np.ndarray)
    assert losses.tolist() == [0.0, 1.0]


def test_containment_loss_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="expected boxes shape"):
        containment_penalty_numpy([[0, 0, 10, 10]], [[[1, 1]], [[2, 2]]])
