import torch
import pytest

from yoloposevf.keypoint_contrast import (
    KeypointProjectionHead,
    OrientedPatchProjectionHead,
    first_keypoints_per_image,
    keypoint_local_contrast_loss,
    make_light_augmented_view,
    oriented_keypoint_patch_contrast_loss,
    sample_background_features,
    sample_local_features,
    sample_oriented_local_patches,
)


def test_first_keypoints_per_image_keeps_visible_points() -> None:
    keypoints = torch.tensor(
        [
            [[0.2, 0.3, 2.0], [0.4, 0.5, 0.0], [0.6, 0.7, 2.0]],
            [[0.1, 0.2, 2.0], [0.3, 0.4, 2.0], [0.5, 0.6, 2.0]],
        ]
    )
    batch_idx = torch.tensor([1, 1])

    points, mask = first_keypoints_per_image(keypoints, batch_idx, batch_size=2, num_keypoints=3)

    assert mask.tolist() == [[False, False, False], [True, False, True]]
    assert points[1, 0].tolist() == pytest.approx([0.2, 0.3])


def test_sample_local_features_uses_normalized_keypoint_coordinates() -> None:
    feature_map = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    keypoints = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    mask = torch.tensor([[True]])

    sampled, sampled_mask = sample_local_features(feature_map, keypoints, mask, patch_radius=0)

    assert sampled.shape == (1, 1, 1)
    assert sampled_mask.tolist() == [[True]]
    assert sampled.item() == 7.5


def test_light_augmented_view_preserves_shapes_and_masks_outside_points() -> None:
    torch.manual_seed(1)
    images = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    keypoints = torch.tensor([[[0.5, 0.5], [0.99, 0.99], [0.2, 0.8]]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False]])

    augmented, transformed, transformed_mask = make_light_augmented_view(
        images,
        keypoints,
        mask,
        degrees=0.0,
        scale=0.0,
        translate=0.5,
        brightness=0.0,
        contrast=0.0,
    )

    assert augmented.shape == images.shape
    assert transformed.shape == keypoints.shape
    assert transformed_mask.shape == mask.shape
    assert transformed_mask[0, 2].item() is False


def test_background_sampling_avoids_visible_keypoint_neighborhood() -> None:
    torch.manual_seed(2)
    feature_map = torch.randn(1, 4, 8, 8)
    keypoints = torch.tensor([[[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])

    negatives = sample_background_features(
        feature_map,
        keypoints,
        mask,
        negative_count=5,
        negative_min_distance=0.1,
        patch_radius=0,
    )

    assert negatives.shape == (5, 4)


def test_keypoint_local_contrast_loss_is_differentiable() -> None:
    torch.manual_seed(3)
    features1 = torch.randn(2, 6, 10, 10, requires_grad=True)
    features2 = features1.detach().clone().requires_grad_(True)
    keypoints = torch.tensor(
        [
            [[0.3, 0.3], [0.6, 0.4], [0.5, 0.7]],
            [[0.25, 0.4], [0.7, 0.5], [0.45, 0.75]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((2, 3), dtype=torch.bool)
    head = KeypointProjectionHead(6, hidden_dim=12, out_dim=5)

    loss = keypoint_local_contrast_loss(
        features1,
        features2,
        keypoints,
        keypoints,
        mask,
        head,
        temperature=0.2,
        patch_radius=1,
        negative_count=3,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert features1.grad is not None
    assert features2.grad is not None


def test_oriented_local_patches_preserve_canonical_shape() -> None:
    feature_map = torch.randn(1, 4, 12, 12)
    keypoints = torch.tensor([[[0.5, 0.5], [0.4, 0.3], [0.6, 0.3]]], dtype=torch.float32)
    mask = torch.ones((1, 3), dtype=torch.bool)
    directions = torch.tensor([[[0.0, -1.0]]], dtype=torch.float32).view(1, 2)

    patches, sampled_mask = sample_oriented_local_patches(
        feature_map,
        keypoints,
        mask,
        directions,
        patch_size_input=48,
        input_size=960,
        output_size=5,
    )

    assert sampled_mask.tolist() == [[True, True, True]]
    assert patches.shape == (1, 3, 4, 5, 5)


def test_oriented_local_patches_accept_per_keypoint_rectangles() -> None:
    feature_map = torch.randn(1, 4, 12, 12)
    keypoints = torch.tensor([[[0.5, 0.5], [0.4, 0.3], [0.6, 0.3]]], dtype=torch.float32)
    mask = torch.ones((1, 3), dtype=torch.bool)
    directions = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

    patches, sampled_mask = sample_oriented_local_patches(
        feature_map,
        keypoints,
        mask,
        directions,
        patch_size_input=[[48, 72], [72, 48], [72, 48]],
        input_size=960,
        output_size=5,
    )

    assert sampled_mask.tolist() == [[True, True, True]]
    assert patches.shape == (1, 3, 4, 5, 5)


def test_oriented_keypoint_patch_contrast_loss_is_differentiable() -> None:
    torch.manual_seed(4)
    features1 = torch.randn(2, 5, 12, 12, requires_grad=True)
    features2 = features1.detach().clone().requires_grad_(True)
    keypoints = torch.tensor(
        [
            [[0.5, 0.75], [0.35, 0.35], [0.65, 0.35]],
            [[0.5, 0.25], [0.35, 0.65], [0.65, 0.65]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((2, 3), dtype=torch.bool)
    head = OrientedPatchProjectionHead(5, patch_size=3, hidden_dim=16, out_dim=6)

    loss = oriented_keypoint_patch_contrast_loss(
        features1,
        features2,
        keypoints,
        keypoints,
        mask,
        head,
        temperature=0.2,
        patch_size_input=48,
        input_size=960,
        output_size=3,
        negative_count=2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert features1.grad is not None
    assert features2.grad is not None
