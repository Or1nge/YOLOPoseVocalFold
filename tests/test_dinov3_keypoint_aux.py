from __future__ import annotations

import torch

from yoloposevf.dinov3_aux import (
    DinoV3KeypointAuxHead,
    build_point_targets,
    dinov3_confidence_gate,
    foreground_mask_from_images,
    sample_point_features,
    sample_oriented_point_regions,
    sample_oriented_point_region_masks,
    score_aux_triplet,
    tokens_to_feature_map,
)


def test_tokens_to_feature_map_drops_cls_token() -> None:
    tokens = torch.arange(2 * 17 * 3, dtype=torch.float32).view(2, 17, 3)
    feature_map = tokens_to_feature_map(tokens)
    assert feature_map.shape == (2, 3, 4, 4)


def test_tokens_to_feature_map_drops_dinov3_register_tokens() -> None:
    tokens = torch.arange(2 * 21 * 3, dtype=torch.float32).view(2, 21, 3)
    feature_map = tokens_to_feature_map(tokens)
    assert feature_map.shape == (2, 3, 4, 4)


def test_build_targets_do_not_pair_left_and_right_as_positives() -> None:
    keypoints = torch.tensor([[[0.5, 0.2], [0.3, 0.7], [0.7, 0.7]]])
    mask = torch.tensor([[True, True, True]])
    points, labels, valid, directions = build_point_targets(keypoints, mask, background_points=2)
    assert points.shape == (1, 5, 2)
    assert directions.shape == (1, 5, 2)
    assert labels[0, :3].tolist() == [1, 2, 3]
    assert valid.all()


def test_oriented_point_region_head_scores_shapes() -> None:
    feature_map = torch.randn(2, 8, 4, 4)
    global_feature = feature_map.mean(dim=(2, 3))
    triplets = torch.tensor(
        [
            [[0.5, 0.2], [0.3, 0.7], [0.7, 0.7]],
            [[0.4, 0.3], [0.2, 0.6], [0.8, 0.6]],
        ],
        dtype=torch.float32,
    )
    head = DinoV3KeypointAuxHead(8, patch_output_size=3, point_hidden_dim=16)
    sampled = sample_point_features(feature_map, triplets)
    assert sampled.shape == (2, 3, 8)
    directions = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]] * 2)
    patches = sample_oriented_point_regions(feature_map, triplets, directions, output_size=3)
    assert patches.shape == (2, 3, 8, 3, 3)
    scores = score_aux_triplet(head, feature_map, global_feature, triplets, input_size=64)
    assert scores["point_expected_probs"].shape == (2, 3)
    assert scores["confidence_factor"].shape == (2,)
    assert scores["direct_accept"].shape == (2,)
    assert scores["hard_reject"].shape == (2,)
    assert scores["valid_fraction"].shape == (2,)


def test_mask_aware_point_head_uses_valid_patch_mask() -> None:
    feature_map = torch.randn(1, 8, 4, 4)
    global_feature = feature_map.mean(dim=(2, 3))
    triplets = torch.tensor([[[0.10, 0.10], [0.3, 0.7], [0.7, 0.7]]], dtype=torch.float32)
    head = DinoV3KeypointAuxHead(8, patch_output_size=3, point_hidden_dim=16, include_valid_mask=True)
    images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    images[:, :, 16:64, 16:64] = 1.0
    valid_mask = foreground_mask_from_images(images, luma_floor=8.0)
    directions = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    patch_masks = sample_oriented_point_region_masks(valid_mask, triplets, directions, input_size=64, output_size=3)
    patches = sample_oriented_point_regions(feature_map, triplets, directions, input_size=64, output_size=3)
    logits = head.point_logits(patches, triplets, valid_mask=patch_masks)
    scores = score_aux_triplet(
        head,
        feature_map,
        global_feature,
        triplets,
        input_size=64,
        valid_mask_map=valid_mask,
    )
    assert logits.shape == (1, 3, 4)
    assert patch_masks.shape == (1, 3, 1, 3, 3)
    assert float(scores["valid_fraction"][0]) < 1.0


def test_reward_only_gate_has_no_hard_reject_when_threshold_is_disabled() -> None:
    scores = torch.tensor([0.00, 0.04, 0.29, 0.30, 0.45, 0.60])
    factors, direct_accept, hard_reject = dinov3_confidence_gate(
        scores,
        gate_mode="reward_only",
        reject_threshold=0.0,
        reward_threshold=0.30,
        direct_accept_threshold=0.60,
        reward_multiplier=1.50,
    )
    assert torch.allclose(factors, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.25, 1.5]))
    assert direct_accept.tolist() == [False, False, False, False, False, True]
    assert hard_reject.tolist() == [False, False, False, False, False, False]
