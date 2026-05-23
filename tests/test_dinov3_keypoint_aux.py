from __future__ import annotations

import torch

from yoloposevf.dinov3_aux import (
    DinoV3KeypointAuxHead,
    build_point_targets,
    build_triplet_targets,
    sample_point_features,
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
    points, labels, valid = build_point_targets(keypoints, mask, background_points=2)
    assert points.shape == (1, 5, 2)
    assert labels[0, :3].tolist() == [1, 2, 3]
    assert valid.all()
    triplets, triplet_labels, owners = build_triplet_targets(keypoints, mask, corrupted_triplets=3)
    assert triplets.shape == (4, 3, 2)
    assert triplet_labels.tolist() == [1, 0, 0, 0]
    assert owners.tolist() == [0, 0, 0, 0]


def test_aux_head_scores_triplet_shapes() -> None:
    feature_map = torch.randn(2, 8, 4, 4)
    global_feature = feature_map.mean(dim=(2, 3))
    triplets = torch.tensor(
        [
            [[0.5, 0.2], [0.3, 0.7], [0.7, 0.7]],
            [[0.4, 0.3], [0.2, 0.6], [0.8, 0.6]],
        ],
        dtype=torch.float32,
    )
    head = DinoV3KeypointAuxHead(8, point_hidden_dim=16, triplet_hidden_dim=32, image_hidden_dim=16)
    sampled = sample_point_features(feature_map, triplets)
    assert sampled.shape == (2, 3, 8)
    scores = score_aux_triplet(head, feature_map, global_feature, triplets)
    assert scores["point_expected_probs"].shape == (2, 3)
    assert scores["triplet_valid_prob"].shape == (2,)
    assert scores["image_reject_prob"].shape == (2,)
    assert scores["confidence_factor"].shape == (2,)
