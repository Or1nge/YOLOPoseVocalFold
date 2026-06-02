from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from tools.score_predictions_with_dinov3_aux import (
    load_aux_config_from_checkpoint,
    maybe_apply_gate,
    resolve_dinov3_prediction_input,
)

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


def test_oriented_patch_mask_marks_out_of_image_samples_invalid() -> None:
    valid_mask = torch.ones((1, 1, 64, 64), dtype=torch.float32)
    triplets = torch.tensor([[[0.02, 0.02], [0.50, 0.50], [0.70, 0.70]]], dtype=torch.float32)
    directions = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])

    patch_masks = sample_oriented_point_region_masks(
        valid_mask,
        triplets,
        directions,
        patch_size_input=48.0,
        input_size=64,
        output_size=5,
    )

    assert float(patch_masks[0, 0].mean()) < 1.0
    assert float(patch_masks[0, 1].mean()) == 1.0


def test_resolve_dinov3_input_prefers_cropped_source_and_transforms_keypoints(tmp_path) -> None:
    source = tmp_path / "source_blackpad.png"
    cropped = tmp_path / "source_cropped.png"
    Image.new("RGB", (90, 80), color=(0, 0, 0)).save(source)
    Image.new("RGB", (50, 40), color=(180, 120, 90)).save(cropped)
    record = {
        "source": str(source),
        "dinov3_source": str(cropped),
        "preprocess": {
            "type": "crop_black_border_then_blackpad",
            "padding_px": 20,
            "cropped_width": 50,
            "cropped_height": 40,
            "no_black_width": 50,
            "no_black_height": 40,
        },
        "keypoints": [[25, 30, 0.9], [40, 45, 0.8], [60, 45, 0.7]],
    }

    resolved = resolve_dinov3_prediction_input(record)

    assert resolved is not None
    assert resolved.image_source == str(cropped)
    assert resolved.image_source_field == "dinov3_source"
    assert resolved.image.size == (50, 40)
    assert resolved.keypoints[0] == [5.0, 10.0, 0.9]


def test_resolve_dinov3_input_can_remove_padding_from_source(tmp_path) -> None:
    source = tmp_path / "source_blackpad.png"
    image = Image.new("RGB", (80, 60), color=(0, 0, 0))
    for y in range(10, 50):
        for x in range(10, 70):
            image.putpixel((x, y), (180, 120, 90))
    image.save(source)
    record = {
        "source": str(source),
        "preprocess": {
            "type": "crop_black_border_then_blackpad",
            "padding_px": 10,
            "cropped_width": 60,
            "cropped_height": 40,
            "no_black_width": 60,
            "no_black_height": 40,
        },
        "keypoints": [[15, 20, 0.9], [30, 35, 0.8], [50, 35, 0.7]],
    }

    resolved = resolve_dinov3_prediction_input(record)

    assert resolved is not None
    assert resolved.image_source_field == "source_minus_padding"
    assert resolved.image.size == (60, 40)
    assert resolved.keypoints[0] == [5.0, 10.0, 0.9]


def test_resolve_dinov3_input_can_rebuild_crop_from_original_source(tmp_path) -> None:
    original = tmp_path / "original.png"
    image = Image.new("RGB", (80, 60), color=(0, 0, 0))
    for y in range(8, 48):
        for x in range(12, 62):
            image.putpixel((x, y), (180, 120, 90))
    image.save(original)
    record = {
        "source": "memory://model_input/0/original.png",
        "original_source": str(original),
        "preprocess": {
            "type": "crop_black_border_then_blackpad",
            "padding_px": 10,
            "crop_bbox_xyxy": [12, 8, 62, 48],
            "cropped_width": 50,
            "cropped_height": 40,
            "no_black_width": 50,
            "no_black_height": 40,
        },
        "keypoints": [[15, 18, 0.9], [30, 35, 0.8], [50, 35, 0.7]],
    }

    resolved = resolve_dinov3_prediction_input(record)

    assert resolved is not None
    assert resolved.image_source_field == "original_source_crop_bbox"
    assert resolved.image.size == (50, 40)
    assert resolved.keypoints[0] == [5.0, 8.0, 0.9]


def test_resolve_dinov3_input_replays_screen_precrop_before_crop_bbox(tmp_path) -> None:
    original = tmp_path / "screen_photo.png"
    image = Image.new("RGB", (100, 80), color=(20, 20, 20))
    for y in range(10, 70):
        for x in range(20, 80):
            image.putpixel((x, y), (160, 100, 70))
    for y in range(18, 48):
        for x in range(25, 65):
            image.putpixel((x, y), (210, 130, 90))
    image.save(original)
    record = {
        "source": "memory://model_input/0/screen_photo.png",
        "original_source": str(original),
        "pre_crop": {
            "triggered": True,
            "mode": "screen_photo_precrop",
            "box_xyxy": [20, 10, 80, 70],
        },
        "preprocess": {
            "type": "crop_black_border_then_blackpad",
            "padding_px": 10,
            "crop_bbox_xyxy": [5, 8, 45, 38],
            "cropped_width": 40,
            "cropped_height": 30,
            "no_black_width": 40,
            "no_black_height": 30,
        },
        "keypoints": [[15, 18, 0.9], [25, 28, 0.8], [35, 28, 0.7]],
    }

    resolved = resolve_dinov3_prediction_input(record)

    assert resolved is not None
    assert resolved.image_source_field == "original_source_pre_crop_crop_bbox"
    assert resolved.image.size == (40, 30)
    assert resolved.image.getpixel((0, 0)) == (210, 130, 90)
    assert resolved.keypoints[0] == [5.0, 8.0, 0.9]


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


def test_reward_only_gate_hard_rejects_below_point_one() -> None:
    scores = torch.tensor([0.00, 0.09, 0.10, 0.29, 0.30, 0.60])
    factors, direct_accept, hard_reject = dinov3_confidence_gate(
        scores,
        gate_mode="reward_only",
        reject_threshold=0.10,
        reward_threshold=0.30,
        direct_accept_threshold=0.60,
        reward_multiplier=1.50,
    )
    assert torch.allclose(factors, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.5]))
    assert direct_accept.tolist() == [False, False, False, False, False, True]
    assert hard_reject.tolist() == [True, True, False, False, False, False]


def test_checkpoint_loader_upgrades_legacy_no_reject_reward_gate() -> None:
    checkpoint = {
        "config": {
            "dinov3": {
                "backend": "timm",
                "confidence_gate_mode": "reward_only",
                "confidence_reject_threshold": 0.0,
            }
        }
    }

    cfg = load_aux_config_from_checkpoint(checkpoint)

    assert cfg.confidence_reject_threshold == 0.10


def test_gate_rejects_keypoints_outside_cropped_image_warning() -> None:
    record = {
        "action": "auto_accept",
        "final_confidence": 0.82,
        "final_bbox": [1, 2, 3, 4],
        "final_box_polygon": [[1, 1], [3, 1], [3, 4], [1, 4]],
        "dinov3_aux": {
            "warnings": ["dinov3_keypoints_outside_cropped_image"],
            "confidence_factor": 1.5,
            "direct_accept": True,
            "hard_reject": False,
        },
    }

    gated = maybe_apply_gate(
        record,
        postprocess_cfg=SimpleNamespace(auto_accept_threshold=0.43),
        min_point_prob=0.30,
        min_triplet_prob=0.30,
        max_image_reject_prob=0.70,
    )

    assert gated["action"] == "reject_or_relabel"
    assert gated["final_confidence"] == 0.0
    assert gated["usable_bbox"] is None
    assert gated["usable_box_polygon"] is None
    assert gated["dinov3_aux_gate_action"] == "reject_keypoints_outside_cropped_image"
