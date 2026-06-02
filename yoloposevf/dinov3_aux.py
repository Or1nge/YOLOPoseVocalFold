from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DinoV3AuxConfig:
    """Configuration for the frozen-DINOv3 point-region classifier."""

    backend: str = "timm"
    model_name: str = "dinov3_vits16"
    timm_model_name: str = "vit_small_patch16_dinov3.lvd1689m"
    repo_dir: str | None = None
    weights: str | None = None
    transformers_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    point_hidden_dim: int = 256
    background_points: int = 3
    near_background_points: int = 0
    hard_negative_points: int = 3
    point_loss_weight: float = 1.0
    min_background_distance: float = 0.08
    near_background_min_distance: float = 0.06
    near_background_max_distance: float = 0.18
    hard_negative_min_confidence: float = 0.30
    hard_negative_min_distance: float = 0.08
    oriented_patch_size_input: float | tuple[float, float] = 48.0
    oriented_patch_output_size: int = 5
    include_point_coordinates: bool = False
    include_valid_mask: bool = True
    valid_mask_luma_floor: float = 8.0
    crop_black_border_luma_floor: float = 8.0
    confidence_gate_mode: str = "reward_only"
    confidence_reject_threshold: float = 0.10
    confidence_penalty_threshold: float = 0.0
    confidence_reward_threshold: float = 0.30
    confidence_direct_accept_threshold: float = 0.60
    confidence_reward_multiplier: float = 1.50
    score_mode: str = "geometric_mean"


class DinoV3KeypointAuxHead(nn.Module):
    """Small trainable point head on top of frozen DINOv3 dense features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        patch_output_size: int = 5,
        point_hidden_dim: int = 256,
        include_coordinates: bool = False,
        include_valid_mask: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_output_size = int(patch_output_size)
        self.include_coordinates = bool(include_coordinates)
        self.include_valid_mask = bool(include_valid_mask)
        point_input_dim = self.feature_dim * self.patch_output_size * self.patch_output_size
        if self.include_valid_mask:
            point_input_dim += self.patch_output_size * self.patch_output_size
        if self.include_coordinates:
            point_input_dim += 2
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, int(point_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(point_hidden_dim), 4),
        )

    def point_logits(
        self,
        point_features: torch.Tensor,
        points01: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.include_valid_mask:
            if valid_mask is None:
                raise ValueError("valid_mask is required when include_valid_mask=True")
            point_features = point_features * valid_mask.to(device=point_features.device, dtype=point_features.dtype)
        flat_features = point_features.flatten(2)
        if self.include_valid_mask:
            mask_features = valid_mask.to(device=point_features.device, dtype=point_features.dtype).flatten(2)
            flat_features = torch.cat([flat_features, mask_features], dim=-1)
        if self.include_coordinates:
            return self.point_mlp(torch.cat([flat_features, points01], dim=-1))
        return self.point_mlp(flat_features)


class DinoV3DenseExtractor(nn.Module):
    """Thin adapter for DINOv3 backbones loaded from torch.hub or Transformers."""

    def __init__(self, model: nn.Module, *, backend: str) -> None:
        super().__init__()
        self.model = model
        self.backend = backend

    def forward_dense(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "transformers":
            outputs = self.model(pixel_values=images)
            tokens = outputs.last_hidden_state
            feature_map = tokens_to_feature_map(tokens)
            global_feature = getattr(outputs, "pooler_output", None)
            if global_feature is None:
                global_feature = feature_map.mean(dim=(2, 3))
            return feature_map, global_feature

        if hasattr(self.model, "forward_features"):
            outputs = self.model.forward_features(images)
        else:
            outputs = self.model(images)
        if isinstance(outputs, torch.Tensor):
            feature_map = tokens_to_feature_map(outputs)
            global_feature = outputs[:, 0] if outputs.ndim == 3 else outputs
            return feature_map, global_feature
        if not isinstance(outputs, dict):
            raise RuntimeError("DINOv3 torch_hub model did not return dense feature tokens.")
        if "x_norm_patchtokens" in outputs:
            patch_tokens = outputs["x_norm_patchtokens"]
        elif "x_prenorm" in outputs:
            patch_tokens = outputs["x_prenorm"]
        else:
            raise RuntimeError(f"Cannot find patch tokens in DINOv3 outputs: {sorted(outputs)}")
        feature_map = tokens_to_feature_map(patch_tokens, has_cls_token=False)
        global_feature = outputs.get("x_norm_clstoken")
        if global_feature is None:
            global_feature = feature_map.mean(dim=(2, 3))
        return feature_map, global_feature


def load_dinov3_extractor(cfg: DinoV3AuxConfig, device: torch.device) -> DinoV3DenseExtractor:
    backend = cfg.backend.lower()
    if backend == "transformers":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for backend='transformers'.") from exc
        model = AutoModel.from_pretrained(cfg.transformers_model)
        extractor = DinoV3DenseExtractor(model, backend="transformers")
    elif backend == "timm":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for backend='timm'.") from exc
        model = timm.create_model(cfg.timm_model_name, pretrained=True)
        extractor = DinoV3DenseExtractor(model, backend="timm")
    elif backend == "torch_hub":
        if not cfg.repo_dir:
            raise ValueError("dinov3.repo_dir is required for backend='torch_hub'.")
        kwargs: dict[str, Any] = {"source": "local"}
        if cfg.weights:
            kwargs["weights"] = cfg.weights
        model = torch.hub.load(str(Path(cfg.repo_dir).expanduser()), cfg.model_name, **kwargs)
        extractor = DinoV3DenseExtractor(model, backend="torch_hub")
    else:
        raise ValueError("dinov3.backend must be 'torch_hub' or 'transformers'.")
    extractor.to(device)
    extractor.eval()
    return extractor


def normalize_for_dinov3(images: torch.Tensor, cfg: DinoV3AuxConfig) -> torch.Tensor:
    mean = torch.tensor(cfg.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(cfg.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std.clamp_min(1e-6)


def tokens_to_feature_map(
    tokens: torch.Tensor,
    *,
    has_cls_token: bool | None = None,
    prefix_tokens: int | None = None,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens with shape [B, N, C], got {tuple(tokens.shape)}")
    token_count = tokens.shape[1]
    if prefix_tokens is None:
        if has_cls_token is not None:
            prefix_tokens = 1 if has_cls_token else 0
        else:
            # DINOv3 ViT outputs usually contain 1 class token + 4 register
            # tokens before patch tokens; older ViTs may contain only class.
            prefix_tokens = None
            for candidate in (0, 1, 5):
                remaining = token_count - candidate
                root_candidate = int(math.sqrt(max(remaining, 0)))
                if remaining > 0 and root_candidate * root_candidate == remaining:
                    prefix_tokens = candidate
                    break
            if prefix_tokens is None:
                prefix_tokens = 0
    if prefix_tokens:
        tokens = tokens[:, int(prefix_tokens):]
        token_count = tokens.shape[1]
    root = int(math.sqrt(token_count))
    if root * root != token_count:
        raise ValueError(f"cannot reshape {token_count} DINO patch tokens to a square feature map")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], root, root).contiguous()


def sample_point_features(feature_map: torch.Tensor, points01: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _, _ = feature_map.shape
    if points01.ndim != 3 or points01.shape[0] != batch_size or points01.shape[-1] != 2:
        raise ValueError("points01 must have shape [B, N, 2]")
    grid = points01.clamp(0.0, 1.0) * 2.0 - 1.0
    grid = grid.view(batch_size, points01.shape[1], 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


def anatomy_directions(keypoints01: torch.Tensor, keypoint_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    anterior = keypoints01[:, 0]
    posterior_mid = 0.5 * (keypoints01[:, 1] + keypoints01[:, 2])
    direction = posterior_mid - anterior
    valid = keypoint_mask.bool().all(dim=1)
    norm = direction.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    direction = direction / norm
    fallback = torch.zeros_like(direction)
    fallback[:, 1] = 1.0
    return torch.where(valid[:, None], direction, fallback), valid


def sample_oriented_point_regions(
    feature_map: torch.Tensor,
    points01: torch.Tensor,
    directions01: torch.Tensor,
    *,
    patch_size_input: float | tuple[float, float] = 48.0,
    input_size: int = 448,
    output_size: int = 5,
) -> torch.Tensor:
    """Sample canonicalized DINO feature patches around arbitrary points."""

    batch_size, channels, _, _ = feature_map.shape
    if points01.ndim != 3 or points01.shape[0] != batch_size or points01.shape[-1] != 2:
        raise ValueError("points01 must have shape [B, N, 2]")
    if directions01.shape != points01.shape:
        raise ValueError("directions01 must have the same shape as points01")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    _, point_count, _ = points01.shape
    patch_xy = _patch_size_xy(
        patch_size_input,
        device=feature_map.device,
        dtype=feature_map.dtype,
    ) / max(float(input_size), 1.0)
    base = torch.linspace(-0.5, 0.5, output_size, device=feature_map.device, dtype=feature_map.dtype)
    xx = base.view(1, output_size) * patch_xy[0]
    yy = base.view(output_size, 1) * patch_xy[1]
    unit_y = directions01.to(device=feature_map.device, dtype=feature_map.dtype)
    norm = unit_y.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    unit_y = unit_y / norm
    unit_x = torch.stack((-unit_y[..., 1], unit_y[..., 0]), dim=-1)
    offsets = (
        xx.view(1, 1, 1, output_size, 1) * unit_x[:, :, None, None, :]
        + yy.view(1, 1, output_size, 1, 1) * unit_y[:, :, None, None, :]
    )
    centers = points01[:, :, None, None, :].to(device=feature_map.device, dtype=feature_map.dtype)
    grid = (centers + offsets) * 2.0 - 1.0
    flat_grid = grid.view(batch_size, point_count * output_size, output_size, 2)
    sampled = F.grid_sample(
        feature_map,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.view(batch_size, channels, point_count, output_size, output_size)
    return sampled.permute(0, 2, 1, 3, 4).contiguous()


def sample_oriented_point_region_masks(
    mask_map: torch.Tensor,
    points01: torch.Tensor,
    directions01: torch.Tensor,
    *,
    patch_size_input: float | tuple[float, float] = 48.0,
    input_size: int = 448,
    output_size: int = 5,
) -> torch.Tensor:
    """Sample foreground-valid masks using the same oriented grids as DINO features."""

    if mask_map.ndim != 4 or mask_map.shape[1] != 1:
        raise ValueError("mask_map must have shape [B, 1, H, W]")
    return sample_oriented_point_regions(
        mask_map,
        points01,
        directions01,
        patch_size_input=patch_size_input,
        input_size=input_size,
        output_size=output_size,
    )


def foreground_mask_from_images(images: torch.Tensor, *, luma_floor: float = 8.0) -> torch.Tensor:
    """Build a soft foreground mask from unnormalized RGB image tensors in 0-1 range."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    rgb = images.clamp(0.0, 1.0)
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    floor = max(float(luma_floor), 0.0) / 255.0
    return (luma > floor).to(dtype=images.dtype)


def _patch_size_xy(value: float | tuple[float, float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.repeat(2)
    if tensor.ndim == 1 and tensor.numel() == 2:
        return tensor
    raise ValueError("oriented_patch_size_input must be a scalar or [x, y]")


def build_point_targets(
    keypoints01: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    background_points: int = 12,
    near_background_points: int = 0,
    min_background_distance: float = 0.08,
    near_background_min_distance: float = 0.06,
    near_background_max_distance: float = 0.18,
    hard_negative_points01: torch.Tensor | None = None,
    hard_negative_mask: torch.Tensor | None = None,
    hard_negative_directions01: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_keypoints, _ = keypoints01.shape
    device = keypoints01.device
    dtype = keypoints01.dtype
    hard_count = 0 if hard_negative_points01 is None else int(hard_negative_points01.shape[1])
    total = num_keypoints + int(background_points) + int(near_background_points) + hard_count
    points = torch.zeros((batch_size, total, 2), device=device, dtype=dtype)
    labels = torch.zeros((batch_size, total), device=device, dtype=torch.long)
    valid = torch.zeros((batch_size, total), device=device, dtype=torch.bool)
    directions = torch.zeros((batch_size, total, 2), device=device, dtype=dtype)
    anatomy_dir, _ = anatomy_directions(keypoints01, keypoint_mask)
    points[:, :num_keypoints] = keypoints01.clamp(0.0, 1.0)
    for kp_index in range(num_keypoints):
        labels[:, kp_index] = kp_index + 1
    valid[:, :num_keypoints] = keypoint_mask.bool()
    directions[:, :num_keypoints] = anatomy_dir[:, None, :]
    cursor = num_keypoints
    if background_points > 0:
        backgrounds = sample_background_points(
            keypoints01,
            keypoint_mask,
            count=int(background_points),
            min_distance=float(min_background_distance),
        )
        points[:, cursor : cursor + int(background_points)] = backgrounds
        labels[:, cursor : cursor + int(background_points)] = 0
        valid[:, cursor : cursor + int(background_points)] = True
        directions[:, cursor : cursor + int(background_points)] = anatomy_dir[:, None, :]
        cursor += int(background_points)
    if near_background_points > 0:
        near_points, near_valid = sample_near_keypoint_background_points(
            keypoints01,
            keypoint_mask,
            count=int(near_background_points),
            min_distance=float(near_background_min_distance),
            max_distance=float(near_background_max_distance),
        )
        points[:, cursor : cursor + int(near_background_points)] = near_points
        labels[:, cursor : cursor + int(near_background_points)] = 0
        valid[:, cursor : cursor + int(near_background_points)] = near_valid
        directions[:, cursor : cursor + int(near_background_points)] = anatomy_dir[:, None, :]
        cursor += int(near_background_points)
    if hard_count:
        hard_points = hard_negative_points01.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        hard_mask = (
            torch.ones((batch_size, hard_count), device=device, dtype=torch.bool)
            if hard_negative_mask is None
            else hard_negative_mask.to(device=device).bool()
        )
        if hard_negative_directions01 is None:
            hard_dirs = anatomy_dir[:, None, :].expand(batch_size, hard_count, 2)
        else:
            hard_dirs = hard_negative_directions01.to(device=device, dtype=dtype)
        points[:, cursor : cursor + hard_count] = hard_points
        labels[:, cursor : cursor + hard_count] = 0
        valid[:, cursor : cursor + hard_count] = hard_mask
        directions[:, cursor : cursor + hard_count] = hard_dirs
    return points, labels, valid, directions


def sample_background_points(
    keypoints01: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    count: int,
    min_distance: float,
) -> torch.Tensor:
    batch_size = keypoints01.shape[0]
    device = keypoints01.device
    dtype = keypoints01.dtype
    result = torch.zeros((batch_size, count, 2), device=device, dtype=dtype)
    min_distance_sq = float(min_distance) ** 2
    oversample = max(count * 8, count + 16)
    for batch_index in range(batch_size):
        candidates = torch.rand((oversample, 2), device=device, dtype=dtype)
        visible = keypoints01[batch_index][keypoint_mask[batch_index].bool()]
        if visible.numel():
            distances = (candidates[:, None, :] - visible[None, :, :]).square().sum(dim=-1)
            candidates = candidates[distances.min(dim=1).values >= min_distance_sq]
        if candidates.shape[0] < count:
            extra = torch.rand((count - candidates.shape[0], 2), device=device, dtype=dtype)
            candidates = torch.cat([candidates, extra], dim=0)
        result[batch_index] = candidates[:count]
    return result


def sample_near_keypoint_background_points(
    keypoints01: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    count: int,
    min_distance: float,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = keypoints01.shape[0]
    device = keypoints01.device
    dtype = keypoints01.dtype
    result = torch.zeros((batch_size, count, 2), device=device, dtype=dtype)
    valid = torch.zeros((batch_size, count), device=device, dtype=torch.bool)
    min_distance = max(float(min_distance), 1e-4)
    max_distance = max(float(max_distance), min_distance)
    for batch_index in range(batch_size):
        visible_indices = keypoint_mask[batch_index].bool().nonzero(as_tuple=False).flatten()
        if visible_indices.numel() == 0:
            continue
        for item_index in range(count):
            keypoint_index = visible_indices[item_index % visible_indices.numel()]
            center = keypoints01[batch_index, keypoint_index]
            angle = torch.rand((), device=device, dtype=dtype) * (2.0 * math.pi)
            radius = min_distance + torch.rand((), device=device, dtype=dtype) * (max_distance - min_distance)
            offset = torch.stack((torch.cos(angle), torch.sin(angle))) * radius
            result[batch_index, item_index] = (center + offset).clamp(0.0, 1.0)
            valid[batch_index, item_index] = True
    return result, valid


def dinov3_confidence_gate(
    point_region_score: torch.Tensor,
    *,
    gate_mode: str = "reward_only",
    reject_threshold: float = 0.0,
    penalty_threshold: float = 0.0,
    reward_threshold: float = 0.30,
    direct_accept_threshold: float = 0.60,
    reward_multiplier: float = 1.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_mode = str(gate_mode).lower()
    reject_threshold = min(max(float(reject_threshold), 0.0), 1.0)
    reward_threshold = min(max(float(reward_threshold), 0.0), 1.0)
    direct_accept_threshold = min(max(float(direct_accept_threshold), reward_threshold), 1.0)
    reward_multiplier = max(float(reward_multiplier), 1.0)

    reward_range = max(direct_accept_threshold - reward_threshold, 1e-6)
    reward_progress = ((point_region_score - reward_threshold) / reward_range).clamp(0.0, 1.0)
    reward_factor = 1.0 + (reward_multiplier - 1.0) * reward_progress
    if gate_mode == "penalty_reward":
        penalty_threshold = max(float(penalty_threshold), 1e-6)
        penalty_factor = (point_region_score / penalty_threshold).clamp(0.0, 1.0)
        reward_factor = torch.where(point_region_score < penalty_threshold, penalty_factor, reward_factor)
    elif gate_mode != "reward_only":
        raise ValueError("DINOv3 confidence_gate_mode must be 'reward_only' or 'penalty_reward'.")

    direct_accept = point_region_score >= direct_accept_threshold
    hard_reject = point_region_score < reject_threshold if reject_threshold > 0.0 else torch.zeros_like(direct_accept)
    return reward_factor, direct_accept, hard_reject


def score_aux_triplet(
    head: DinoV3KeypointAuxHead,
    feature_map: torch.Tensor,
    global_feature: torch.Tensor,
    triplets01: torch.Tensor,
    *,
    patch_size_input: float | tuple[float, float] = 48.0,
    input_size: int = 448,
    score_mode: str = "geometric_mean",
    gate_mode: str = "reward_only",
    reject_threshold: float = 0.0,
    penalty_threshold: float = 0.0,
    reward_threshold: float = 0.30,
    direct_accept_threshold: float = 0.60,
    reward_multiplier: float = 1.50,
    valid_mask_map: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    del global_feature
    directions, _ = anatomy_directions(triplets01, torch.ones(triplets01.shape[:2], device=triplets01.device, dtype=torch.bool))
    point_features = sample_oriented_point_regions(
        feature_map,
        triplets01,
        directions[:, None, :].expand_as(triplets01),
        patch_size_input=patch_size_input,
        input_size=input_size,
        output_size=head.patch_output_size,
    )
    point_valid_mask = None
    if head.include_valid_mask:
        if valid_mask_map is None:
            raise ValueError("valid_mask_map is required when head.include_valid_mask=True")
        point_valid_mask = sample_oriented_point_region_masks(
            valid_mask_map,
            triplets01,
            directions[:, None, :].expand_as(triplets01),
            patch_size_input=patch_size_input,
            input_size=input_size,
            output_size=head.patch_output_size,
        )
    point_logits = head.point_logits(point_features, triplets01, valid_mask=point_valid_mask)
    expected = torch.arange(1, 4, device=triplets01.device).view(1, 3)
    point_probs = F.softmax(point_logits, dim=-1).gather(-1, expected[:, :, None].expand(triplets01.shape[0], -1, 1)).squeeze(-1)
    if score_mode == "min":
        point_region_score = point_probs.min(dim=1).values
    else:
        point_region_score = point_probs.clamp_min(1e-6).prod(dim=1).pow(1.0 / 3.0)
    confidence_factor, direct_accept, hard_reject = dinov3_confidence_gate(
        point_region_score,
        gate_mode=gate_mode,
        reject_threshold=reject_threshold,
        penalty_threshold=penalty_threshold,
        reward_threshold=reward_threshold,
        direct_accept_threshold=direct_accept_threshold,
        reward_multiplier=reward_multiplier,
    )
    return {
        "point_expected_probs": point_probs,
        "point_region_score": point_region_score,
        "confidence_factor": confidence_factor,
        "direct_accept": direct_accept,
        "hard_reject": hard_reject,
        "valid_fraction": (
            torch.ones_like(point_region_score)
            if point_valid_mask is None
            else point_valid_mask.mean(dim=(2, 3, 4)).mean(dim=1)
        ),
        "point_logits": point_logits,
    }
