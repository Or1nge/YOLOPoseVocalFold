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
    """Configuration for the frozen-DINOv3 keypoint plausibility head."""

    backend: str = "timm"
    model_name: str = "dinov3_vits16"
    timm_model_name: str = "vit_small_patch16_dinov3.lvd1689m"
    repo_dir: str | None = None
    weights: str | None = None
    transformers_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    point_hidden_dim: int = 256
    triplet_hidden_dim: int = 512
    image_hidden_dim: int = 256
    background_points: int = 12
    corrupted_triplets: int = 3
    point_loss_weight: float = 1.0
    triplet_loss_weight: float = 1.0
    image_loss_weight: float = 0.20
    corrupted_jitter: float = 0.10
    min_background_distance: float = 0.08
    score_mode: str = "geometric_mean"


class DinoV3KeypointAuxHead(nn.Module):
    """Small trainable head on top of frozen DINOv3 dense features.

    The point branch predicts 4 classes: background, anterior, left posterior,
    right posterior. The triplet branch predicts whether an ordered A/L/R triplet
    is anatomically plausible. L/R are never treated as positive pairs.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        point_hidden_dim: int = 256,
        triplet_hidden_dim: int = 512,
        image_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + 2, int(point_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(point_hidden_dim), 4),
        )
        triplet_in = self.feature_dim * 3 + 6 + 6
        self.triplet_mlp = nn.Sequential(
            nn.Linear(triplet_in, int(triplet_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(triplet_hidden_dim), 2),
        )
        self.image_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, int(image_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(image_hidden_dim), 2),
        )

    def point_logits(self, point_features: torch.Tensor, points01: torch.Tensor) -> torch.Tensor:
        return self.point_mlp(torch.cat([point_features, points01], dim=-1))

    def triplet_logits(self, triplet_features: torch.Tensor, triplets01: torch.Tensor) -> torch.Tensor:
        flat_features = triplet_features.flatten(1)
        flat_points = triplets01.flatten(1)
        anterior = triplets01[:, 0]
        left = triplets01[:, 1]
        right = triplets01[:, 2]
        rel = torch.cat(
            [
                left - anterior,
                right - anterior,
                right - left,
            ],
            dim=-1,
        )
        return self.triplet_mlp(torch.cat([flat_features, flat_points, rel], dim=-1))

    def image_logits(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.image_mlp(global_features)


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


def build_point_targets(
    keypoints01: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    background_points: int = 12,
    min_background_distance: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_keypoints, _ = keypoints01.shape
    device = keypoints01.device
    dtype = keypoints01.dtype
    total = num_keypoints + int(background_points)
    points = torch.zeros((batch_size, total, 2), device=device, dtype=dtype)
    labels = torch.zeros((batch_size, total), device=device, dtype=torch.long)
    valid = torch.zeros((batch_size, total), device=device, dtype=torch.bool)
    points[:, :num_keypoints] = keypoints01.clamp(0.0, 1.0)
    for kp_index in range(num_keypoints):
        labels[:, kp_index] = kp_index + 1
    valid[:, :num_keypoints] = keypoint_mask.bool()
    if background_points > 0:
        backgrounds = sample_background_points(
            keypoints01,
            keypoint_mask,
            count=int(background_points),
            min_distance=float(min_background_distance),
        )
        points[:, num_keypoints:] = backgrounds
        labels[:, num_keypoints:] = 0
        valid[:, num_keypoints:] = True
    return points, labels, valid


def build_triplet_targets(
    keypoints01: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    corrupted_triplets: int = 3,
    corrupted_jitter: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_images = keypoint_mask.bool().all(dim=1)
    image_indices = valid_images.nonzero(as_tuple=False).flatten()
    if image_indices.numel() == 0:
        empty_triplets = keypoints01.new_zeros((0, 3, 2))
        empty_labels = torch.zeros((0,), device=keypoints01.device, dtype=torch.long)
        return empty_triplets, empty_labels, empty_labels

    triplets: list[torch.Tensor] = []
    labels: list[int] = []
    owners: list[int] = []
    for image_index in image_indices.tolist():
        true_triplet = keypoints01[image_index].clamp(0.0, 1.0)
        triplets.append(true_triplet)
        labels.append(1)
        owners.append(image_index)
        for corrupt_index in range(int(corrupted_triplets)):
            triplets.append(corrupt_triplet(true_triplet, corrupt_index, jitter=float(corrupted_jitter)))
            labels.append(0)
            owners.append(image_index)
    return (
        torch.stack(triplets, dim=0),
        torch.tensor(labels, device=keypoints01.device, dtype=torch.long),
        torch.tensor(owners, device=keypoints01.device, dtype=torch.long),
    )


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


def corrupt_triplet(triplet: torch.Tensor, corrupt_index: int, *, jitter: float) -> torch.Tensor:
    if corrupt_index % 3 == 0:
        order = torch.tensor([1, 0, 2], device=triplet.device)
        return triplet[order]
    if corrupt_index % 3 == 1:
        noisy = triplet + torch.randn_like(triplet) * float(jitter)
        return noisy.clamp(0.0, 1.0)
    replaced = triplet.clone()
    replaced[corrupt_index % 3] = torch.rand((2,), device=triplet.device, dtype=triplet.dtype)
    return replaced.clamp(0.0, 1.0)


def score_aux_triplet(
    head: DinoV3KeypointAuxHead,
    feature_map: torch.Tensor,
    global_feature: torch.Tensor,
    triplets01: torch.Tensor,
    *,
    score_mode: str = "geometric_mean",
) -> dict[str, torch.Tensor]:
    point_features = sample_point_features(feature_map, triplets01)
    point_logits = head.point_logits(point_features, triplets01)
    expected = torch.arange(1, 4, device=triplets01.device).view(1, 3)
    point_probs = F.softmax(point_logits, dim=-1).gather(-1, expected[:, :, None].expand(triplets01.shape[0], -1, 1)).squeeze(-1)
    triplet_logits = head.triplet_logits(point_features, triplets01)
    triplet_valid_prob = F.softmax(triplet_logits, dim=-1)[:, 1]
    image_reject_prob = F.softmax(head.image_logits(global_feature), dim=-1)[:, 1]
    if score_mode == "min":
        confidence_factor = torch.minimum(point_probs.min(dim=1).values, triplet_valid_prob)
        confidence_factor = torch.minimum(confidence_factor, 1.0 - image_reject_prob)
    else:
        confidence_factor = (
            point_probs.clamp_min(1e-6).prod(dim=1)
            * triplet_valid_prob.clamp_min(1e-6)
            * (1.0 - image_reject_prob).clamp_min(1e-6)
        ).pow(1.0 / 5.0)
    return {
        "point_expected_probs": point_probs,
        "triplet_valid_prob": triplet_valid_prob,
        "image_reject_prob": image_reject_prob,
        "confidence_factor": confidence_factor,
        "point_logits": point_logits,
        "triplet_logits": triplet_logits,
    }
