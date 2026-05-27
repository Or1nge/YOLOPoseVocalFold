from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KeypointContrastConfig:
    """Configuration for local keypoint feature consistency."""

    feature_index: int = 0
    projection_dim: int = 64
    projection_hidden_dim: int = 128
    temperature: float = 0.10
    patch_radius: int = 1
    negative_count: int = 12
    negative_min_distance: float = 0.12
    augment_degrees: float = 5.0
    augment_scale: float = 0.08
    augment_translate: float = 0.03
    augment_brightness: float = 0.08
    augment_contrast: float = 0.12


def first_keypoints_per_image(keypoints, batch_idx, batch_size: int, num_keypoints: int):
    """Return one normalized keypoint set per image from an Ultralytics pose batch."""

    import torch

    device = keypoints.device
    dtype = keypoints.dtype
    points = torch.zeros((batch_size, num_keypoints, 2), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, num_keypoints), device=device, dtype=torch.bool)
    filled = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    if keypoints.numel() == 0:
        return points, mask

    flat_batch = batch_idx.view(-1).long().to(device)
    for row, image_index in enumerate(flat_batch.tolist()):
        if image_index < 0 or image_index >= batch_size or filled[image_index]:
            continue
        image_keypoints = keypoints[row, :num_keypoints]
        points[image_index] = image_keypoints[..., :2].clamp(0.0, 1.0)
        if image_keypoints.shape[-1] >= 3:
            visible = image_keypoints[..., 2] > 0
        else:
            visible = torch.ones((num_keypoints,), device=device, dtype=torch.bool)
        mask[image_index] = visible
        filled[image_index] = True
    return points, mask


class KeypointProjectionHead(nn.Module):
    """Small MLP projection head for sampled local feature vectors."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return F.normalize(self.net(x), dim=-1)


class OrientedPatchProjectionHead(nn.Module):
    """Projection head for canonicalized local keypoint patches."""

    def __init__(self, in_channels: int, patch_size: int = 5, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        in_dim = int(in_channels) * self.patch_size * self.patch_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, patches):  # type: ignore[no-untyped-def]
        return F.normalize(self.net(patches.flatten(1)), dim=-1)


class GlobalProjectionHead(nn.Module):
    """Small projection head for image-level mixed/non-mixed contrast."""

    def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return F.normalize(self.net(x), dim=-1)


class RejectClassificationHead(nn.Module):
    """Binary head: 0 = usable vocal-fold image, 1 = mixed/reject image."""

    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)


def make_light_augmented_view(
    images,
    keypoints01,
    keypoint_mask,
    *,
    degrees: float = 5.0,
    scale: float = 0.08,
    translate: float = 0.03,
    brightness: float = 0.08,
    contrast: float = 0.12,
):
    """Apply mild tensor-space affine/photometric augmentation and transform keypoints."""

    import math

    import torch
    batch_size = images.shape[0]
    device = images.device
    dtype = images.dtype
    angles = (
        (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0)
        * math.radians(float(degrees))
    )
    scales = 1.0 + (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * float(scale)
    tx = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * float(translate) * 2.0
    ty = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * float(translate) * 2.0

    cos = torch.cos(angles) * scales
    sin = torch.sin(angles) * scales
    forward = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
    forward[:, 0, 0] = cos
    forward[:, 0, 1] = -sin
    forward[:, 1, 0] = sin
    forward[:, 1, 1] = cos
    forward[:, 0, 2] = tx
    forward[:, 1, 2] = ty

    matrix = forward[:, :, :2]
    inv_matrix = torch.linalg.inv(matrix)
    inv_offset = -torch.bmm(inv_matrix, forward[:, :, 2:3]).squeeze(-1)
    inverse = torch.cat([inv_matrix, inv_offset[:, :, None]], dim=2)
    grid = F.affine_grid(inverse, size=images.shape, align_corners=False)
    augmented = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    contrast_factor = 1.0 + (
        torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0
    ) * float(contrast)
    brightness_offset = (
        torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0
    ) * float(brightness)
    augmented = ((augmented - 0.5) * contrast_factor + 0.5 + brightness_offset).clamp(0.0, 1.0)

    points = keypoints01 * 2.0 - 1.0
    transformed = torch.matmul(points, matrix.transpose(1, 2)) + forward[:, None, :, 2]
    transformed01 = (transformed + 1.0) * 0.5
    inside = (
        (transformed01[..., 0] >= 0.0)
        & (transformed01[..., 0] <= 1.0)
        & (transformed01[..., 1] >= 0.0)
        & (transformed01[..., 1] <= 1.0)
    )
    return augmented, transformed01.clamp(0.0, 1.0), keypoint_mask & inside


def keypoint_anatomy_directions(keypoints01, keypoint_mask):
    """Return unit anterior-to-posterior bisector directions for each image."""

    import torch

    anterior = keypoints01[:, 0]
    posterior_mid = 0.5 * (keypoints01[:, 1] + keypoints01[:, 2])
    direction = posterior_mid - anterior
    valid = keypoint_mask[:, 0] & keypoint_mask[:, 1] & keypoint_mask[:, 2]
    norm = direction.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    direction = direction / norm
    fallback = torch.zeros_like(direction)
    fallback[:, 1] = 1.0
    direction = torch.where(valid[:, None], direction, fallback)
    return direction, valid


def sample_oriented_local_patches(
    feature_map,
    keypoints01,
    keypoint_mask,
    directions01,
    *,
    patch_size_input: float | list | tuple = 48.0,
    input_size: int = 960,
    output_size: int = 5,
):
    """Sample anatomy-oriented local feature patches around normalized keypoints.

    The canonical patch y-axis follows the anterior-to-posterior angle-bisector
    direction, and the x-axis is perpendicular to it. This preserves local
    anatomy such as which side of the anterior commissure patch faces the vocal
    folds.
    """

    import torch

    batch_size, channels, _, _ = feature_map.shape
    _, num_keypoints, _ = keypoints01.shape
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    base_values = torch.linspace(
        -0.5,
        0.5,
        output_size,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    patch_xy = _patch_size_xy_tensor(
        patch_size_input,
        num_keypoints=num_keypoints,
        device=feature_map.device,
        dtype=feature_map.dtype,
    ) / max(float(input_size), 1.0)
    x_values = base_values.view(1, output_size) * patch_xy[:, 0:1]
    y_values = base_values.view(1, output_size) * patch_xy[:, 1:2]
    yy = y_values[:, :, None].expand(num_keypoints, output_size, output_size)
    xx = x_values[:, None, :].expand(num_keypoints, output_size, output_size)
    unit_y = directions01.to(device=feature_map.device, dtype=feature_map.dtype)
    unit_x = torch.stack((-unit_y[:, 1], unit_y[:, 0]), dim=-1)
    offsets = (
        xx.view(1, num_keypoints, output_size, output_size, 1) * unit_x[:, None, None, None, :]
        + yy.view(1, num_keypoints, output_size, output_size, 1) * unit_y[:, None, None, None, :]
    )
    centers = keypoints01[:, :, None, None, :].to(device=feature_map.device, dtype=feature_map.dtype)
    grid = (centers + offsets).clamp(0.0, 1.0) * 2.0 - 1.0
    flat_grid = grid.view(batch_size, num_keypoints * output_size, output_size, 2)
    sampled = F.grid_sample(
        feature_map,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    patches = sampled.view(batch_size, channels, num_keypoints, output_size, output_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    return patches, keypoint_mask


def _patch_size_xy_tensor(patch_size_input, *, num_keypoints: int, device, dtype):  # type: ignore[no-untyped-def]
    """Return per-keypoint canonical x/y input-pixel patch footprints."""

    import torch

    if isinstance(patch_size_input, dict):
        values = [
            patch_size_input.get("anterior", patch_size_input.get(0, 48.0)),
            patch_size_input.get("left_posterior", patch_size_input.get(1, 48.0)),
            patch_size_input.get("right_posterior", patch_size_input.get(2, 48.0)),
        ][:num_keypoints]
    else:
        values = patch_size_input
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_keypoints, 2)
    elif tensor.ndim == 1:
        if tensor.numel() == 1:
            tensor = tensor.repeat(num_keypoints * 2).view(num_keypoints, 2)
        elif tensor.numel() == 2:
            tensor = tensor.view(1, 2).repeat(num_keypoints, 1)
        elif tensor.numel() == num_keypoints:
            tensor = tensor[:, None].repeat(1, 2)
        else:
            raise ValueError(
                "patch_size_input must be a scalar, [x, y], one size per keypoint, "
                "or [[x, y], ...] per keypoint."
            )
    elif tensor.ndim == 2:
        if tensor.shape != (num_keypoints, 2):
            raise ValueError(f"patch_size_input with 2 dims must have shape ({num_keypoints}, 2).")
    else:
        raise ValueError("patch_size_input has too many dimensions.")
    return tensor


def sample_local_features(feature_map, keypoints01, keypoint_mask, *, patch_radius: int = 1):
    """Bilinearly sample local feature vectors around normalized keypoints."""

    import torch
    batch_size, channels, height, width = feature_map.shape
    _, num_keypoints, _ = keypoints01.shape
    offsets = _patch_offsets(
        int(patch_radius),
        width,
        height,
        feature_map.device,
        feature_map.dtype,
    )
    grid = keypoints01[:, :, None, :] * 2.0 - 1.0
    grid = grid + offsets.view(1, 1, -1, 2)
    grid = grid.view(batch_size, num_keypoints * offsets.shape[0], 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.view(batch_size, channels, num_keypoints, offsets.shape[0])
    sampled = sampled.mean(dim=-1).permute(0, 2, 1).contiguous()
    return sampled, keypoint_mask


def sample_background_features(
    feature_map,
    keypoints01,
    keypoint_mask,
    *,
    negative_count: int = 12,
    negative_min_distance: float = 0.12,
    patch_radius: int = 1,
):
    """Sample background/hard-negative feature vectors away from visible keypoints."""

    import torch

    if negative_count <= 0:
        return feature_map.new_zeros((0, feature_map.shape[1]))

    batch_size = feature_map.shape[0]
    negative_points = torch.zeros(
        (batch_size, negative_count, 2),
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    negative_mask = torch.ones(
        (batch_size, negative_count),
        device=feature_map.device,
        dtype=torch.bool,
    )
    oversample = max(negative_count * 8, negative_count + 16)
    min_distance_sq = float(negative_min_distance) ** 2

    for batch_index in range(batch_size):
        candidates = torch.rand((oversample, 2), device=feature_map.device, dtype=feature_map.dtype)
        visible_points = keypoints01[batch_index][keypoint_mask[batch_index]]
        if visible_points.numel():
            distances = (candidates[:, None, :] - visible_points[None, :, :]).square().sum(dim=-1)
            keep = distances.min(dim=1).values >= min_distance_sq
            candidates = candidates[keep]
        if candidates.shape[0] < negative_count:
            extra = torch.rand(
                (negative_count - candidates.shape[0], 2),
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            candidates = torch.cat([candidates, extra], dim=0)
        negative_points[batch_index] = candidates[:negative_count]

    negatives, mask = sample_local_features(
        feature_map,
        negative_points,
        negative_mask,
        patch_radius=patch_radius,
    )
    return negatives[mask]


def keypoint_local_contrast_loss(
    feature_view1,
    feature_view2,
    keypoints_view1,
    keypoints_view2,
    keypoint_mask,
    projection_head,
    *,
    temperature: float = 0.10,
    patch_radius: int = 1,
    negative_count: int = 12,
    negative_min_distance: float = 0.12,
):
    """Contrast same-image same-keypoint embeddings across two augmented views."""

    import torch
    mask = keypoint_mask.bool()
    if not mask.any():
        return feature_view1.new_zeros(())

    local1, _ = sample_local_features(
        feature_view1,
        keypoints_view1,
        mask,
        patch_radius=patch_radius,
    )
    local2, _ = sample_local_features(
        feature_view2,
        keypoints_view2,
        mask,
        patch_radius=patch_radius,
    )
    z1 = projection_head(local1[mask])
    z2 = projection_head(local2[mask])
    positive_loss = 1.0 - (z1 * z2).sum(dim=-1)

    negatives1 = sample_background_features(
        feature_view1,
        keypoints_view1,
        mask,
        negative_count=negative_count,
        negative_min_distance=negative_min_distance,
        patch_radius=patch_radius,
    )
    negatives2 = sample_background_features(
        feature_view2,
        keypoints_view2,
        mask,
        negative_count=negative_count,
        negative_min_distance=negative_min_distance,
        patch_radius=patch_radius,
    )
    negatives = torch.cat([negatives1, negatives2], dim=0)
    if negatives.numel() == 0:
        return positive_loss.mean()

    z_neg = projection_head(negatives)
    loss12 = _one_way_info_nce(z1, z2, z_neg, temperature=temperature)
    loss21 = _one_way_info_nce(z2, z1, z_neg, temperature=temperature)
    return 0.5 * (loss12 + loss21)


def oriented_keypoint_patch_contrast_loss(
    feature_view1,
    feature_view2,
    keypoints_view1,
    keypoints_view2,
    keypoint_mask,
    projection_head,
    *,
    temperature: float = 0.10,
    patch_size_input: float | list | tuple = 48.0,
    input_size: int = 960,
    output_size: int = 5,
    negative_count: int = 12,
    negative_min_distance: float = 0.12,
):
    """Contrast same keypoint after sampling anatomy-oriented local patches."""

    import torch

    direction1, valid1 = keypoint_anatomy_directions(keypoints_view1, keypoint_mask)
    direction2, valid2 = keypoint_anatomy_directions(keypoints_view2, keypoint_mask)
    mask = keypoint_mask.bool() & valid1[:, None] & valid2[:, None]
    if not mask.any():
        return feature_view1.new_zeros(())

    patches1, _ = sample_oriented_local_patches(
        feature_view1,
        keypoints_view1,
        mask,
        direction1,
        patch_size_input=patch_size_input,
        input_size=input_size,
        output_size=output_size,
    )
    patches2, _ = sample_oriented_local_patches(
        feature_view2,
        keypoints_view2,
        mask,
        direction2,
        patch_size_input=patch_size_input,
        input_size=input_size,
        output_size=output_size,
    )
    z1 = projection_head(patches1[mask])
    z2 = projection_head(patches2[mask])

    negatives1 = sample_background_features(
        feature_view1,
        keypoints_view1,
        mask,
        negative_count=negative_count,
        negative_min_distance=negative_min_distance,
        patch_radius=0,
    )
    negatives2 = sample_background_features(
        feature_view2,
        keypoints_view2,
        mask,
        negative_count=negative_count,
        negative_min_distance=negative_min_distance,
        patch_radius=0,
    )
    negatives = torch.cat([negatives1, negatives2], dim=0)
    if negatives.numel() == 0:
        return (1.0 - (z1 * z2).sum(dim=-1)).mean()

    # Give background vectors the same spatial shape expected by the patch head.
    size = int(output_size)
    negatives = negatives[:, :, None, None].expand(-1, -1, size, size).contiguous()
    z_neg = projection_head(negatives)
    loss12 = _one_way_info_nce(z1, z2, z_neg, temperature=temperature)
    loss21 = _one_way_info_nce(z2, z1, z_neg, temperature=temperature)
    return 0.5 * (loss12 + loss21)


def supervised_image_contrast_loss(z1, z2, labels, *, temperature: float = 0.10):
    """Supervised contrast for mixed/reject versus non-mixed images."""

    import torch

    if labels.numel() <= 1:
        return z1.new_zeros(())
    embeddings = torch.cat([z1, z2], dim=0)
    labels = labels.view(-1).long()
    labels = torch.cat([labels, labels], dim=0)
    logits = embeddings @ embeddings.transpose(0, 1) / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    if not same.any():
        return z1.new_zeros(())
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denom = same.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * same).sum(dim=1) / denom
    valid = same.any(dim=1)
    return per_anchor[valid].mean()


def _one_way_info_nce(anchor, positive, negatives, *, temperature: float):
    import torch
    temp = max(float(temperature), 1e-6)
    pos_logits = (anchor * positive).sum(dim=-1, keepdim=True) / temp
    neg_logits = anchor @ negatives.transpose(0, 1) / temp
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros((anchor.shape[0],), device=anchor.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def _patch_offsets(radius: int, width: int, height: int, device, dtype):
    import torch

    if radius <= 0:
        return torch.zeros((1, 2), device=device, dtype=dtype)
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    offsets = torch.stack(
        [
            xx.reshape(-1) * (2.0 / max(width, 1)),
            yy.reshape(-1) * (2.0 / max(height, 1)),
        ],
        dim=1,
    )
    return offsets
