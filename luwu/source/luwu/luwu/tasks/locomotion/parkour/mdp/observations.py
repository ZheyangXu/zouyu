# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific observation functions for the Go2 parkour task.

Provides privileged oracle observations for the parkour field policy and depth camera
processing for the distilled vision policy.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def height_measurements_oracle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.2,
) -> torch.Tensor:
    """Compute virtual terrain height measurements for oracle parkour policy.

    Uses the barrier track's virtual height sampling (not the physical height scanner).
    This provides privilege information about upcoming obstacles.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.
        offset: Vertical offset applied to height measurements.

    Returns:
        A tensor of shape (num_envs, num_points) with sampled virtual terrain heights.
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.zeros(env.num_envs, 1, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    heights = env.barrier_track.height_at_positions(
        asset.data.root_pos_w[:, :2],
        asset.data.root_quat_w,
        offset,
    )
    return heights


def engaging_block_info(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide engaging block metadata for the oracle policy.

    Returns a concatenated tensor of [block_type_onehot, distance_to_block, block_height, block_width].
    The block type is a 10-dim one-hot over parkour obstacle types.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        A tensor of shape (num_envs, 13) with engaging block information.
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.zeros(env.num_envs, 13, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    return env.barrier_track.get_engaging_block_info(asset.data.root_pos_w)


def forward_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    apply_noise: bool = False,
) -> torch.Tensor:
    """Read forward depth camera data for the distilled vision policy.

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration for the depth camera.
        apply_noise: Whether to apply realistic stereo depth noise.

    Returns:
        A tensor of shape (num_envs, H, W) with normalized depth values in [0, 1].
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[0]  # type: ignore[index]
    depth = depth.squeeze(-1)  # shape: (num_envs, H, W)

    if apply_noise and hasattr(env, "_depth_noise_cfg"):
        depth = _apply_stereo_noise(depth, env._depth_noise_cfg, env.device)

    depth = torch.clamp(depth / 3.0, 0.0, 1.0)
    return depth.unsqueeze(1)  # shape: (num_envs, 1, H, W)


def _apply_stereo_noise(
    depth: torch.Tensor,
    noise_cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    """Apply realistic stereo depth camera noise.

    Simulates artifacts common in stereo depth: near/far noise, block artifacts,
    and sky dropouts.

    Args:
        depth: Raw depth tensor of shape (num_envs, H, W).
        noise_cfg: Configuration dict with noise parameters.
        device: The compute device.

    Returns:
        Noisy depth tensor of same shape.
    """
    num_envs, h, w = depth.shape

    # Near-plane and far-plane noise
    near_noise_std = noise_cfg.get("stereo_near_noise_std", 0.02)
    far_noise_std = noise_cfg.get("stereo_far_noise_std", 0.08)
    far_distance = noise_cfg.get("stereo_far_distance", 1.2)
    min_distance = noise_cfg.get("stereo_min_distance", 0.175)

    noise = torch.randn(num_envs, h, w, device=device)
    far_mask = depth > far_distance
    noise_scale = torch.where(far_mask, far_noise_std, near_noise_std)
    depth = depth + noise * noise_scale

    # Min distance clamp
    depth = torch.clamp(depth, min=min_distance)

    # Block artifacts: occasional full-block corruption
    block_prob = noise_cfg.get("full_block_prob", 0.008)
    block_size = 8
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            mask = torch.rand(num_envs, device=device) < block_prob
            if mask.any():
                i_end = min(i + block_size, h)
                j_end = min(j + block_size, w)
                depth[mask, i:i_end, j:j_end] = min_distance

    # Spark artifacts: small random patches
    spark_prob = noise_cfg.get("half_block_spark_prob", 0.02)
    spark_size = 4
    for i in range(0, h, spark_size):
        for j in range(0, w, spark_size):
            mask = torch.rand(num_envs, device=device) < spark_prob
            if mask.any():
                i_end = min(i + spark_size, h)
                j_end = min(j + spark_size, w)
                depth[mask, i:i_end, j:j_end] = torch.rand_like(depth[mask, i:i_end, j:j_end]) * far_distance

    # Sky artifacts: random rows at top of image
    sky_prob = noise_cfg.get("sky_artifacts_prob", 0.0001)
    sky_values = noise_cfg.get("sky_values", [min_distance, min_distance + 0.1])
    sky_h = noise_cfg.get("sky_h", 3)
    for row in range(sky_h):
        mask = torch.rand(num_envs, device=device) < sky_prob
        if mask.any():
            val = sky_values[0] + torch.rand(mask.sum(), device=device) * (sky_values[1] - sky_values[0])
            for ci in range(w):
                depth[mask, row, ci] = val[: mask.sum()]

    return depth


def robot_body_sample_points(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_points: int = 100,
) -> torch.Tensor:
    """Sample points on the robot body for penetration depth computation.

    Uses the barrier track's body measurement points if available.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.
        num_points: Number of sample points (used as fallback).

    Returns:
        A tensor of shape (num_envs, num_points, 3) with sample point world positions.
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.zeros(env.num_envs, num_points, 3, device=env.device)

    return env.barrier_track.get_body_sample_points()
