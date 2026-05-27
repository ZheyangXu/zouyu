# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific domain randomization events for the Go2 parkour task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def randomize_motor_strength(
    env: ManagerBasedRLEnv,
    env_ids: tuple[int, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    strength_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize motor strength (joint stiffness multiplier) per environment.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply randomization to.
        asset_cfg: The scene entity configuration for the robot asset.
        strength_range: (min, max) multiplier range.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    strength = torch.empty(len(env_ids), device=env.device).uniform_(*strength_range)
    for i, env_id in enumerate(env_ids):
        asset.write_joint_stiffness_to_sim(
            asset.data.default_joint_stiffness[asset_cfg.joint_ids] * strength[i],
            joint_ids=asset_cfg.joint_ids,
            env_ids=[env_id],
        )


def randomize_init_base_pose(
    env: ManagerBasedRLEnv,
    env_ids: tuple[int, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_range: dict[str, tuple[float, float]] | None = None,
    rot_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize initial base position and rotation per environment.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply randomization to.
        asset_cfg: The scene entity configuration for the robot asset.
        pos_range: Per-axis position ranges. Default: x=(0.05, 0.6), y=(-0.25, 0.25).
        rot_range: Per-axis rotation ranges. Default: roll=(-0.75,0.75), pitch=(-0.75,0.75).
    """
    if pos_range is None:
        pos_range = {"x": (0.05, 0.6), "y": (-0.25, 0.25)}
    if rot_range is None:
        rot_range = {"roll": (-0.75, 0.75), "pitch": (-0.75, 0.75)}

    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    # Position offsets
    root_pos = asset.data.root_pos_w[env_ids].clone()
    if "x" in pos_range:
        root_pos[:, 0] += torch.empty(n, device=device).uniform_(*pos_range["x"])
    if "y" in pos_range:
        root_pos[:, 1] += torch.empty(n, device=device).uniform_(*pos_range["y"])

    # Rotation offsets (applied as incremental euler rotations)
    import isaaclab.utils.math as math_utils

    roll = torch.empty(n, device=device).uniform_(*rot_range.get("roll", (0.0, 0.0)))
    pitch = torch.empty(n, device=device).uniform_(*rot_range.get("pitch", (0.0, 0.0)))
    yaw = torch.empty(n, device=device).uniform_(*rot_range.get("yaw", (-3.14, 3.14)))
    delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    root_quat = asset.data.root_quat_w[env_ids]
    new_quat = math_utils.quat_mul(root_quat, delta_quat)

    asset.write_root_pose_to_sim(
        torch.cat([root_pos, new_quat], dim=-1),
        env_ids=env_ids,
    )


def randomize_init_dof_vel(
    env: ManagerBasedRLEnv,
    env_ids: tuple[int, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_range: tuple[float, float] = (-5.0, 5.0),
) -> None:
    """Randomize initial joint velocities.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply randomization to.
        asset_cfg: The scene entity configuration for the robot asset.
        vel_range: (min, max) velocity range in rad/s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    vel = torch.empty(n, len(asset_cfg.joint_ids), device=env.device).uniform_(*vel_range)
    asset.write_joint_velocity_to_sim(vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
