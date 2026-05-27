# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific reward functions for the Go2 parkour task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def energy_substeps(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize energy consumption computed as sum of |torque * velocity| per joint.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        A tensor of shape (num_envs,) with the energy penalty per environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def dof_error_named(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of named joints from their default positions.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration specifying target joints.

    Returns:
        A tensor of shape (num_envs,) with the L2 norm of joint position error.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dof_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(dof_error), dim=-1)


def exceeding_dof_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints exceeding their soft position limits.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        A tensor of shape (num_envs,) with the limit violation penalty per environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower_limits = soft_limits[..., 0]
    upper_limits = soft_limits[..., 1]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower_violation = torch.clamp(lower_limits - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - upper_limits, min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=-1)


def exceeding_torque_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints exceeding their torque limits as L1 norm.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        A tensor of shape (num_envs,) with the torque limit violation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    return torch.sum(torch.clamp(applied_torque - torque_limits, min=0.0), dim=-1)


def dof_vel_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joints exceeding their velocity limits.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        A tensor of shape (num_envs,) with the velocity limit violation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(torch.clamp(joint_vel - vel_limits, min=0.0), dim=-1)


def stand_still_when_cmd(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize joint deviation from default pose when velocity command is non-zero.

    This prevents the robot from "lazily" standing still instead of tracking active commands.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        asset_cfg: The scene entity configuration for the robot asset.
        velocity_threshold: Minimum command magnitude below which penalty is disabled.

    Returns:
        A tensor of shape (num_envs,) with the stand-still penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    dof_error = asset.data.joint_pos - asset.data.default_joint_pos
    penalty = torch.sum(torch.abs(dof_error), dim=1)
    return penalty * (cmd > velocity_threshold).float()


def lazy_stop(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize large stopping behavior: high velocity command but low actual velocity.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        asset_cfg: The scene entity configuration for the robot asset.
        threshold: The ratio threshold below which the penalty is applied.

    Returns:
        A tensor of shape (num_envs,) with the lazy stop penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_norm = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ratio = vel_norm / (cmd_norm + 1e-6)
    return torch.clamp(threshold - ratio, min=0.0) * (cmd_norm > 0.1).float()


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize undesired body contacts (shanks, thighs, base).

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor configuration for bodies to monitor.
        threshold: Force threshold above which contact is considered active.

    Returns:
        A tensor of shape (num_envs,) with the collision penalty.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    is_contact = torch.max(torch.norm(net_forces, dim=-1), dim=-1).values > threshold
    return is_contact.float()


def penetration_depth_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize virtual obstacle penetration for parkour tasks.

    Requires the environment to have a ``barrier_track`` attribute that provides
    ``get_penetration_depths()``.

    Args:
        env: The environment instance.
        threshold: Maximum penetration depth before penalty saturates.

    Returns:
        A tensor of shape (num_envs,) with the penetration penalty.
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.zeros(env.num_envs, device=env.device)
    depths = env.barrier_track.get_penetration_depths()
    return torch.clamp(depths - threshold, min=0.0)
