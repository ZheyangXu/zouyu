"""AMP-specific reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward for feet air time, designed for bipedal gaits.

    Positive reward for feet being off the ground when velocity commands
    are non-zero (encouraging alternating foot lift).

    Args:
        env: The environment.
        command_name: Name of the command term.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Contact force threshold to consider a foot in contact.

    Returns:
        Reward tensor (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    first_contact = net_forces[:, 0, sensor_cfg.body_ids, :].norm(dim=-1) > threshold

    # Reward when exactly one foot is in the air (off-ground)
    air_time = torch.sum(~first_contact, dim=1) / float(len(sensor_cfg.body_ids))

    # Gate: only reward when velocity command is non-trivial
    command = env.command_manager.get_command(command_name)
    gate = torch.norm(command[:, :2], dim=1) > 0.1
    return air_time * gate.float()


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot sliding when in contact with the ground.

    Args:
        env: The environment.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Penalty tensor (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    in_contact = (
        contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
        .norm(dim=-1) > 1.0
    )
    foot_lin_vel_w = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]
    slide_speed = torch.norm(foot_lin_vel_w, dim=-1)
    return torch.sum(slide_speed * in_contact.float(), dim=-1)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when velocity command is small.

    Args:
        env: The environment.
        command_name: Name of the command term.
        command_threshold: Velocity magnitude below which to penalize deviation.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Penalty tensor (num_envs,).
    """
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (
        torch.norm(command[:, :2], dim=1) < command_threshold
    )


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward penalty for early termination.

    Args:
        env: The environment.

    Returns:
        Penalty tensor (num_envs,). 1.0 if terminated, 0.0 otherwise.
    """
    return env.termination_manager.terminated.float()
