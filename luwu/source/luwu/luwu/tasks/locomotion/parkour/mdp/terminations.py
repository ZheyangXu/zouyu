# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific termination conditions.

Includes obstacle-conditioned termination thresholds and border timeouts
for the barrier track environment.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def bad_orientation_parkour(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_threshold: float = 1.4,
    pitch_threshold: float = 1.6,
) -> torch.Tensor:
    """Terminate when the robot's roll or pitch exceeds parkour-specific thresholds.

    The parkour thresholds (1.4 rad roll, 1.6 rad pitch) are tighter than the
    standard locomotion thresholds because the robot needs to maintain stable
    orientation on challenging obstacles.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.
        roll_threshold: Maximum allowed roll angle in radians.
        pitch_threshold: Maximum allowed pitch angle in radians.

    Returns:
        A boolean tensor of shape (num_envs,) indicating termination.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    roll = torch.atan2(
        asset.data.projected_gravity_b[:, 1],
        asset.data.projected_gravity_b[:, 2],
    )
    pitch = torch.atan2(
        -asset.data.projected_gravity_b[:, 0],
        torch.sqrt(asset.data.projected_gravity_b[:, 1] ** 2 + asset.data.projected_gravity_b[:, 2] ** 2),
    )
    return torch.logical_or(torch.abs(roll) > roll_threshold, torch.abs(pitch) > pitch_threshold)


def timeout_at_border(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    border_margin: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot moves outside the barrier track border.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.
        border_margin: Margin from the border before termination triggers.

    Returns:
        A boolean tensor of shape (num_envs,) indicating termination.
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    return env.barrier_track.is_out_of_bounds(asset.data.root_pos_w, margin=border_margin)
