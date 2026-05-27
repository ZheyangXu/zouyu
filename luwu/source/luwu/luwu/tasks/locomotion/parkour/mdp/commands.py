# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-based velocity command generation for parkour tasks.

Extends IsaacLab's uniform velocity command with a hybrid mode where forward velocity
is sampled from a range and lateral/yaw velocities are commanded as ratios relative to
the forward direction. Commands re-sample when the robot engages a new obstacle block.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

logger = logging.getLogger(__name__)


@configclass
class GoalBasedVelocityCommandCfg:
    """Configuration for goal-based (hybrid) velocity commands.

    In goal-based mode, forward velocity is sampled from ``lin_vel_x`` range,
    while lateral and yaw velocities are commanded as ratios relative to the
    forward move direction. This mirrors the original parkour command system.
    """

    @configclass
    class GoalBasedRanges:
        """Ratio-based ranges for goal-conditioned commands."""

        x_ratio: float | None = None
        """Ratio for forward velocity. None means use the lin_vel_x range directly."""

        y_ratio: float = 1.2
        """Ratio of lateral to forward velocity."""

        yaw_ratio: float = 1.0
        """Ratio of yaw rate to forward velocity."""

    is_goal_based: bool = False
    """Whether to use goal-based (ratio) command mode."""

    goal_based: GoalBasedRanges = GoalBasedRanges()
    """Goal-based range configuration."""

    follow_cmd_cutoff: bool = False
    """If True, re-sample when engaging a new obstacle block."""

    x_stop_by_yaw_threshold: float = 1.0
    """Yaw deviation threshold (rad) above which forward command is set to zero."""

    lin_cmd_cutoff: float = 0.2
    """Linear velocity cutoff below which velocity is set to zero."""

    ang_cmd_cutoff: float = 0.2
    """Angular velocity cutoff below which angular velocity is set to zero."""


class GoalBasedVelocityCommand(UniformVelocityCommand):
    """Hybrid goal-based velocity command generator for parkour tasks.

    Forward velocity is sampled uniformly from ``lin_vel_x`` range.
    Lateral and yaw velocities are computed as ratios of the forward velocity.
    Commands are re-sampled when the robot engages a new obstacle block
    (tracked via the environment's ``barrier_track`` attribute).
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv) -> None:
        """Initialize the command generator.

        Args:
            cfg: The command configuration (must include goal-based fields).
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Track the current block index per env for re-sampling on engage
        self._current_block_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample velocity commands in goal-based hybrid mode.

        Args:
            env_ids: The environment IDs to resample commands for.
        """
        r = torch.empty(len(env_ids), device=self.device)

        if getattr(self.cfg, "is_goal_based", False):
            goal_cfg = self.cfg.goal_based
            # Forward velocity from configured range
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)

            # Lateral velocity as ratio of forward
            y_ratio = goal_cfg.y_ratio if goal_cfg.y_ratio is not None else 1.0
            self.vel_command_b[env_ids, 1] = r.uniform_(0.0, 1.0) * y_ratio * self.vel_command_b[env_ids, 0]

            # Yaw velocity as ratio of forward
            yaw_ratio = goal_cfg.yaw_ratio if goal_cfg.yaw_ratio is not None else 1.0
            self.vel_command_b[env_ids, 2] = r.uniform_(0.0, 1.0) * yaw_ratio * self.vel_command_b[env_ids, 0]
        else:
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Apply cutoffs
        lin_cutoff = getattr(self.cfg, "lin_cmd_cutoff", 0.0)
        ang_cutoff = getattr(self.cfg, "ang_cmd_cutoff", 0.0)
        self.vel_command_b[env_ids, 0] = torch.where(
            self.vel_command_b[env_ids, 0].abs() < lin_cutoff,
            torch.zeros_like(self.vel_command_b[env_ids, 0]),
            self.vel_command_b[env_ids, 0],
        )
        self.vel_command_b[env_ids, 2] = torch.where(
            self.vel_command_b[env_ids, 2].abs() < ang_cutoff,
            torch.zeros_like(self.vel_command_b[env_ids, 2]),
            self.vel_command_b[env_ids, 2],
        )

        # Standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        """Post-process commands: handle standing and obstacle re-sampling."""
        # Re-sample on new block engagement
        if getattr(self.cfg, "follow_cmd_cutoff", False):
            if hasattr(self._env, "barrier_track") and self._env.barrier_track is not None:
                new_block_idx = self._env.barrier_track.get_current_block_indices()
                changed = new_block_idx != self._current_block_idx
                if changed.any():
                    env_ids = changed.nonzero(as_tuple=False).flatten()
                    self._resample_command(env_ids)
                    self._current_block_idx = new_block_idx.clone()

        # Standing envs get zero command
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
