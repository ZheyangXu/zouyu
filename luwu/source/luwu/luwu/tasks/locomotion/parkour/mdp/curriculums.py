# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Penetration-based terrain curriculum for parkour tasks.

Increases terrain difficulty (more challenging obstacles) when the robot achieves
low penetration depth, and decreases difficulty when penetration is too high or
the robot falls.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def penetration_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    harder_threshold: float = 100.0,
    easier_threshold: float = 200.0,
    no_moveup_when_fall: bool = True,
) -> torch.Tensor:
    """Adjust terrain difficulty based on penetration metrics.

    If the average penetration depth is below ``harder_threshold``, increase
    terrain difficulty. If above ``easier_threshold`` (or robot fell), decrease it.

    Args:
        env: The environment instance.
        env_ids: The environment IDs being reset.
        harder_threshold: Penetration sum below which difficulty increases.
        easier_threshold: Penetration sum above which difficulty decreases.
        no_moveup_when_fall: If True, prevent difficulty increase when the robot falls.

    Returns:
        The current terrain level (scalar).
    """
    if not hasattr(env, "barrier_track") or env.barrier_track is None:
        return torch.tensor(0.0, device=env.device)

    penetration = env.barrier_track.get_penetration_depths()

    if env.common_step_counter % env.max_episode_length == 0:
        episode_penetration = penetration[env_ids].sum()

        if episode_penetration < harder_threshold:
            if no_moveup_when_fall:
                # Check if any env in batch fell (terminated due to orientation)
                if not env.termination_manager._reset_buf[env_ids].any():
                    env.barrier_track.increase_difficulty(env_ids)
            else:
                env.barrier_track.increase_difficulty(env_ids)
        elif episode_penetration > easier_threshold:
            env.barrier_track.decrease_difficulty(env_ids)

    return env.barrier_track.get_difficulty_levels().float().mean()
