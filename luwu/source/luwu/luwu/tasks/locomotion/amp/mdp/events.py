"""AMP-specific event functions."""

from __future__ import annotations


import torch
from isaaclab.managers import SceneEntityCfg

# Re-export standard events from Isaac Lab
from isaaclab.envs.mdp import (
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    apply_external_force_torque,
    push_by_setting_velocity,
)
from isaaclab.envs import ManagerBasedEnv


def reset_from_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    height_offset: float = 0.0,
) -> torch.Tensor:
    """Reset robot joints to default pose with a configurable height offset.

    This is a simplified version of reference state initialization. For full
    AMP-style reference state initialization (AMP paper Section 6.3), the
    AnimationManager needs to be available.

    Args:
        env: The environment.
        env_ids: Indices of environments to reset.
        height_offset: Additional height offset applied to the root position.

    Returns:
        Reset success indicator tensor (num_envs,).
    """
    robot = env.scene["robot"]
    num_ids = len(env_ids)

    root_pose = robot.data.root_state_w[env_ids, :7].clone()
    root_pose[:, 2] += height_offset
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_velocity_to_sim(
        torch.zeros(num_ids, 6, device=env.device), env_ids=env_ids
    )

    return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
