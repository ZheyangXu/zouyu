"""AMP-specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import AnimationTerm


def root_local_rot_tan_norm(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """6D tangent-normal encoding of the root's local rotation.

    The root quaternion is expressed in a heading-aligned frame so the
    encoding captures the tilt relative to the heading direction.

    Args:
        env: The environment.
        asset_cfg: Scene entity configuration.

    Returns:
        Concatenated tangent and normal vectors (N, 6).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    root_quat = robot.data.root_quat_w  # (N, 4)
    yaw_quat = math_utils.yaw_quat(root_quat)
    root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(yaw_quat), root_quat
    )
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    return torch.cat([tan_vec, norm_vec], dim=-1)


def key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """3D positions of key bodies (end-effectors) in the robot's local frame.

    Args:
        env: The environment.
        asset_cfg: Scene entity configuration specifying body_names.

    Returns:
        Flattened body positions (N, 3 * num_bodies).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    num_bodies = body_pos_w.shape[1]
    # Position relative to root in world frame
    rel_pos_w = body_pos_w - root_pos.unsqueeze(1)
    # Rotate to local (body) frame
    root_quat_conj = math_utils.quat_conjugate(root_quat)
    root_quat_conj = root_quat_conj.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_pos_b = math_utils.quat_apply(root_quat_conj, rel_pos_w)
    return rel_pos_b.reshape(env.num_envs, -1)

