"""G1-specific AMP locomotion environment configuration."""

import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import luwu.tasks.locomotion.amp.mdp as mdp
from luwu.assets.robots import UNITREE_G1_29DOF_CFG
from luwu.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

# Key body names for end-effector position observations.
# Order must be consistent between policy, critic, discriminator, and
# reference observation groups.
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

AMP_NUM_STEPS = 4

# Motion dataset path (walk_and_run)
_MOTION_DATA_DIR = (
    "/opt/zouyu-workspaces/amp/legged_lab/source/legged_lab/legged_lab/"
    "data/MotionData/g1_29dof/amp/walk_and_run"
)


@configclass
class G1AmpRewards:
    """G1-specific reward configuration."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP locomotion environment."""

    rewards: G1AmpRewards = G1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot ----
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # ---- Observations: set key body params and history lengths ----
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", body_names=KEY_BODY_NAMES, preserve_order=True
            )
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", body_names=KEY_BODY_NAMES, preserve_order=True
            )
        }

        # ---- Events ----
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "torso_link"
        ]
        self.events.reset_from_ref.params = {
            "height_offset": 0.1,
        }

        # ---- Commands ----
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ---- Curriculum ----
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ---- Terminations ----
        self.terminations.base_contact = None


@configclass
class G1AmpEnvCfgPlay(G1AmpEnvCfg):
    """Configuration for the G1 AMP environment (playback/evaluation)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
