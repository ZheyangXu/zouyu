# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2: Oracle parkour policy environment configuration for Go2.

Trains a privileged (oracle) parkour policy on BarrierTrack terrain with height-map
observations. Uses goal-based velocity commands, obstacle-conditioned termination
thresholds, and penetration-aware terrain curriculum.

This stage resumes from a Stage 1 walking checkpoint.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from luwu.tasks.locomotion.parkour import mdp

from .walk_env_cfg import Go2WalkEnvCfg


@configclass
class Go2FieldSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Go2 parkour field environment.

    Uses a flat base plane; the BarrierTrack obstacles are added as separate
    mesh prims during environment initialization via the BarrierTrack generator.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 25.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.025, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class Go2FieldEventCfg:
    """Domain randomization events for Go2 parkour field."""

    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
        },
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.05, 0.6),
                "y": (-0.25, 0.25),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
                "roll": (-0.75, 0.75),
                "pitch": (-0.75, 0.75),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-5.0, 5.0)},
    )

    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Go2FieldCommandsCfg:
    """Goal-based velocity commands for parkour field."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class Go2FieldObservationsCfg:
    """Observation configuration for parkour field (privileged oracle)."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy group: proprioception + height scan."""

        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=AdditiveUniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        joint_vel_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            clip=(-100, 100),
            scale=0.05,
            noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic group: clean proprioception + privileged oracle info."""

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))
        engaging_block = ObservationTermCfg(
            func=mdp.engaging_block_info,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2FieldRewardsCfg:
    """Reward configuration for parkour field (adjusted scales)."""

    # Velocity tracking
    track_lin_vel = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Penalties (reduced for parkour)
    energy = RewardTermCfg(func=mdp.energy_substeps, weight=-2e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    stand_still = RewardTermCfg(
        func=mdp.stand_still_when_cmd,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lazy_stop = RewardTermCfg(
        func=mdp.lazy_stop,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_error_hip = RewardTermCfg(
        func=mdp.dof_error_named,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"],
            )
        },
    )
    dof_error = RewardTermCfg(
        func=mdp.dof_error_named,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    exceed_dof_pos = RewardTermCfg(
        func=mdp.exceeding_dof_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    exceed_torque = RewardTermCfg(
        func=mdp.exceeding_torque_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    collision = RewardTermCfg(
        func=mdp.collision_penalty,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh", ".*_calf"]),
        },
    )
    penetration = RewardTermCfg(
        func=mdp.penetration_depth_penalty,
        weight=-0.05,
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class Go2FieldTerminationsCfg:
    """Termination configuration for parkour field (stricter orientation, border timeout)."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation_parkour,
        params={"roll_threshold": 1.4, "pitch_threshold": 1.6},
    )
    border_timeout = TerminationTermCfg(
        func=mdp.timeout_at_border,
        params={"border_margin": 0.5},
    )


@configclass
class Go2FieldCurriculumCfg:
    """Penetration-based terrain curriculum for parkour field."""

    penetration_curriculum = CurriculumTermCfg(
        func=mdp.penetration_curriculum,
        params={
            "harder_threshold": 100.0,
            "easier_threshold": 200.0,
            "no_moveup_when_fall": True,
        },
    )


@configclass
class Go2FieldEnvCfg(Go2WalkEnvCfg):
    """Stage 2: Go2 parkour field (oracle) environment configuration.

    Inherits base settings from walk and overrides terrain, commands,
    observations, rewards, terminations, and curriculum for parkour training.
    """

    scene: Go2FieldSceneCfg = Go2FieldSceneCfg(num_envs=4096, env_spacing=3.0)
    events: Go2FieldEventCfg = Go2FieldEventCfg()
    commands: Go2FieldCommandsCfg = Go2FieldCommandsCfg()
    observations: Go2FieldObservationsCfg = Go2FieldObservationsCfg()
    rewards: Go2FieldRewardsCfg = Go2FieldRewardsCfg()
    terminations: Go2FieldTerminationsCfg = Go2FieldTerminationsCfg()
    curriculum: Go2FieldCurriculumCfg = Go2FieldCurriculumCfg()

    def __post_init__(self) -> None:
        """Post-initialization: set parkour-specific parameters."""
        super().__post_init__()

        # BarrierTrack terrain integration flag
        # When True, the BarrierTrack mesh is imported during env setup
        self.use_barrier_track = True
