# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: Walking policy environment configuration for Go2 parkour.

Trains a basic locomotion policy on flat + mild roughness terrain using
proprioceptive observations and standard velocity commands. This serves
as the pre-trained starting point for the parkour field policy.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
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

# Terrain config: mild roughness for walking baseline
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class Go2WalkSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Go2 walking environment."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
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
class Go2WalkEventCfg:
    """Domain randomization events for Go2 walking."""

    # Startup randomization
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

    # Reset events
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

    # Interval events
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Go2WalkCommandsCfg:
    """Velocity command configuration for Go2 walking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class Go2WalkActionsCfg:
    """Action configuration for Go2 walking (joint position PD control)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class Go2WalkObservationsCfg:
    """Observation configuration for Go2 walking (proprioceptive + height scan)."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy group: noisy proprioception + height scan."""

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
        """Critic group: clean proprioception + height scan (privileged)."""

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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WalkRewardsCfg:
    """Reward configuration for Go2 walking."""

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

    # Penalties
    energy = RewardTermCfg(func=mdp.energy_substeps, weight=-2e-5, params={"asset_cfg": SceneEntityCfg("robot")})
    stand_still = RewardTermCfg(
        func=mdp.stand_still_when_cmd,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_error_hip = RewardTermCfg(
        func=mdp.dof_error_named,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FR_hip_joint",
                    "FL_hip_joint",
                    "RR_hip_joint",
                    "RL_hip_joint",
                ],
            )
        },
    )
    dof_error = RewardTermCfg(
        func=mdp.dof_error_named,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    exceed_dof_pos = RewardTermCfg(
        func=mdp.exceeding_dof_pos_limits,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    exceed_torque = RewardTermCfg(
        func=mdp.exceeding_torque_limits,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_vel_limit = RewardTermCfg(
        func=mdp.dof_vel_limits,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class Go2WalkTerminationsCfg:
    """Termination configuration for Go2 walking."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 1.0},  # ~3.0 rad roll/pitch
    )


@configclass
class Go2WalkEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: Go2 walking environment configuration."""

    scene: Go2WalkSceneCfg = Go2WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    events: Go2WalkEventCfg = Go2WalkEventCfg()
    commands: Go2WalkCommandsCfg = Go2WalkCommandsCfg()
    actions: Go2WalkActionsCfg = Go2WalkActionsCfg()
    observations: Go2WalkObservationsCfg = Go2WalkObservationsCfg()
    rewards: Go2WalkRewardsCfg = Go2WalkRewardsCfg()
    terminations: Go2WalkTerminationsCfg = Go2WalkTerminationsCfg()

    def __post_init__(self) -> None:
        """Post-initialization: set simulation and control parameters."""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Scale down terrains for Go2 size
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
