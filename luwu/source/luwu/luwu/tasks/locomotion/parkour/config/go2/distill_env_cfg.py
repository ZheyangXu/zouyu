# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3: Distilled vision-based parkour policy environment configuration for Go2.

Trains a deployable vision-based policy that mimics the Stage 2 oracle policy
using a forward depth camera. The privileged critic still accesses height-map
observations. Uses teacher-student distillation with behavior cloning loss.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from luwu.tasks.locomotion.parkour import mdp

from .field_env_cfg import Go2FieldEnvCfg


@configclass
class Go2DistillSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Go2 distilled parkour environment.

    Includes a forward depth camera for vision-based policy training.
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

    # Forward depth camera for vision-based policy
    forward_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/forward_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.24, -0.0175, 0.12),
            rot=(0.0, 0.0, 0.0),
        ),
        update_period=0.1,  # 10 Hz camera
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.0,
            horizontal_aperture=1.2,
            clipping_range=(0.05, 3.0),
        ),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class Go2DistillEventCfg:
    """Domain randomization events for Go2 distilled parkour.

    Adds camera latency and position noise on top of standard DR.
    """

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
class Go2DistillObservationsCfg:
    """Observation configuration for distilled parkour.

    Policy receives forward depth instead of height scan.
    Critic retains privileged height scan for value estimation.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy group: proprioception + forward depth camera (noisy)."""

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
        forward_depth = ObservationTermCfg(
            func=mdp.forward_depth,
            params={"sensor_cfg": SceneEntityCfg("forward_camera"), "apply_noise": True},
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic group: clean proprioception + privileged height scan."""

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
        engaging_block = ObservationTermCfg(
            func=mdp.engaging_block_info,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100, 100),
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2DistillEnvCfg(Go2FieldEnvCfg):
    """Stage 3: Go2 distilled vision-based parkour environment configuration.

    Inherits from field config and overrides scene (adds camera), observations
    (depth camera instead of height scan for policy), and reduces env count.
    Curriculum is disabled since the teacher provides the learning signal.
    """

    scene: Go2DistillSceneCfg = Go2DistillSceneCfg(num_envs=256, env_spacing=3.0)
    events: Go2DistillEventCfg = Go2DistillEventCfg()
    observations: Go2DistillObservationsCfg = Go2DistillObservationsCfg()

    # Disable curriculum for distillation (teacher provides the signal)
    curriculum = None

    # Depth camera noise configuration for realistic stereo simulation
    depth_noise_cfg: dict = {
        "stereo_min_distance": 0.175,
        "stereo_far_distance": 1.2,
        "stereo_far_noise_std": 0.08,
        "stereo_near_noise_std": 0.02,
        "full_block_prob": 0.008,
        "half_block_spark_prob": 0.02,
        "sky_artifacts_prob": 0.0001,
        "sky_values": [0.175, 0.275],
        "sky_h": 3,
    }

    def __post_init__(self) -> None:
        """Post-initialization: set distillation-specific parameters."""
        super().__post_init__()

        self.use_barrier_track = True

        # Camera noise: activate depth noise in observation
        if hasattr(self.scene, "forward_camera"):
            self.scene.forward_camera.update_period = 0.1
