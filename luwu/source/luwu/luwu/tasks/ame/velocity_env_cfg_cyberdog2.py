import math

# import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

import luwu.tasks.ame.terrains as terrain_gen
from luwu.assets.robots.cyberdog2 import CYBERDOG2_CFG as ROBOT_CFG
from luwu.tasks.ame.terrains.finetune_terrain_cfg import (
    FINETUNE_ROUGH_TERRAINS_CFG,
)

from luwu.tasks.ame.terrains.terrain_cfg import (  # isort: skip
    ROUGH_TERRAINS_CFG,
)

FINETUNE = False


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=(
            FINETUNE_ROUGH_TERRAINS_CFG if FINETUNE else ROUGH_TERRAINS_CFG
        ),
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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 25.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05, size=[1.6, 1.0]
        ),  # 0.05m resolution, 1.6m x 1.0m, grid 33x21
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),   # 0.1m resolution, 1.6m x 1.0m, grid 17x11
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
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
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # base_lin_vel = ObservationTermCfg(
        #     func=mdp.base_lin_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1)
        # )
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
            noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            clip=(-100, 100),
            scale=0.05,
            noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5),
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))
        height_scan = ObservationTermCfg(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for critic group."""

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel, clip=(-100, 100), scale=0.2
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity, clip=(-100, 100)
        )
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
        )
        joint_vel_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel, clip=(-100, 100), scale=0.05
        )
        joint_effort = ObservationTermCfg(
            func=mdp.joint_effort, clip=(-100, 100), scale=0.01
        )
        last_action = ObservationTermCfg(func=mdp.last_action, clip=(-100, 100))
        height_scan = ObservationTermCfg(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events"""

    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    body_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # penalties
    # base_linear_velocity = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "head",
                    "FR_hip",
                    "FL_hip",
                    "RR_hip",
                    "RL_hip",
                    "FR_thigh",
                    "FL_thigh",
                    "RR_thigh",
                    "RL_thigh",
                    "FR_calf",
                    "FL_calf",
                    "RR_calf",
                    "RL_calf",
                ],
            ),
        },
    )
    dof_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
    )
    dof_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
    )
    # dof_vel_l2 = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )
    dof_torques_limits = RewardTermCfg(
        func=mdp.applied_torque_limits,
        weight=-0.01,
    )
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0)

    # style
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_air_time_variance = RewardTermCfg(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    feet_stumble = RewardTermCfg(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
        },
    )
    feet_too_near = RewardTermCfg(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    # bad_orientation = TerminationTermCfg(
    #     func=mdp.bad_orientation, params={"limit_angle": 10}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurriculumTermCfg(func=mdp.lin_vel_cmd_levels)


@configclass
class Cyberdog2RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity tracking environment."""

    # Scene
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # basic
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # self.scene.contact_forces.update_period = self.sim.dt
        # self.scene.height_scanner.update_period = self.sim.dt * self.decimation

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Cyberdog2RobotPlayEnvCfg(Cyberdog2RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class Cyberdog2RobotRoughEnvCfg(Cyberdog2RobotEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
            0.025,
            0.1,
        )
        self.scene.terrain.terrain_generator.sub_terrains[
            "random_rough"
        ].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = (
            0.01
        )

        # reduce action scale
        self.actions.JointPositionAction.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.body_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        self.rewards.dof_pos_limits.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Cyberdog2RobotRoughPlayEnvCfg(Cyberdog2RobotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # add visualization camera only for play
        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/visualize_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 3.0), rot=(0.707, 0.0, 0.707, 0.0), convention="world"
            ),
        )
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains = {
                # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                #     proportion=0.3,
                #     step_height_range=(0.15, 0.15),
                #     step_width=0.4,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                #     proportion=0.3,
                #     step_height_range=(0.15, 0.15),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                #     proportion=0.2, grid_width=0.45, grid_height_range=(0.1, 0.1), platform_width=2.0
                # ),
                # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                #     proportion=0.1, noise_range=(0.02, 0.1), noise_step=0.01, border_width=0.25
                # ),
                # "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
                #     proportion=1.0, stone_height_max=0.0, stone_width_range=(0.3, 0.3), stone_distance_range=(0.2, 0.2), platform_width=2.0,
                #     holes_depth=-2.0, border_width=0.25
                # ),
                # "stonebridge": terrain_gen.HfStonesBridgeTerrainCfg(
                #     proportion=1.0, platform_width=2.0, border_width=0.25, holes_depth=-2.0,
                #     stone_height_max=0.0, stone_width_range=(0.3, 0.3), stone_distance_range=(0.2, 0.2),
                #     stone_length_range=(0.4, 0.4), stone_lateral_distance_range=(0.0, 0.0)
                # ),
                "stakes": terrain_gen.HfAlternateColumnStakesTerrainCfg(
                    proportion=0.5,
                    stake_height_max=0.0,
                    stake_side_range=(0.2, 0.2),
                    stake_gap_range=(0.3, 0.3),
                    column_gap_range=(0.3, 0.3),
                    column_jitter=0.0,
                    holes_depth=-2.0,
                    platform_width=2.0,
                    border_width=0.25,
                ),
                # "hf_gaps": terrain_gen.HfConcentricGapTerrainCfg(
                #             proportion=0.5, gap_width_range=(0.5, 0.5), platform_width=2.0, border_width=0.25, gap_depth=-1.0,
                #             ground_width_range=(0.5, 0.5), ground_height_max=0.0
                # ),
                # "rails": terrain_gen.MeshRailsTerrainCfg(
                #     proportion=0.1, rail_height_range=(0.30, 0.30), rail_thickness_range=(0.3, 0.3), platform_width=2.0
                # ),
            }

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
