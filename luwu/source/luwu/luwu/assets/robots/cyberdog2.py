import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

CYBERDOG2_JOINT_SDK_NAMES = [
    "FR_abad_joint",
    "FR_hip_joint",
    "FR_knee_joint",
    "FL_abad_joint",
    "FL_hip_joint",
    "FL_knee_joint",
    "RR_abad_joint",
    "RR_hip_joint",
    "RR_knee_joint",
    "RL_abad_joint",
    "RL_hip_joint",
    "RL_knee_joint",
]


@configclass
class Cyberdog2ArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None


@configclass
class Cyberdog2UsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    merge_fixed_joints: bool = False
    collision_from_visuals: bool = False
    replace_cylinders_with_capsules: bool = True
    self_collision: bool = False
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


CYBERDOG2_CFG = Cyberdog2ArticulationCfg(
    spawn=Cyberdog2UsdFileCfg(
        usd_path="/opt/zouyu-workspaces/zouyu/assets/cyberdog_robot/cyberdog2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            "FR_abad_joint": -0.10,
            "FL_abad_joint": 0.10,
            "RR_abad_joint": -0.10,
            "RL_abad_joint": 0.10,
            "F.*_hip_joint": 0.80,
            "R.*_hip_joint": 1.00,
            ".*_knee_joint": -1.50,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_abad_joint", ".*_hip_joint", ".*_knee_joint"],
            effort_limit=12.0,
            effort_limit_sim=12.0,
            saturation_effort=12.0,
            velocity_limit=30.9971,
            velocity_limit_sim=30.9971,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
    joint_sdk_names=CYBERDOG2_JOINT_SDK_NAMES,
)
