from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class Cyberdog2ActuatorCfg(DelayedPDActuatorCfg):
    """
    Configuration for actuators.
    """

    class_type: type = DelayedPDActuator

    X1: float = 1e9
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = 1e9
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float | None = None
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""

    Fs: float = 0.0
    """ Static friction coefficient """

    Fd: float = 0.0
    """ Dynamic friction coefficient """

    Va: float = 0.01
    """ Velocity at which the friction is fully activated """


@configclass
class Cyberdog2ActuatorCfg_Cyberdog2HV(Cyberdog2ActuatorCfg):
    X1 = 12.3
    X2 = 27.0
    Y1 = 20.0
    Y2 = 23.0


@configclass
class Cyberdog2ArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor: float = 0.9


@configclass
class Cyberdog2UsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
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
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.66,
            "FR_calf_joint": -1.17,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.66,
            "FL_calf_joint": -1.17,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.66,
            "RR_calf_joint": -1.17,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.66,
            "RL_calf_joint": -1.17,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "CYBERDOG2HV": Cyberdog2ActuatorCfg_Cyberdog2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    joint_sdk_names=[
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
)
