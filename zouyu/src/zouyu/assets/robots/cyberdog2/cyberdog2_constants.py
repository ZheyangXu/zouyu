"""Unitree Cyberdog2 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

from zouyu.assets.robots.utils import update_assets

##
# MJCF and assets.
##
SRC_PATH = Path("/opt/zouyu-workspaces/zouyu/zouyu/src/zouyu")

CYBERDOG2_XML: Path = SRC_PATH / "assets" / "robots" / "cyberdog2" / "xmls" / "cyberdog2.xml"
assert CYBERDOG2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, CYBERDOG2_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CYBERDOG2_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config (URDF: effort=12.0, damping=0.01, friction=0.1,
# reduction=7.75).
##

CYBERDOG2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*hip_joint",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=12.0,
    armature=0.01,
)
CYBERDOG2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*thigh_joint",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=12.0,
    armature=0.01,
)
CYBERDOG2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*calf_joint",),
    stiffness=40.0,
    damping=2.0,
    effort_limit=12.0,
    armature=0.02,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={
        ".*hip_joint": 0.0,
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
    solimp={_foot_regex: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
)

##
# Final config.
##

CYBERDOG2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        CYBERDOG2_ACTUATOR_HIP,
        CYBERDOG2_ACTUATOR_THIGH,
        CYBERDOG2_ACTUATOR_CALF,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_cyberdog2_robot_cfg() -> EntityCfg:
    """Get a fresh Cyberdog2 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=CYBERDOG2_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_cyberdog2_robot_cfg())

    viewer.launch(robot.spec.compile())
