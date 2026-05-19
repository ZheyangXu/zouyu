"""Robot asset configurations for Luwu."""

import os

LUWU_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from luwu.assets.robots.unitree_g1 import UNITREE_G1_29DOF_CFG

__all__ = ["UNITREE_G1_29DOF_CFG", "LUWU_ROOT_DIR"]
