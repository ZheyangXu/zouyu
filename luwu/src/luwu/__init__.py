"""LuWu: Advanced Legged Robot Parkour Training System.

A unified framework for training legged robots in parkour environments
supporting multiple simulation engines (Isaac Sim/Lab, MuJoCo) and
Gymnasium-compatible APIs.
"""

__version__ = "0.1.0"
__author__ = "ZheyangXu"
__email__ = "jishengxzy@hotmail.com"

from luwu.domain.entities import (
    EnvironmentConfig,
    RobotConfig,
    SimulationEngine,
    TrainingConfig,
)
from luwu.infrastructure.config import config_manager

__all__ = [
    "EnvironmentConfig",
    "RobotConfig",
    "SimulationEngine",
    "TrainingConfig",
    "config_manager",
]
