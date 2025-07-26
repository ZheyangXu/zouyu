"""
Luwu - Advanced Legged Robot Parkour Training with Isaac Sim and Isaac Lab.

This package provides a modular and extensible framework for training
legged robots to perform parkour tasks using reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "ZheyangXu"
__email__ = "jishengxzy@hotmail.com"

from luwu.application.services import ApplicationService
from luwu.infrastructure.config import ConfigManager, config_manager

__all__ = [
    "ApplicationService",
    "ConfigManager",
    "config_manager",
]
