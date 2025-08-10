"""Core domain entities for legged robot parkour training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field


class SimulationEngine(Enum):
    """Supported simulation engines."""

    ISAAC_SIM = "isaac_sim"
    ISAAC_LAB = "isaac_lab"
    MUJOCO = "mujoco"


class TrackingBackend(Enum):
    """Supported tracking backends."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"


@dataclass
class RobotState:
    """Current state of the robot."""

    position: torch.Tensor  # [x, y, z]
    orientation: torch.Tensor  # [qx, qy, qz, qw]
    linear_velocity: torch.Tensor  # [vx, vy, vz]
    angular_velocity: torch.Tensor  # [wx, wy, wz]
    joint_positions: torch.Tensor  # Joint angles
    joint_velocities: torch.Tensor  # Joint angular velocities
    joint_torques: torch.Tensor  # Applied joint torques
    contact_forces: torch.Tensor  # Contact forces
    height_measurements: Optional[torch.Tensor] = None  # Terrain height measurements


@dataclass
class Action:
    """Action to be executed by the robot."""

    joint_targets: torch.Tensor  # Target joint positions/torques
    action_type: str = "position"  # "position", "velocity", "torque"


class RewardComponent(BaseModel):
    """Individual reward component configuration."""

    name: str
    weight: float  # Allow negative weights for penalty terms
    scale: float = 1.0
    enabled: bool = True


class RobotConfig(BaseModel):
    """Robot configuration model."""

    name: str
    urdf_path: str
    mujoco_path: Optional[str] = None  # Path to MuJoCo XML model
    num_joints: int
    joint_names: List[str]
    default_joint_positions: List[float]
    joint_limits: Dict[str, Tuple[float, float]]
    mass: float
    base_dimensions: List[float]  # [length, width, height]
    motor_strength: float = 1.0

    class Config:
        arbitrary_types_allowed = True


class EnvironmentConfig(BaseModel):
    """Environment configuration model."""

    name: str
    terrain_type: str
    terrain_size: Tuple[float, float]  # [length, width]
    num_envs: int = Field(gt=0)
    env_spacing: float = Field(gt=0.0)
    episode_length: int = Field(gt=0)
    reward_components: List[RewardComponent]
    observation_space_dim: int
    action_space_dim: int

    # Physics parameters
    gravity: List[float] = [0.0, 0.0, -9.81]
    dt: float = Field(gt=0.0, default=0.02)
    substeps: int = Field(gt=0, default=4)

    class Config:
        arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """Training configuration model."""

    algorithm: str = "PPO"
    num_iterations: int = Field(gt=0)
    num_steps_per_env: int = Field(gt=0)
    mini_batch_size: int = Field(gt=0)
    num_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    gamma: float = Field(ge=0.0, le=1.0, default=0.99)
    lam: float = Field(ge=0.0, le=1.0, default=0.95)
    clip_coef: float = Field(gt=0.0, default=0.2)
    entropy_coef: float = Field(ge=0.0, default=0.01)
    value_loss_coef: float = Field(ge=0.0, default=0.5)
    max_grad_norm: float = Field(ge=0.0, default=1.0)

    # Checkpointing
    save_interval: int = Field(gt=0, default=100)
    checkpoint_dir: str = "checkpoints"

    class Config:
        arbitrary_types_allowed = True


class SimulationBackend(ABC):
    """Abstract base class for simulation backends."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the simulation backend."""
        pass

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the simulation forward."""
        pass

    @abstractmethod
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments."""
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """Get current observations."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
