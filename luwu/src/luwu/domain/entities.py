"""
Core domain entities for the legged robot system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class RobotType(Enum):
    """Supported robot types."""

    A1 = "a1"
    GO1 = "go1"
    ANYMAL_B = "anymal_b"
    ANYMAL_C = "anymal_c"


class ControlMode(Enum):
    """Control modes for the robot."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


class TrackingBackend(Enum):
    """Experiment tracking backends."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    BOTH = "both"


@dataclass
class RobotState:
    """Current state of the robot."""

    # Base state
    base_position: np.ndarray  # [x, y, z]
    base_orientation: np.ndarray  # quaternion [x, y, z, w]
    base_linear_velocity: np.ndarray  # [vx, vy, vz]
    base_angular_velocity: np.ndarray  # [wx, wy, wz]

    # Joint state
    joint_positions: np.ndarray  # joint angles [rad]
    joint_velocities: np.ndarray  # joint velocities [rad/s]
    joint_torques: np.ndarray  # joint torques [Nm]

    # Contact state
    foot_contacts: np.ndarray  # boolean array for foot contacts
    contact_forces: np.ndarray  # forces at contact points

    # Sensor data
    imu_data: Optional[np.ndarray] = None
    height_measurements: Optional[np.ndarray] = None

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return len(self.joint_positions)

    @property
    def num_feet(self) -> int:
        """Number of feet."""
        return len(self.foot_contacts)


@dataclass
class Command:
    """Command for the robot."""

    linear_velocity: np.ndarray  # desired linear velocity [vx, vy, vz]
    angular_velocity: np.ndarray  # desired angular velocity [wx, wy, wz]
    body_height: float = 0.0  # desired body height
    footstep_targets: Optional[np.ndarray] = None  # target footstep positions


@dataclass
class Action:
    """Action to be executed by the robot."""

    values: np.ndarray  # action values (interpretation depends on control mode)
    control_mode: ControlMode

    def clip(self, min_val: float = -1.0, max_val: float = 1.0) -> "Action":
        """Clip action values to the specified range."""
        clipped_values = np.clip(self.values, min_val, max_val)
        return Action(values=clipped_values, control_mode=self.control_mode)


@dataclass
class TrainingMetrics:
    """Training metrics for tracking."""

    episode_reward: float
    episode_length: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    explained_variance: float
    clipfrac: float
    approx_kl: float
    learning_rate: float

    # Reward components
    reward_components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        base_metrics = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "explained_variance": self.explained_variance,
            "clipfrac": self.clipfrac,
            "approx_kl": self.approx_kl,
            "learning_rate": self.learning_rate,
        }

        # Add reward components with prefix
        for key, value in self.reward_components.items():
            base_metrics[f"reward/{key}"] = value

        return base_metrics


class Robot(ABC):
    """Abstract base class for robots."""

    def __init__(self, robot_type: RobotType, config: Dict[str, Any]) -> None:
        """Initialize the robot.

        Args:
            robot_type: Type of the robot
            config: Robot configuration
        """
        self.robot_type = robot_type
        self.config = config
        self._state: Optional[RobotState] = None

    @property
    def state(self) -> Optional[RobotState]:
        """Current robot state."""
        return self._state

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get current observation from the robot.

        Returns:
            Observation array
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> RobotState:
        """Execute an action and return the new state.

        Args:
            action: Action to execute

        Returns:
            New robot state
        """
        pass

    @abstractmethod
    def reset(self) -> RobotState:
        """Reset the robot to its initial state.

        Returns:
            Initial robot state
        """
        pass

    @abstractmethod
    def get_reward(self, command: Command) -> Tuple[float, Dict[str, float]]:
        """Calculate reward based on current state and command.

        Args:
            command: Current command

        Returns:
            Tuple of (total_reward, reward_components)
        """
        pass


class Environment(ABC):
    """Abstract base class for environments."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self._robots: List[Robot] = []

    @property
    def robots(self) -> List[Robot]:
        """List of robots in the environment."""
        return self._robots

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return len(self._robots)

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset all environments.

        Returns:
            Initial observations
        """
        pass

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Step all environments.

        Args:
            actions: Actions for all environments

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.

        Args:
            mode: Rendering mode

        Returns:
            Rendered image if mode is "rgb_array"
        """
        pass


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def predict(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict actions and values.

        Args:
            observations: Input observations

        Returns:
            Tuple of (actions, values)
        """
        pass

    @abstractmethod
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> TrainingMetrics:
        """Update the policy.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            dones: Batch of done flags
            values: Batch of values

        Returns:
            Training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the policy.

        Args:
            path: Path to save the policy
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the policy.

        Args:
            path: Path to load the policy from
        """
        pass


class Tracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize a new tracking run.

        Args:
            project_name: Name of the project
            run_name: Name of the run
            config: Configuration to log
            tags: Tags for the run
            notes: Notes for the run
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Metrics to log
            step: Step number
        """
        pass

    @abstractmethod
    def log_video(self, video: np.ndarray, name: str, step: Optional[int] = None) -> None:
        """Log a video.

        Args:
            video: Video array
            name: Name of the video
            step: Step number
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish the tracking run."""
        pass
