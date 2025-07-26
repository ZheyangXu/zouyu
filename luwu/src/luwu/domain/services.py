"""
Domain services for business logic.
"""

from typing import Dict, List, Optional, Tuple

from luwu.domain.entities import Command, Robot, RobotState, RobotType, TrainingMetrics
from luwu.domain.repositories import ConfigRepository, RobotRepository


class RobotService:
    """Service for robot-related operations."""

    def __init__(
        self,
        robot_repository: RobotRepository,
        config_repository: ConfigRepository,
    ) -> None:
        """Initialize the robot service.

        Args:
            robot_repository: Robot repository
            config_repository: Configuration repository
        """
        self.robot_repository = robot_repository
        self.config_repository = config_repository

    def create_robot(self, robot_name: str) -> Robot:
        """Create a robot from configuration.

        Args:
            robot_name: Name of the robot

        Returns:
            Created robot instance

        Raises:
            ValueError: If robot type is not supported or configuration is invalid
        """
        # Get robot configuration
        config = self.config_repository.get_robot_config(robot_name)

        # Determine robot type
        try:
            robot_type = RobotType(robot_name)
        except ValueError:
            raise ValueError(f"Unsupported robot type: {robot_name}")

        # Validate configuration
        if not self.robot_repository.validate_config(robot_type, config):
            raise ValueError(f"Invalid configuration for robot {robot_name}")

        # Create robot
        return self.robot_repository.create_robot(robot_type, config)

    def get_supported_robots(self) -> List[str]:
        """Get list of supported robot names.

        Returns:
            List of supported robot names
        """
        robot_types = self.robot_repository.get_supported_robots()
        return [robot_type.value for robot_type in robot_types]

    def validate_command(self, robot: Robot, command: Command) -> bool:
        """Validate a command for a robot.

        Args:
            robot: Robot instance
            command: Command to validate

        Returns:
            True if command is valid
        """
        # Check if command velocities are within reasonable bounds
        max_linear_vel = robot.config.get("max_linear_velocity", 5.0)
        max_angular_vel = robot.config.get("max_angular_velocity", 3.0)

        linear_vel_magnitude = (command.linear_velocity**2).sum() ** 0.5
        angular_vel_magnitude = (command.angular_velocity**2).sum() ** 0.5

        return linear_vel_magnitude <= max_linear_vel and angular_vel_magnitude <= max_angular_vel


class TrainingService:
    """Service for training-related operations."""

    def __init__(self, config_repository: ConfigRepository) -> None:
        """Initialize the training service.

        Args:
            config_repository: Configuration repository
        """
        self.config_repository = config_repository

    def get_training_config(self, algorithm_name: str) -> Dict:
        """Get training configuration for an algorithm.

        Args:
            algorithm_name: Name of the training algorithm

        Returns:
            Training configuration
        """
        return self.config_repository.get_training_config(algorithm_name)

    def validate_training_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """Validate training configuration.

        Args:
            config: Training configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required_fields = [
            "algorithm.learning_rate",
            "algorithm.num_learning_iterations",
            "algorithm.num_steps_per_env",
        ]

        for field in required_fields:
            if not self._get_nested_value(config, field):
                errors.append(f"Missing required field: {field}")

        # Check value ranges
        learning_rate = self._get_nested_value(config, "algorithm.learning_rate")
        if learning_rate and (learning_rate <= 0 or learning_rate > 1.0):
            errors.append("Learning rate must be between 0 and 1")

        num_iterations = self._get_nested_value(config, "algorithm.num_learning_iterations")
        if num_iterations and num_iterations <= 0:
            errors.append("Number of learning iterations must be positive")

        return len(errors) == 0, errors

    def calculate_training_progress(
        self,
        current_iteration: int,
        total_iterations: int,
        recent_metrics: List[TrainingMetrics],
    ) -> Dict[str, float]:
        """Calculate training progress metrics.

        Args:
            current_iteration: Current training iteration
            total_iterations: Total training iterations
            recent_metrics: Recent training metrics

        Returns:
            Progress metrics dictionary
        """
        progress = {
            "iteration_progress": current_iteration / total_iterations,
            "remaining_iterations": total_iterations - current_iteration,
        }

        if recent_metrics:
            # Calculate moving averages
            recent_rewards = [m.episode_reward for m in recent_metrics[-100:]]
            recent_lengths = [m.episode_length for m in recent_metrics[-100:]]

            progress.update(
                {
                    "avg_episode_reward": sum(recent_rewards) / len(recent_rewards),
                    "avg_episode_length": sum(recent_lengths) / len(recent_lengths),
                    "reward_trend": self._calculate_trend(recent_rewards),
                }
            )

        return progress

    def _get_nested_value(self, config: Dict, key: str) -> Optional[any]:
        """Get nested value from configuration using dot notation.

        Args:
            config: Configuration dictionary
            key: Key with dot notation (e.g., "algorithm.learning_rate")

        Returns:
            Value if found, None otherwise
        """
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None

        return value

    def _calculate_trend(self, values: List[float], window: int = 20) -> float:
        """Calculate trend of values (positive = increasing, negative = decreasing).

        Args:
            values: List of values
            window: Window size for trend calculation

        Returns:
            Trend value
        """
        if len(values) < window:
            return 0.0

        recent = values[-window:]
        older = values[-window * 2 : -window] if len(values) >= window * 2 else values[:-window]

        if not older:
            return 0.0

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        return recent_avg - older_avg


class RewardService:
    """Service for reward calculation and analysis."""

    def __init__(self) -> None:
        """Initialize the reward service."""
        pass

    def analyze_reward_components(
        self,
        reward_history: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze reward components over time.

        Args:
            reward_history: History of reward components

        Returns:
            Analysis results including means, stds, trends
        """
        if not reward_history:
            return {}

        # Get all reward component names
        all_components = set()
        for rewards in reward_history:
            all_components.update(rewards.keys())

        analysis = {}

        for component in all_components:
            values = []
            for rewards in reward_history:
                values.append(rewards.get(component, 0.0))

            if values:
                analysis[component] = {
                    "mean": sum(values) / len(values),
                    "std": self._calculate_std(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._calculate_trend(values),
                }

        return analysis

    def identify_problematic_rewards(
        self,
        reward_analysis: Dict[str, Dict[str, float]],
        threshold: float = -0.1,
    ) -> List[str]:
        """Identify reward components that may be problematic.

        Args:
            reward_analysis: Result from analyze_reward_components
            threshold: Threshold for identifying problematic rewards

        Returns:
            List of problematic reward component names
        """
        problematic = []

        for component, stats in reward_analysis.items():
            # Check if reward is consistently negative or decreasing
            if stats["mean"] < threshold or stats["trend"] < -0.05:
                problematic.append(component)

        return problematic

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation.

        Args:
            values: List of values

        Returns:
            Standard deviation
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5
