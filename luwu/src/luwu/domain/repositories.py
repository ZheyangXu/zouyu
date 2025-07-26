"""
Repository interfaces for the domain layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from luwu.domain.entities import Robot, RobotType


class RobotRepository(ABC):
    """Repository interface for robots."""

    @abstractmethod
    def create_robot(self, robot_type: RobotType, config: Dict) -> Robot:
        """Create a robot instance.

        Args:
            robot_type: Type of robot to create
            config: Robot configuration

        Returns:
            Created robot instance
        """
        pass

    @abstractmethod
    def get_supported_robots(self) -> List[RobotType]:
        """Get list of supported robot types.

        Returns:
            List of supported robot types
        """
        pass

    @abstractmethod
    def validate_config(self, robot_type: RobotType, config: Dict) -> bool:
        """Validate robot configuration.

        Args:
            robot_type: Type of robot
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        pass


class ConfigRepository(ABC):
    """Repository interface for configurations."""

    @abstractmethod
    def get_robot_config(self, robot_name: str) -> Dict:
        """Get robot configuration.

        Args:
            robot_name: Name of the robot

        Returns:
            Robot configuration
        """
        pass

    @abstractmethod
    def get_environment_config(self, env_name: str) -> Dict:
        """Get environment configuration.

        Args:
            env_name: Name of the environment

        Returns:
            Environment configuration
        """
        pass

    @abstractmethod
    def get_training_config(self, algorithm_name: str) -> Dict:
        """Get training configuration.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Training configuration
        """
        pass

    @abstractmethod
    def save_config(self, name: str, config: Dict, config_type: str) -> None:
        """Save configuration to file.

        Args:
            name: Configuration name
            config: Configuration data
            config_type: Type of configuration (robot, environment, training)
        """
        pass


class ModelRepository(ABC):
    """Repository interface for trained models."""

    @abstractmethod
    def save_model(self, model_path: str, metadata: Dict) -> None:
        """Save a trained model.

        Args:
            model_path: Path to the model file
            metadata: Model metadata
        """
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> Optional[str]:
        """Load a trained model.

        Args:
            model_id: Model identifier

        Returns:
            Path to the loaded model file, or None if not found
        """
        pass

    @abstractmethod
    def list_models(self, robot_type: Optional[str] = None) -> List[Dict]:
        """List available models.

        Args:
            robot_type: Filter by robot type

        Returns:
            List of model metadata
        """
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a model.

        Args:
            model_id: Model identifier

        Returns:
            True if model was deleted successfully
        """
        pass
