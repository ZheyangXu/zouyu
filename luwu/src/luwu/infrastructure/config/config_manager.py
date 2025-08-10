"""Configuration management for LuWu parkour training system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dynaconf import Dynaconf


class ConfigManager:
    """Centralized configuration manager using Dynaconf."""

    def __init__(self, config_dir: str = "configs") -> None:
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.settings = Dynaconf(
            envvar_prefix="LUWU",
            settings_files=[
                f"{config_dir}/settings.yaml",
                f"{config_dir}/settings.toml",
                f"{config_dir}/settings.json",
                f"{config_dir}/.secrets.yaml",
                f"{config_dir}/.secrets.toml",
                f"{config_dir}/.secrets.json",
            ],
            environments=True,
            load_dotenv=True,
            env_switcher="LUWU_ENV",
        )

    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            filepath: Path to YAML file

        Returns:
            Configuration dictionary
        """
        if not filepath.exists():
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.settings.get(key, default)

    def get_robot_config(self, robot_name: str) -> Dict[str, Any]:
        """Get robot-specific configuration.

        Args:
            robot_name: Name of the robot

        Returns:
            Robot configuration dictionary
        """
        robot_file = self.config_dir / "robots" / f"{robot_name}.yaml"
        return self._load_yaml_file(robot_file)

    def get_env_config(self, env_name: str) -> Dict[str, Any]:
        """Get environment-specific configuration.

        Args:
            env_name: Name of the environment

        Returns:
            Environment configuration dictionary
        """
        env_file = self.config_dir / "environments" / f"{env_name}.yaml"
        return self._load_yaml_file(env_file)

    def get_training_config(self, training_name: str) -> Dict[str, Any]:
        """Get training-specific configuration.

        Args:
            training_name: Name of the training configuration

        Returns:
            Training configuration dictionary
        """
        training_file = self.config_dir / "training" / f"{training_name}.yaml"
        return self._load_yaml_file(training_file)

    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration.

        Returns:
            Simulation configuration dictionary
        """
        return self.get("simulation", {})

    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration.

        Returns:
            Tracking configuration dictionary
        """
        return self.get("tracking", {})


# Global configuration manager instance
config_manager = ConfigManager()
