"""
Tests for configuration management.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import yaml

from luwu.infrastructure.config import ConfigManager
from luwu.infrastructure.repositories import FileConfigRepository


class TestConfigManager:
    """Test cases for ConfigManager."""

    def test_config_manager_initialization(self) -> None:
        """Test config manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create a basic settings file
            settings_file = config_dir / "settings.yaml"
            settings_data = {
                "project_name": "test_project",
                "simulation": {"num_envs": 1024},
            }
            with open(settings_file, "w") as f:
                yaml.safe_dump(settings_data, f)

            # Initialize config manager
            config_manager = ConfigManager(config_dir)
            config = config_manager.get_config()

            assert config.project_name == "test_project"
            assert config.simulation.num_envs == 1024

    def test_get_robot_config(self) -> None:
        """Test getting robot configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            robots_dir = config_dir / "robots"
            robots_dir.mkdir()

            # Create robot config
            robot_config = {
                "robot": {"name": "test_robot"},
                "physics": {"mass": 10.0},
            }
            robot_file = robots_dir / "test_robot.yaml"
            with open(robot_file, "w") as f:
                yaml.safe_dump(robot_config, f)

            config_manager = ConfigManager(config_dir)
            retrieved_config = config_manager.get_robot_config("test_robot")

            assert retrieved_config["robot"]["name"] == "test_robot"
            assert retrieved_config["physics"]["mass"] == 10.0

    def test_get_nonexistent_config(self) -> None:
        """Test getting non-existent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_manager = ConfigManager(config_dir)

            with pytest.raises(ValueError, match="Robot configuration not found"):
                config_manager.get_robot_config("nonexistent_robot")


class TestFileConfigRepository:
    """Test cases for FileConfigRepository."""

    def test_load_yaml_config(self) -> None:
        """Test loading YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            repo = FileConfigRepository(config_dir)

            # Create test config
            test_config = {"test_key": "test_value", "nested": {"key": 123}}
            config_file = config_dir / "robots" / "test.yaml"
            config_file.parent.mkdir(parents=True)
            with open(config_file, "w") as f:
                yaml.safe_dump(test_config, f)

            # Load config
            loaded_config = repo.get_robot_config("test")
            assert loaded_config == test_config

    def test_load_json_config(self) -> None:
        """Test loading JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            repo = FileConfigRepository(config_dir)

            # Create test config
            test_config = {"test_key": "test_value", "nested": {"key": 123}}
            config_file = config_dir / "environments" / "test.json"
            config_file.parent.mkdir(parents=True)
            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Load config
            loaded_config = repo.get_environment_config("test")
            assert loaded_config == test_config

    def test_save_config(self) -> None:
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            repo = FileConfigRepository(config_dir)

            # Save config
            test_config = {"key": "value", "number": 42}
            repo.save_config("test_robot", test_config, "robot")

            # Verify file was created
            config_file = config_dir / "robots" / "test_robot.yaml"
            assert config_file.exists()

            # Verify content
            with open(config_file, "r") as f:
                loaded_config = yaml.safe_load(f)
            assert loaded_config == test_config

    def test_unsupported_config_type(self) -> None:
        """Test saving with unsupported config type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            repo = FileConfigRepository(config_dir)

            with pytest.raises(ValueError, match="Unsupported config type"):
                repo.save_config("test", {}, "unsupported_type")


@pytest.fixture
def sample_robot_config() -> Dict:
    """Sample robot configuration for testing."""
    return {
        "robot": {
            "name": "test_robot",
            "urdf_path": "test.urdf",
        },
        "control": {
            "type": "position",
            "stiffness": {"hip": 20.0, "thigh": 20.0, "calf": 20.0},
        },
        "rewards": {
            "linear_velocity_xy": 1.0,
            "torques": -1e-5,
        },
    }


@pytest.fixture
def sample_environment_config() -> Dict:
    """Sample environment configuration for testing."""
    return {
        "environment": {
            "name": "test_env",
        },
        "terrain": {
            "type": "procedural",
            "size": [100.0, 100.0],
        },
        "episode": {
            "max_episode_length": 1000,
        },
    }


@pytest.fixture
def sample_training_config() -> Dict:
    """Sample training configuration for testing."""
    return {
        "algorithm": {
            "name": "ppo",
            "learning_rate": 3e-4,
            "num_learning_iterations": 5000,
            "num_steps_per_env": 24,
        },
        "checkpoints": {
            "save_interval": 100,
        },
    }
