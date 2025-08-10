"""Tests for configuration management."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from luwu.infrastructure.config import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager."""

    def test_init_with_default_dir(self):
        """Test ConfigManager initialization with default directory."""
        config_manager = ConfigManager()
        assert config_manager.settings is not None

    def test_init_with_custom_dir(self):
        """Test ConfigManager initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(config_dir=temp_dir)
            assert config_manager.settings is not None

    def test_get_default_value(self):
        """Test getting default value when key doesn't exist."""
        config_manager = ConfigManager()
        result = config_manager.get("nonexistent.key", "default_value")
        assert result == "default_value"

    def test_get_none_default(self):
        """Test getting None as default value."""
        config_manager = ConfigManager()
        result = config_manager.get("nonexistent.key")
        assert result is None

    @patch("dynaconf.Dynaconf")
    def test_get_existing_value(self, mock_dynaconf):
        """Test getting existing configuration value."""
        # Mock dynaconf settings
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = "test_value"

        config_manager = ConfigManager()
        result = config_manager.get("test.key", "default")

        assert result == "test_value"
        mock_settings.get.assert_called_once_with("test.key", "default")

    @patch("dynaconf.Dynaconf")
    def test_get_robot_config(self, mock_dynaconf):
        """Test getting robot-specific configuration."""
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = {"name": "test_robot", "num_joints": 12}

        config_manager = ConfigManager()
        result = config_manager.get_robot_config("test_robot")

        assert result == {"name": "test_robot", "num_joints": 12}
        mock_settings.get.assert_called_once_with("robots.test_robot", {})

    @patch("dynaconf.Dynaconf")
    def test_get_env_config(self, mock_dynaconf):
        """Test getting environment-specific configuration."""
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = {"name": "test_env", "num_envs": 1024}

        config_manager = ConfigManager()
        result = config_manager.get_env_config("test_env")

        assert result == {"name": "test_env", "num_envs": 1024}
        mock_settings.get.assert_called_once_with("environments.test_env", {})

    @patch("dynaconf.Dynaconf")
    def test_get_training_config(self, mock_dynaconf):
        """Test getting training-specific configuration."""
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = {"algorithm": "PPO", "learning_rate": 0.001}

        config_manager = ConfigManager()
        result = config_manager.get_training_config("test_training")

        assert result == {"algorithm": "PPO", "learning_rate": 0.001}
        mock_settings.get.assert_called_once_with("training.test_training", {})

    @patch("dynaconf.Dynaconf")
    def test_get_simulation_config(self, mock_dynaconf):
        """Test getting simulation configuration."""
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = {"engine": "mujoco", "dt": 0.02}

        config_manager = ConfigManager()
        result = config_manager.get_simulation_config()

        assert result == {"engine": "mujoco", "dt": 0.02}
        mock_settings.get.assert_called_once_with("simulation", {})

    @patch("dynaconf.Dynaconf")
    def test_get_tracking_config(self, mock_dynaconf):
        """Test getting tracking configuration."""
        mock_settings = mock_dynaconf.return_value
        mock_settings.get.return_value = {"backend": "tensorboard", "log_dir": "logs"}

        config_manager = ConfigManager()
        result = config_manager.get_tracking_config()

        assert result == {"backend": "tensorboard", "log_dir": "logs"}
        mock_settings.get.assert_called_once_with("tracking", {})

    def test_empty_robot_config(self):
        """Test getting empty robot configuration."""
        config_manager = ConfigManager()
        result = config_manager.get_robot_config("nonexistent_robot")
        assert result == {}

    def test_empty_env_config(self):
        """Test getting empty environment configuration."""
        config_manager = ConfigManager()
        result = config_manager.get_env_config("nonexistent_env")
        assert result == {}

    def test_empty_training_config(self):
        """Test getting empty training configuration."""
        config_manager = ConfigManager()
        result = config_manager.get_training_config("nonexistent_training")
        assert result == {}

    def test_empty_simulation_config(self):
        """Test getting empty simulation configuration."""
        config_manager = ConfigManager()
        result = config_manager.get_simulation_config()
        # Should return empty dict when no simulation config is found
        assert isinstance(result, dict)

    def test_empty_tracking_config(self):
        """Test getting empty tracking configuration."""
        config_manager = ConfigManager()
        result = config_manager.get_tracking_config()
        # Should return empty dict when no tracking config is found
        assert isinstance(result, dict)
