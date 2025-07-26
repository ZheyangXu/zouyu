"""
Core configuration management using dynaconf.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from dynaconf import Dynaconf
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Simulation configuration."""

    backend: str = Field(default="isaac_sim", description="Simulation backend")
    device: str = Field(default="cuda:0", description="Device to run simulation on")
    headless: bool = Field(default=False, description="Run in headless mode")
    num_envs: int = Field(default=4096, description="Number of environments")
    physics_dt: float = Field(default=0.005, description="Physics timestep")
    control_dt: float = Field(default=0.02, description="Control timestep")


class TrainingConfig(BaseModel):
    """Training configuration."""

    algorithm: str = Field(default="ppo", description="Training algorithm")
    max_iterations: int = Field(default=5000, description="Maximum training iterations")
    save_interval: int = Field(default=100, description="Save interval")
    eval_interval: int = Field(default=100, description="Evaluation interval")
    checkpoint_dir: str = Field(default="./logs/checkpoints", description="Checkpoint directory")


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    enabled: bool = Field(default=True, description="Enable tracking")
    backend: str = Field(
        default="wandb", description="Tracking backend: wandb, tensorboard, or both"
    )
    project_name: str = Field(default="luwu-parkour", description="Project name")
    entity: Optional[str] = Field(default=None, description="Entity/team name")
    tags: list[str] = Field(default_factory=list, description="Tags for the run")
    notes: str = Field(default="", description="Notes for the run")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="./logs", description="Log directory")
    console_output: bool = Field(default=True, description="Enable console output")
    file_output: bool = Field(default=True, description="Enable file output")


class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    name: str = Field(default="parkour", description="Environment name")
    robot: str = Field(default="a1", description="Robot type")
    terrain_type: str = Field(default="rough", description="Terrain type")
    curriculum_learning: bool = Field(default=True, description="Enable curriculum learning")


class Config(BaseModel):
    """Main configuration class."""

    project_name: str = Field(default="luwu", description="Project name")
    version: str = Field(default="0.1.0", description="Project version")
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class ConfigManager:
    """Configuration manager using dynaconf."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs"

        self.config_dir = Path(config_dir)
        self._dynaconf = Dynaconf(
            envvar_prefix="LUWU",
            settings_files=[
                str(self.config_dir / "settings.yaml"),
                str(self.config_dir / "settings.toml"),
                str(self.config_dir / "settings.json"),
            ],
            environments=True,
            load_dotenv=True,
        )

    def get_config(self) -> Config:
        """Get the main configuration."""
        config_dict = dict(self._dynaconf)
        return Config(**config_dict)

    def get_robot_config(self, robot_name: str) -> Dict[str, Any]:
        """Get robot-specific configuration.

        Args:
            robot_name: Name of the robot

        Returns:
            Robot configuration dictionary
        """
        robot_config_path = self.config_dir / "robots" / f"{robot_name}.yaml"
        if not robot_config_path.exists():
            raise ValueError(f"Robot configuration not found: {robot_config_path}")

        robot_dynaconf = Dynaconf(settings_files=[str(robot_config_path)])
        return dict(robot_dynaconf)

    def get_environment_config(self, env_name: str) -> Dict[str, Any]:
        """Get environment-specific configuration.

        Args:
            env_name: Name of the environment

        Returns:
            Environment configuration dictionary
        """
        env_config_path = self.config_dir / "environments" / f"{env_name}.yaml"
        if not env_config_path.exists():
            raise ValueError(f"Environment configuration not found: {env_config_path}")

        env_dynaconf = Dynaconf(settings_files=[str(env_config_path)])
        return dict(env_dynaconf)

    def get_training_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Get training algorithm configuration.

        Args:
            algorithm_name: Name of the training algorithm

        Returns:
            Training configuration dictionary
        """
        training_config_path = self.config_dir / "training" / f"{algorithm_name}.yaml"
        if not training_config_path.exists():
            raise ValueError(f"Training configuration not found: {training_config_path}")

        training_dynaconf = Dynaconf(settings_files=[str(training_config_path)])
        return dict(training_dynaconf)

    def set_environment(self, env: str) -> None:
        """Set the environment (development, production, etc.).

        Args:
            env: Environment name
        """
        self._dynaconf.setenv(env)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._dynaconf.get(key, default)


# Global configuration manager instance
config_manager = ConfigManager()
