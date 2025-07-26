"""
Repository implementations for configuration management.
"""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from luwu.domain.repositories import ConfigRepository


class FileConfigRepository(ConfigRepository):
    """File-based configuration repository."""

    def __init__(self, config_dir: Path) -> None:
        """Initialize the file configuration repository.

        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = Path(config_dir)
        self.robots_dir = self.config_dir / "robots"
        self.environments_dir = self.config_dir / "environments"
        self.training_dir = self.config_dir / "training"

        # Ensure directories exist
        for dir_path in [self.robots_dir, self.environments_dir, self.training_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_robot_config(self, robot_name: str) -> Dict[str, Any]:
        """Get robot configuration from file.

        Args:
            robot_name: Name of the robot

        Returns:
            Robot configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        config_path = self._find_config_file(self.robots_dir, robot_name)
        if not config_path:
            raise FileNotFoundError(f"Robot configuration not found: {robot_name}")

        return self._load_config_file(config_path)

    def get_environment_config(self, env_name: str) -> Dict[str, Any]:
        """Get environment configuration from file.

        Args:
            env_name: Name of the environment

        Returns:
            Environment configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        config_path = self._find_config_file(self.environments_dir, env_name)
        if not config_path:
            raise FileNotFoundError(f"Environment configuration not found: {env_name}")

        return self._load_config_file(config_path)

    def get_training_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Get training configuration from file.

        Args:
            algorithm_name: Name of the training algorithm

        Returns:
            Training configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        config_path = self._find_config_file(self.training_dir, algorithm_name)
        if not config_path:
            raise FileNotFoundError(f"Training configuration not found: {algorithm_name}")

        return self._load_config_file(config_path)

    def save_config(self, name: str, config: Dict[str, Any], config_type: str) -> None:
        """Save configuration to file.

        Args:
            name: Configuration name
            config: Configuration data
            config_type: Type of configuration (robot, environment, training)

        Raises:
            ValueError: If config_type is not supported
        """
        if config_type == "robot":
            config_dir = self.robots_dir
        elif config_type == "environment":
            config_dir = self.environments_dir
        elif config_type == "training":
            config_dir = self.training_dir
        else:
            raise ValueError(f"Unsupported config type: {config_type}")

        config_path = config_dir / f"{name}.yaml"
        self._save_config_file(config_path, config)

    def _find_config_file(self, directory: Path, name: str) -> Path | None:
        """Find configuration file by name.

        Args:
            directory: Directory to search in
            name: Configuration name

        Returns:
            Path to configuration file, or None if not found
        """
        # Try different extensions
        for ext in [".yaml", ".yml", ".json", ".toml"]:
            config_path = directory / f"{name}{ext}"
            if config_path.exists():
                return config_path

        return None

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file format is not supported or file is invalid
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix == ".json":
                    return json.load(f)
                elif config_path.suffix == ".toml":
                    try:
                        import tomllib

                        content = f.read()
                        return tomllib.loads(content)
                    except ImportError:
                        try:
                            import toml

                            f.seek(0)
                            return toml.load(f)
                        except ImportError:
                            raise ImportError(
                                "TOML support requires tomllib (Python 3.11+) or toml package"
                            )
                else:
                    raise ValueError(f"Unsupported file format: {config_path.suffix}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def _save_config_file(self, config_path: Path, config: Dict[str, Any]) -> None:
        """Save configuration to file.

        Args:
            config_path: Path to save configuration
            config: Configuration data

        Raises:
            ValueError: If file format is not supported
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    yaml.safe_dump(config, f, default_flow_style=False, indent=2)
                elif config_path.suffix == ".json":
                    json.dump(config, f, indent=2)
                elif config_path.suffix == ".toml":
                    try:
                        import toml

                        toml.dump(config, f)
                    except ImportError:
                        raise ImportError("TOML support requires toml package")
                else:
                    raise ValueError(f"Unsupported file format: {config_path.suffix}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_path}: {e}")


class CachedConfigRepository(ConfigRepository):
    """Cached configuration repository for better performance."""

    def __init__(self, base_repository: ConfigRepository) -> None:
        """Initialize the cached configuration repository.

        Args:
            base_repository: Base repository to cache
        """
        self.base_repository = base_repository
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_robot_config(self, robot_name: str) -> Dict[str, Any]:
        """Get robot configuration with caching."""
        cache_key = f"robot:{robot_name}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.base_repository.get_robot_config(robot_name)
        return self._cache[cache_key].copy()  # Return a copy to prevent modification

    def get_environment_config(self, env_name: str) -> Dict[str, Any]:
        """Get environment configuration with caching."""
        cache_key = f"environment:{env_name}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.base_repository.get_environment_config(env_name)
        return self._cache[cache_key].copy()

    def get_training_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Get training configuration with caching."""
        cache_key = f"training:{algorithm_name}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.base_repository.get_training_config(algorithm_name)
        return self._cache[cache_key].copy()

    def save_config(self, name: str, config: Dict[str, Any], config_type: str) -> None:
        """Save configuration and invalidate cache."""
        self.base_repository.save_config(name, config, config_type)

        # Invalidate cache
        cache_key = f"{config_type}:{name}"
        if cache_key in self._cache:
            del self._cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
