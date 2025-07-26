"""
Application services for coordinating domain services and infrastructure.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luwu.domain.entities import TrainingMetrics, Tracker
from luwu.domain.services import RobotService, TrainingService as DomainTrainingService
from luwu.infrastructure.config import ConfigManager
from luwu.infrastructure.repositories import CachedConfigRepository, FileConfigRepository
from luwu.infrastructure.tracking import create_tracker


class ApplicationService:
    """Main application service for coordinating operations."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize the application service.

        Args:
            config_dir: Path to configuration directory
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_dir)
        self.config = self.config_manager.get_config()

        # Initialize repositories
        file_repo = FileConfigRepository(self.config_manager.config_dir)
        self.config_repository = CachedConfigRepository(file_repo)

        # Initialize domain services
        # Note: RobotRepository implementation would be created here
        # For now, we'll skip robot service initialization
        self.training_service = DomainTrainingService(self.config_repository)

        # Initialize tracking
        self.tracker: Optional[Tracker] = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.logging

        # Create log directory
        log_dir = Path(log_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        handlers = []

        if log_config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_config.level))
            handlers.append(console_handler)

        if log_config.file_output:
            file_handler = logging.FileHandler(log_dir / "luwu.log")
            file_handler.setLevel(getattr(logging, log_config.level))
            handlers.append(file_handler)

        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

        self.logger = logging.getLogger(__name__)

    def initialize_tracking(self) -> None:
        """Initialize experiment tracking."""
        if not self.config.tracking.enabled:
            self.logger.info("Tracking disabled")
            return

        tracking_config = self.config.tracking

        try:
            self.tracker = create_tracker(
                backend=tracking_config.backend,
                entity=tracking_config.entity,
                log_dir=self.config.logging.log_dir,
            )
            self.logger.info(f"Initialized tracking with backend: {tracking_config.backend}")
        except Exception as e:
            self.logger.error(f"Failed to initialize tracking: {e}")
            self.tracker = None

    def start_training_run(
        self,
        experiment_name: Optional[str] = None,
        robot_name: Optional[str] = None,
        environment_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new training run.

        Args:
            experiment_name: Name of the experiment
            robot_name: Name of the robot (overrides config)
            environment_name: Name of the environment (overrides config)
            algorithm_name: Name of the algorithm (overrides config)

        Returns:
            Combined configuration for the training run
        """
        # Use provided names or fall back to config
        robot_name = robot_name or self.config.environment.robot
        environment_name = environment_name or self.config.environment.name
        algorithm_name = algorithm_name or self.config.training.algorithm

        self.logger.info(f"Starting training run: {experiment_name}")
        self.logger.info(
            f"Robot: {robot_name}, Environment: {environment_name}, Algorithm: {algorithm_name}"
        )

        # Get configurations
        try:
            robot_config = self.config_repository.get_robot_config(robot_name)
            env_config = self.config_repository.get_environment_config(environment_name)
            training_config = self.config_repository.get_training_config(algorithm_name)
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise

        # Validate training configuration
        is_valid, errors = self.training_service.validate_training_config(training_config)
        if not is_valid:
            error_msg = f"Invalid training configuration: {', '.join(errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Combine configurations
        combined_config = {
            "main": dict(self.config),
            "robot": robot_config,
            "environment": env_config,
            "training": training_config,
            "metadata": {
                "experiment_name": experiment_name,
                "robot_name": robot_name,
                "environment_name": environment_name,
                "algorithm_name": algorithm_name,
            },
        }

        # Initialize tracking run
        if self.tracker:
            try:
                self.tracker.init_run(
                    project_name=self.config.tracking.project_name,
                    run_name=experiment_name,
                    config=combined_config,
                    tags=self.config.tracking.tags,
                    notes=self.config.tracking.notes,
                )
                self.logger.info("Tracking run initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize tracking run: {e}")

        return combined_config

    def log_training_metrics(
        self,
        metrics: TrainingMetrics,
        iteration: int,
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Training metrics to log
            iteration: Current training iteration
        """
        if self.tracker:
            try:
                metrics_dict = metrics.to_dict()
                self.tracker.log_metrics(metrics_dict, step=iteration)
            except Exception as e:
                self.logger.error(f"Failed to log metrics: {e}")

        # Log to console/file
        self.logger.info(
            f"Iteration {iteration}: "
            f"reward={metrics.episode_reward:.3f}, "
            f"length={metrics.episode_length}, "
            f"policy_loss={metrics.policy_loss:.6f}"
        )

    def log_training_video(
        self,
        video: Any,
        name: str,
        iteration: int,
    ) -> None:
        """Log training video.

        Args:
            video: Video array
            name: Name of the video
            iteration: Current training iteration
        """
        if self.tracker:
            try:
                self.tracker.log_video(video, name, step=iteration)
                self.logger.info(f"Logged video: {name} at iteration {iteration}")
            except Exception as e:
                self.logger.error(f"Failed to log video: {e}")

    def finish_training_run(self) -> None:
        """Finish the current training run."""
        if self.tracker:
            try:
                self.tracker.finish()
                self.logger.info("Training run finished")
            except Exception as e:
                self.logger.error(f"Failed to finish tracking run: {e}")

    def get_training_progress(
        self,
        current_iteration: int,
        total_iterations: int,
        recent_metrics: List[TrainingMetrics],
    ) -> Dict[str, Any]:
        """Get training progress information.

        Args:
            current_iteration: Current training iteration
            total_iterations: Total training iterations
            recent_metrics: Recent training metrics

        Returns:
            Progress information dictionary
        """
        return self.training_service.calculate_training_progress(
            current_iteration, total_iterations, recent_metrics
        )

    def save_checkpoint(
        self,
        model_path: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save training checkpoint.

        Args:
            model_path: Path to the model file
            metadata: Checkpoint metadata
        """
        # Create checkpoint directory
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata alongside model
        metadata_path = Path(model_path).with_suffix(".json")

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Checkpoint saved: {model_path}")

    def list_available_configs(self) -> Dict[str, List[str]]:
        """List available configurations.

        Returns:
            Dictionary with lists of available configurations
        """
        config_dir = self.config_manager.config_dir

        def list_config_files(directory: Path) -> List[str]:
            if not directory.exists():
                return []

            files = []
            for file_path in directory.glob("*"):
                if file_path.suffix in [".yaml", ".yml", ".json", ".toml"]:
                    files.append(file_path.stem)
            return sorted(files)

        return {
            "robots": list_config_files(config_dir / "robots"),
            "environments": list_config_files(config_dir / "environments"),
            "training": list_config_files(config_dir / "training"),
        }

    def validate_configuration(
        self,
        robot_name: str,
        environment_name: str,
        algorithm_name: str,
    ) -> Tuple[bool, List[str]]:
        """Validate a complete configuration setup.

        Args:
            robot_name: Name of the robot
            environment_name: Name of the environment
            algorithm_name: Name of the algorithm

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if configurations exist
        try:
            self.config_repository.get_robot_config(robot_name)
        except FileNotFoundError:
            errors.append(f"Robot configuration not found: {robot_name}")

        try:
            self.config_repository.get_environment_config(environment_name)
        except FileNotFoundError:
            errors.append(f"Environment configuration not found: {environment_name}")

        try:
            training_config = self.config_repository.get_training_config(algorithm_name)

            # Validate training configuration
            is_valid, training_errors = self.training_service.validate_training_config(
                training_config
            )
            if not is_valid:
                errors.extend(training_errors)

        except FileNotFoundError:
            errors.append(f"Training configuration not found: {algorithm_name}")

        return len(errors) == 0, errors


class TrainingService:
    """Service for training robot policies (CLI interface)."""

    def __init__(self):
        """Initialize the training service."""
        self.config_manager = ConfigManager()

    def train(
        self,
        robot_config: Dict[str, Any],
        env_config: Dict[str, Any],
        training_config: Dict[str, Any],
        checkpoint_dir: Optional[Path] = None,
        num_envs: int = 4096,
        max_iterations: int = 5000,
        headless: bool = True,
    ) -> TrainingMetrics:
        """Train a robot policy.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
            training_config: Training algorithm configuration
            checkpoint_dir: Directory to save checkpoints
            num_envs: Number of parallel environments
            max_iterations: Maximum training iterations
            headless: Run in headless mode

        Returns:
            Training metrics
        """
        # Placeholder implementation
        print(f"Training with {num_envs} environments for {max_iterations} iterations")

        # This would integrate with Isaac Sim/Lab in the actual implementation
        metrics = TrainingMetrics(
            episode=max_iterations,
            total_steps=max_iterations * num_envs,
            mean_reward=100.0,
            episode_length=1000,
            success_rate=0.95,
            policy_loss=0.001,
            value_loss=0.002,
            entropy=0.1,
        )

        return metrics


class EvaluationService:
    """Service for evaluating trained robot policies."""

    def __init__(self):
        """Initialize the evaluation service."""
        self.config_manager = ConfigManager()

    def evaluate(
        self,
        robot_config: Dict[str, Any],
        env_config: Dict[str, Any],
        checkpoint_path: Path,
        num_episodes: int = 100,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate a trained robot policy.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
            checkpoint_path: Path to trained model checkpoint
            num_episodes: Number of evaluation episodes
            output_dir: Directory to save results

        Returns:
            Evaluation metrics
        """
        # Placeholder implementation
        print(f"Evaluating policy from {checkpoint_path} for {num_episodes} episodes")

        # This would integrate with Isaac Sim/Lab in the actual implementation
        metrics = {
            "mean_reward": 95.5,
            "std_reward": 5.2,
            "success_rate": 0.92,
            "mean_episode_length": 950.0,
            "completion_time": 45.3,
        }

        return metrics
