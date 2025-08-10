"""Unified tracking system supporting both WandB and TensorBoard."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch

from luwu.domain.entities import TrackingBackend


class TrackingLogger(ABC):
    """Abstract base class for tracking loggers."""

    @abstractmethod
    def log_scalar(self, name: str, value: Union[float, int, torch.Tensor], step: int) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_dict(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: int) -> None:
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log histogram of values."""
        pass

    @abstractmethod
    def save_model(self, model: torch.nn.Module, name: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish logging session."""
        pass


class WandBLogger(TrackingLogger):
    """WandB tracking logger implementation."""

    def __init__(
        self, project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
        """
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config)
        except ImportError:
            raise ImportError("wandb package is required for WandB logging")

    def log_scalar(self, name: str, value: Union[float, int, torch.Tensor], step: int) -> None:
        """Log a scalar value to WandB."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._wandb.log({name: value}, step=step)

    def log_dict(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: int) -> None:
        """Log a dictionary of metrics to WandB."""
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value
        self._wandb.log(processed_metrics, step=step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log histogram to WandB."""
        self._wandb.log({name: self._wandb.Histogram(values.cpu().numpy())}, step=step)

    def save_model(self, model: torch.nn.Module, name: str) -> None:
        """Save model to WandB."""
        torch.save(model.state_dict(), f"{name}.pt")
        self._wandb.save(f"{name}.pt")

    def finish(self) -> None:
        """Finish WandB session."""
        self._wandb.finish()


class TensorBoardLogger(TrackingLogger):
    """TensorBoard tracking logger implementation."""

    def __init__(self, log_dir: str) -> None:
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            raise ImportError("tensorboard package is required for TensorBoard logging")

    def log_scalar(self, name: str, value: Union[float, int, torch.Tensor], step: int) -> None:
        """Log a scalar value to TensorBoard."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.writer.add_scalar(name, value, step)

    def log_dict(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: int) -> None:
        """Log a dictionary of metrics to TensorBoard."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(key, value, step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log histogram to TensorBoard."""
        self.writer.add_histogram(name, values, step)

    def save_model(self, model: torch.nn.Module, name: str) -> None:
        """Save model checkpoint."""
        torch.save(model.state_dict(), f"{name}.pt")

    def finish(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()


class TrackingManager:
    """Manager for unified tracking across different backends."""

    def __init__(self, backend: TrackingBackend, **kwargs: Any) -> None:
        """Initialize tracking manager.

        Args:
            backend: Tracking backend to use
            **kwargs: Backend-specific configuration
        """
        self.backend = backend

        if backend == TrackingBackend.WANDB:
            self.logger = WandBLogger(**kwargs)
        elif backend == TrackingBackend.TENSORBOARD:
            self.logger = TensorBoardLogger(**kwargs)
        else:
            raise ValueError(f"Unsupported tracking backend: {backend}")

    def log_dict(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: int) -> None:
        """Log dictionary of metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step
        """
        self.logger.log_dict(metrics, step)

    def log_training_metrics(
        self, rewards: Dict[str, torch.Tensor], losses: Dict[str, torch.Tensor], step: int
    ) -> None:
        """Log training metrics.

        Args:
            rewards: Dictionary of reward components
            losses: Dictionary of loss components
            step: Training step
        """
        # Log reward components
        for name, value in rewards.items():
            self.logger.log_scalar(f"rewards/{name}", value, step)

        # Log loss components
        for name, value in losses.items():
            self.logger.log_scalar(f"losses/{name}", value, step)

        # Log total reward
        total_reward = sum(rewards.values())
        self.logger.log_scalar("rewards/total", total_reward, step)

    def log_evaluation_metrics(
        self, metrics: Dict[str, Union[float, torch.Tensor]], step: int
    ) -> None:
        """Log evaluation metrics.

        Args:
            metrics: Dictionary of evaluation metrics
            step: Evaluation step
        """
        eval_metrics = {f"eval/{key}": value for key, value in metrics.items()}
        self.logger.log_dict(eval_metrics, step)

    def log_policy_stats(self, policy_stats: Dict[str, torch.Tensor], step: int) -> None:
        """Log policy statistics.

        Args:
            policy_stats: Dictionary of policy statistics
            step: Training step
        """
        for name, values in policy_stats.items():
            self.logger.log_histogram(f"policy/{name}", values, step)

    def save_checkpoint(self, model: torch.nn.Module, name: str) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save
            name: Checkpoint name
        """
        self.logger.save_model(model, name)

    def finish(self) -> None:
        """Finish tracking session."""
        self.logger.finish()
