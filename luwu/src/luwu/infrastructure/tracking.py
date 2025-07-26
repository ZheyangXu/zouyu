"""
Experiment tracking implementations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from luwu.domain.entities import Tracker


class WandbTracker(Tracker):
    """Weights & Biases tracker implementation."""

    def __init__(self, entity: Optional[str] = None) -> None:
        """Initialize the W&B tracker.

        Args:
            entity: W&B entity/team name
        """
        self.entity = entity
        self._run = None

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbTracker. " "Install it with: pip install wandb"
            )

    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize a new W&B run."""
        self._run = self.wandb.init(
            project=project_name,
            name=run_name,
            entity=self.entity,
            config=config,
            tags=tags,
            notes=notes,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self._run is None:
            raise RuntimeError("Run not initialized. Call init_run() first.")

        self.wandb.log(metrics, step=step)

    def log_video(self, video: np.ndarray, name: str, step: Optional[int] = None) -> None:
        """Log a video to W&B."""
        if self._run is None:
            raise RuntimeError("Run not initialized. Call init_run() first.")

        # Convert video to W&B format
        wandb_video = self.wandb.Video(video, fps=30, format="mp4")
        self.wandb.log({name: wandb_video}, step=step)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


class TensorboardTracker(Tracker):
    """TensorBoard tracker implementation."""

    def __init__(self, log_dir: str = "./logs/tensorboard") -> None:
        """Initialize the TensorBoard tracker.

        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self._writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError(
                "tensorboard is required for TensorboardTracker. "
                "Install it with: pip install tensorboard"
            )

    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize a new TensorBoard run."""
        run_dir = self.log_dir / project_name
        if run_name:
            run_dir = run_dir / run_name

        run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = self.SummaryWriter(log_dir=str(run_dir))

        # Log configuration as text
        if config:
            config_text = self._dict_to_markdown(config)
            self._writer.add_text("config", config_text)

        # Log notes and tags as text
        if notes:
            self._writer.add_text("notes", notes)
        if tags:
            self._writer.add_text("tags", ", ".join(tags))

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to TensorBoard."""
        if self._writer is None:
            raise RuntimeError("Run not initialized. Call init_run() first.")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)

    def log_video(self, video: np.ndarray, name: str, step: Optional[int] = None) -> None:
        """Log a video to TensorBoard."""
        if self._writer is None:
            raise RuntimeError("Run not initialized. Call init_run() first.")

        # Ensure video is in the right format for TensorBoard
        # Expected format: (N, T, C, H, W) where N=1, T=time, C=channels, H=height, W=width
        if video.ndim == 4:  # (T, H, W, C)
            video = video.transpose(0, 3, 1, 2)  # (T, C, H, W)
            video = video[np.newaxis, ...]  # (1, T, C, H, W)

        self._writer.add_video(name, video, step, fps=30)

    def finish(self) -> None:
        """Finish the TensorBoard run."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _dict_to_markdown(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to markdown format."""
        lines = []
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(self._dict_to_markdown(value, indent + 1))
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
        return "\n".join(lines)


class CompositeTracker(Tracker):
    """Composite tracker that uses multiple tracking backends."""

    def __init__(self, trackers: List[Tracker]) -> None:
        """Initialize the composite tracker.

        Args:
            trackers: List of tracker instances
        """
        self.trackers = trackers

    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize runs for all trackers."""
        for tracker in self.trackers:
            tracker.init_run(project_name, run_name, config, tags, notes)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to {type(tracker).__name__}: {e}")

    def log_video(self, video: np.ndarray, name: str, step: Optional[int] = None) -> None:
        """Log video to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_video(video, name, step)
            except Exception as e:
                print(f"Warning: Failed to log video to {type(tracker).__name__}: {e}")

    def finish(self) -> None:
        """Finish all tracker runs."""
        for tracker in self.trackers:
            try:
                tracker.finish()
            except Exception as e:
                print(f"Warning: Failed to finish {type(tracker).__name__}: {e}")


class TrackerFactory:
    """Factory for creating tracker instances."""

    @staticmethod
    def create_tracker(
        backend: str,
        entity: Optional[str] = None,
        log_dir: str = "./logs",
    ) -> Tracker:
        """Create a tracker instance.

        Args:
            backend: Tracking backend ("wandb", "tensorboard", or "both")
            entity: Entity/team name for W&B
            log_dir: Log directory for TensorBoard

        Returns:
            Tracker instance

        Raises:
            ValueError: If backend is not supported
        """
        if backend == "wandb":
            return WandbTracker(entity=entity)
        elif backend == "tensorboard":
            return TensorboardTracker(log_dir=log_dir)
        elif backend == "both":
            return CompositeTracker(
                [
                    WandbTracker(entity=entity),
                    TensorboardTracker(log_dir=log_dir),
                ]
            )
        else:
            raise ValueError(f"Unsupported tracking backend: {backend}")


# Convenience function for creating trackers
def create_tracker(
    backend: str,
    entity: Optional[str] = None,
    log_dir: str = "./logs",
) -> Tracker:
    """Create a tracker instance.

    Args:
        backend: Tracking backend ("wandb", "tensorboard", or "both")
        entity: Entity/team name for W&B
        log_dir: Log directory for TensorBoard

    Returns:
        Tracker instance
    """
    return TrackerFactory.create_tracker(backend, entity, log_dir)
