"""
Tests for tracking functionality.
"""

import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from luwu.infrastructure.tracking import (
    CompositeTracker,
    TensorboardTracker,
    TrackerFactory,
    WandbTracker,
)


class TestWandbTracker:
    """Test cases for WandbTracker."""

    @patch("luwu.infrastructure.tracking.wandb")
    def test_wandb_tracker_init(self, mock_wandb: MagicMock) -> None:
        """Test W&B tracker initialization."""
        tracker = WandbTracker(entity="test_entity")

        # Initialize run
        tracker.init_run(
            project_name="test_project",
            run_name="test_run",
            config={"test": "config"},
            tags=["test"],
            notes="test notes",
        )

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            name="test_run",
            entity="test_entity",
            config={"test": "config"},
            tags=["test"],
            notes="test notes",
        )

    @patch("luwu.infrastructure.tracking.wandb")
    def test_wandb_tracker_log_metrics(self, mock_wandb: MagicMock) -> None:
        """Test logging metrics to W&B."""
        tracker = WandbTracker()
        tracker._run = MagicMock()  # Mock the run object

        metrics = {"loss": 0.5, "reward": 100.0}
        tracker.log_metrics(metrics, step=10)

        mock_wandb.log.assert_called_once_with(metrics, step=10)

    @patch("luwu.infrastructure.tracking.wandb")
    def test_wandb_tracker_log_video(self, mock_wandb: MagicMock) -> None:
        """Test logging video to W&B."""
        tracker = WandbTracker()
        tracker._run = MagicMock()

        video = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        tracker.log_video(video, "test_video", step=5)

        mock_wandb.Video.assert_called_once()
        mock_wandb.log.assert_called_once()


class TestTensorboardTracker:
    """Test cases for TensorboardTracker."""

    @patch("luwu.infrastructure.tracking.SummaryWriter")
    def test_tensorboard_tracker_init(self, mock_summary_writer: MagicMock) -> None:
        """Test TensorBoard tracker initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = TensorboardTracker(log_dir=temp_dir)
            tracker.SummaryWriter = mock_summary_writer

            # Initialize run
            tracker.init_run(
                project_name="test_project",
                run_name="test_run",
                config={"test": "config"},
                notes="test notes",
            )

            expected_log_dir = Path(temp_dir) / "test_project" / "test_run"
            mock_summary_writer.assert_called_once_with(log_dir=str(expected_log_dir))

    def test_tensorboard_tracker_log_metrics(self) -> None:
        """Test logging metrics to TensorBoard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = TensorboardTracker(log_dir=temp_dir)

            # Mock the writer
            mock_writer = MagicMock()
            tracker._writer = mock_writer

            metrics = {"loss": 0.5, "reward": 100.0, "text_metric": "ignored"}
            tracker.log_metrics(metrics, step=10)

            # Should only log numeric metrics
            mock_writer.add_scalar.assert_any_call("loss", 0.5, 10)
            mock_writer.add_scalar.assert_any_call("reward", 100.0, 10)
            assert mock_writer.add_scalar.call_count == 2


class TestCompositeTracker:
    """Test cases for CompositeTracker."""

    def test_composite_tracker_delegates_to_all(self) -> None:
        """Test that composite tracker delegates to all sub-trackers."""
        # Create mock trackers
        tracker1 = MagicMock()
        tracker2 = MagicMock()

        composite = CompositeTracker([tracker1, tracker2])

        # Test init_run
        composite.init_run("project", "run", {"config": "test"})
        tracker1.init_run.assert_called_once_with("project", "run", {"config": "test"}, None, None)
        tracker2.init_run.assert_called_once_with("project", "run", {"config": "test"}, None, None)

        # Test log_metrics
        metrics = {"test": 1.0}
        composite.log_metrics(metrics, step=5)
        tracker1.log_metrics.assert_called_once_with(metrics, 5)
        tracker2.log_metrics.assert_called_once_with(metrics, 5)

        # Test finish
        composite.finish()
        tracker1.finish.assert_called_once()
        tracker2.finish.assert_called_once()

    def test_composite_tracker_handles_exceptions(self) -> None:
        """Test that composite tracker handles exceptions gracefully."""
        # Create mock trackers, one that raises an exception
        tracker1 = MagicMock()
        tracker2 = MagicMock()
        tracker1.log_metrics.side_effect = Exception("Test error")

        composite = CompositeTracker([tracker1, tracker2])

        # Should not raise exception, but still call tracker2
        metrics = {"test": 1.0}
        composite.log_metrics(metrics, step=5)

        tracker1.log_metrics.assert_called_once_with(metrics, 5)
        tracker2.log_metrics.assert_called_once_with(metrics, 5)


class TestTrackerFactory:
    """Test cases for TrackerFactory."""

    def test_create_wandb_tracker(self) -> None:
        """Test creating W&B tracker."""
        tracker = TrackerFactory.create_tracker("wandb", entity="test")
        assert isinstance(tracker, WandbTracker)
        assert tracker.entity == "test"

    def test_create_tensorboard_tracker(self) -> None:
        """Test creating TensorBoard tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = TrackerFactory.create_tracker("tensorboard", log_dir=temp_dir)
            assert isinstance(tracker, TensorboardTracker)
            assert str(tracker.log_dir) == temp_dir

    def test_create_composite_tracker(self) -> None:
        """Test creating composite tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = TrackerFactory.create_tracker("both", log_dir=temp_dir)
            assert isinstance(tracker, CompositeTracker)
            assert len(tracker.trackers) == 2
            assert isinstance(tracker.trackers[0], WandbTracker)
            assert isinstance(tracker.trackers[1], TensorboardTracker)

    def test_create_invalid_tracker(self) -> None:
        """Test creating tracker with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported tracking backend"):
            TrackerFactory.create_tracker("invalid_backend")


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Sample metrics for testing."""
    return {
        "episode_reward": 150.0,
        "policy_loss": 0.001,
        "value_loss": 0.5,
        "entropy": 0.8,
    }


@pytest.fixture
def sample_video() -> np.ndarray:
    """Sample video for testing."""
    return np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
