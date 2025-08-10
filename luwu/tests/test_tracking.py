"""Tests for tracking system."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from luwu.domain.entities import TrackingBackend
from luwu.infrastructure.tracking import (
    TensorBoardLogger,
    TrackingManager,
    WandBLogger,
)


class TestWandBLogger:
    """Test cases for WandB logger."""

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_init(self, mock_wandb):
        """Test WandB logger initialization."""
        mock_wandb.init.return_value = MagicMock()

        logger = WandBLogger(project="test_project", name="test_run")

        mock_wandb.init.assert_called_once_with(
            project="test_project", name="test_run", config=None
        )
        assert logger._wandb == mock_wandb

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_log_scalar(self, mock_wandb):
        """Test WandB scalar logging."""
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()

        logger = WandBLogger(project="test_project")
        logger.log_scalar("test_metric", 1.5, step=10)

        mock_wandb.log.assert_called_once_with({"test_metric": 1.5}, step=10)

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_log_tensor(self, mock_wandb):
        """Test WandB tensor logging."""
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()

        logger = WandBLogger(project="test_project")
        tensor_value = torch.tensor(2.5)
        logger.log_scalar("test_tensor", tensor_value, step=10)

        mock_wandb.log.assert_called_once_with({"test_tensor": 2.5}, step=10)

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_log_dict(self, mock_wandb):
        """Test WandB dictionary logging."""
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()

        logger = WandBLogger(project="test_project")
        metrics = {
            "metric1": 1.0,
            "metric2": torch.tensor(2.0),
            "metric3": 3,
        }
        logger.log_dict(metrics, step=10)

        expected_metrics = {"metric1": 1.0, "metric2": 2.0, "metric3": 3}
        mock_wandb.log.assert_called_once_with(expected_metrics, step=10)

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_log_histogram(self, mock_wandb):
        """Test WandB histogram logging."""
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.Histogram = MagicMock()

        logger = WandBLogger(project="test_project")
        values = torch.randn(100)
        logger.log_histogram("test_hist", values, step=10)

        mock_wandb.Histogram.assert_called_once()
        mock_wandb.log.assert_called_once()

    @patch("luwu.infrastructure.tracking.tracking_manager.wandb")
    def test_wandb_logger_finish(self, mock_wandb):
        """Test WandB logger finish."""
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.finish = MagicMock()

        logger = WandBLogger(project="test_project")
        logger.finish()

        mock_wandb.finish.assert_called_once()


class TestTensorBoardLogger:
    """Test cases for TensorBoard logger."""

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_init(self, mock_summary_writer):
        """Test TensorBoard logger initialization."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")

        mock_summary_writer.assert_called_once_with(log_dir="test_logs")
        assert logger.writer == mock_writer

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_log_scalar(self, mock_summary_writer):
        """Test TensorBoard scalar logging."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")
        logger.log_scalar("test_metric", 1.5, step=10)

        mock_writer.add_scalar.assert_called_once_with("test_metric", 1.5, 10)

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_log_tensor(self, mock_summary_writer):
        """Test TensorBoard tensor logging."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")
        tensor_value = torch.tensor(2.5)
        logger.log_scalar("test_tensor", tensor_value, step=10)

        mock_writer.add_scalar.assert_called_once_with("test_tensor", 2.5, 10)

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_log_dict(self, mock_summary_writer):
        """Test TensorBoard dictionary logging."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")
        metrics = {
            "metric1": 1.0,
            "metric2": torch.tensor(2.0),
            "metric3": 3,
        }
        logger.log_dict(metrics, step=10)

        # Should call add_scalar for each metric
        assert mock_writer.add_scalar.call_count == 3

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_log_histogram(self, mock_summary_writer):
        """Test TensorBoard histogram logging."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")
        values = torch.randn(100)
        logger.log_histogram("test_hist", values, step=10)

        mock_writer.add_histogram.assert_called_once_with("test_hist", values, 10)

    @patch("luwu.infrastructure.tracking.tracking_manager.SummaryWriter")
    def test_tensorboard_logger_finish(self, mock_summary_writer):
        """Test TensorBoard logger finish."""
        mock_writer = MagicMock()
        mock_summary_writer.return_value = mock_writer

        logger = TensorBoardLogger(log_dir="test_logs")
        logger.finish()

        mock_writer.close.assert_called_once()


class TestTrackingManager:
    """Test cases for TrackingManager."""

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_wandb_init(self, mock_wandb_logger):
        """Test TrackingManager initialization with WandB."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        mock_wandb_logger.assert_called_once_with(project="test_project")
        assert manager.logger == mock_logger
        assert manager.backend == TrackingBackend.WANDB

    @patch("luwu.infrastructure.tracking.tracking_manager.TensorBoardLogger")
    def test_tracking_manager_tensorboard_init(self, mock_tb_logger):
        """Test TrackingManager initialization with TensorBoard."""
        mock_logger = MagicMock()
        mock_tb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.TENSORBOARD, log_dir="test_logs")

        mock_tb_logger.assert_called_once_with(log_dir="test_logs")
        assert manager.logger == mock_logger
        assert manager.backend == TrackingBackend.TENSORBOARD

    def test_tracking_manager_invalid_backend(self):
        """Test TrackingManager with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported tracking backend"):
            TrackingManager(backend="invalid_backend")

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_log_training_metrics(self, mock_wandb_logger):
        """Test TrackingManager training metrics logging."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        rewards = {
            "survival": torch.tensor(1.0),
            "velocity": torch.tensor(2.0),
        }
        losses = {
            "policy_loss": torch.tensor(0.1),
            "value_loss": torch.tensor(0.2),
        }

        manager.log_training_metrics(rewards, losses, step=10)

        # Should log individual rewards, losses, and total reward
        assert mock_logger.log_scalar.call_count >= 4
        mock_logger.log_scalar.assert_any_call("rewards/survival", torch.tensor(1.0), 10)
        mock_logger.log_scalar.assert_any_call("rewards/velocity", torch.tensor(2.0), 10)
        mock_logger.log_scalar.assert_any_call("losses/policy_loss", torch.tensor(0.1), 10)
        mock_logger.log_scalar.assert_any_call("losses/value_loss", torch.tensor(0.2), 10)

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_log_evaluation_metrics(self, mock_wandb_logger):
        """Test TrackingManager evaluation metrics logging."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        metrics = {
            "mean_reward": 10.5,
            "success_rate": 0.8,
        }

        manager.log_evaluation_metrics(metrics, step=100)

        expected_metrics = {
            "eval/mean_reward": 10.5,
            "eval/success_rate": 0.8,
        }
        mock_logger.log_dict.assert_called_once_with(expected_metrics, 100)

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_log_policy_stats(self, mock_wandb_logger):
        """Test TrackingManager policy statistics logging."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        policy_stats = {
            "action_std": torch.randn(100),
            "value_estimates": torch.randn(100),
        }

        manager.log_policy_stats(policy_stats, step=50)

        assert mock_logger.log_histogram.call_count == 2
        mock_logger.log_histogram.assert_any_call(
            "policy/action_std", policy_stats["action_std"], 50
        )
        mock_logger.log_histogram.assert_any_call(
            "policy/value_estimates", policy_stats["value_estimates"], 50
        )

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_save_checkpoint(self, mock_wandb_logger):
        """Test TrackingManager checkpoint saving."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        mock_model = MagicMock()
        manager.save_checkpoint(mock_model, "test_checkpoint")

        mock_logger.save_model.assert_called_once_with(mock_model, "test_checkpoint")

    @patch("luwu.infrastructure.tracking.tracking_manager.WandBLogger")
    def test_tracking_manager_finish(self, mock_wandb_logger):
        """Test TrackingManager finish."""
        mock_logger = MagicMock()
        mock_wandb_logger.return_value = mock_logger

        manager = TrackingManager(backend=TrackingBackend.WANDB, project="test_project")

        manager.finish()

        mock_logger.finish.assert_called_once()
