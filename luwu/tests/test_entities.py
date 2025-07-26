"""
Tests for domain entities.
"""

import numpy as np
import pytest

from luwu.domain.entities import (
    Action,
    Command,
    ControlMode,
    RobotState,
    RobotType,
    TrainingMetrics,
)


class TestRobotState:
    """Test cases for RobotState."""

    def test_robot_state_creation(self) -> None:
        """Test creating a robot state."""
        state = RobotState(
            base_position=np.array([0.0, 0.0, 0.35]),
            base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            base_linear_velocity=np.array([1.0, 0.0, 0.0]),
            base_angular_velocity=np.array([0.0, 0.0, 0.5]),
            joint_positions=np.array([0.1, 0.2, -1.5] * 4),
            joint_velocities=np.array([0.0] * 12),
            joint_torques=np.array([10.0] * 12),
            foot_contacts=np.array([True, True, True, False]),
            contact_forces=np.array([[0, 0, 100], [0, 0, 100], [0, 0, 100], [0, 0, 0]]),
        )

        assert state.num_joints == 12
        assert state.num_feet == 4
        assert np.allclose(state.base_position, [0.0, 0.0, 0.35])
        assert state.foot_contacts[3] == False

    def test_robot_state_with_sensor_data(self) -> None:
        """Test robot state with sensor data."""
        imu_data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        height_measurements = np.array([0.35, 0.36, 0.34, 0.37])

        state = RobotState(
            base_position=np.array([0.0, 0.0, 0.35]),
            base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            base_linear_velocity=np.array([0.0, 0.0, 0.0]),
            base_angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_positions=np.array([0.0] * 12),
            joint_velocities=np.array([0.0] * 12),
            joint_torques=np.array([0.0] * 12),
            foot_contacts=np.array([True] * 4),
            contact_forces=np.array([[0, 0, 50]] * 4),
            imu_data=imu_data,
            height_measurements=height_measurements,
        )

        assert state.imu_data is not None
        assert state.height_measurements is not None
        assert len(state.height_measurements) == 4


class TestCommand:
    """Test cases for Command."""

    def test_command_creation(self) -> None:
        """Test creating a command."""
        command = Command(
            linear_velocity=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.5]),
            body_height=0.35,
        )

        assert np.allclose(command.linear_velocity, [1.0, 0.0, 0.0])
        assert np.allclose(command.angular_velocity, [0.0, 0.0, 0.5])
        assert command.body_height == 0.35
        assert command.footstep_targets is None

    def test_command_with_footsteps(self) -> None:
        """Test command with footstep targets."""
        footstep_targets = np.array(
            [[0.3, 0.15, 0.0], [0.3, -0.15, 0.0], [0.0, 0.15, 0.0], [0.0, -0.15, 0.0]]
        )

        command = Command(
            linear_velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            footstep_targets=footstep_targets,
        )

        assert command.footstep_targets is not None
        assert command.footstep_targets.shape == (4, 3)


class TestAction:
    """Test cases for Action."""

    def test_action_creation(self) -> None:
        """Test creating an action."""
        action_values = np.array([0.1, -0.2, 0.5] * 4)
        action = Action(values=action_values, control_mode=ControlMode.POSITION)

        assert len(action.values) == 12
        assert action.control_mode == ControlMode.POSITION
        assert np.allclose(action.values[:3], [0.1, -0.2, 0.5])

    def test_action_clipping(self) -> None:
        """Test action clipping."""
        action_values = np.array([-2.0, 0.5, 1.5, 3.0])
        action = Action(values=action_values, control_mode=ControlMode.POSITION)

        clipped_action = action.clip(-1.0, 1.0)

        assert np.allclose(clipped_action.values, [-1.0, 0.5, 1.0, 1.0])
        assert clipped_action.control_mode == ControlMode.POSITION

        # Original action should be unchanged
        assert not np.allclose(action.values, clipped_action.values)


class TestTrainingMetrics:
    """Test cases for TrainingMetrics."""

    def test_training_metrics_creation(self) -> None:
        """Test creating training metrics."""
        reward_components = {
            "linear_velocity": 10.0,
            "angular_velocity": 5.0,
            "torques": -2.0,
            "orientation": -1.0,
        }

        metrics = TrainingMetrics(
            episode_reward=150.0,
            episode_length=800,
            policy_loss=0.001,
            value_loss=0.5,
            entropy_loss=0.01,
            explained_variance=0.8,
            clipfrac=0.1,
            approx_kl=0.02,
            learning_rate=3e-4,
            reward_components=reward_components,
        )

        assert metrics.episode_reward == 150.0
        assert metrics.episode_length == 800
        assert len(metrics.reward_components) == 4
        assert metrics.reward_components["linear_velocity"] == 10.0

    def test_training_metrics_to_dict(self) -> None:
        """Test converting training metrics to dictionary."""
        reward_components = {
            "tracking": 20.0,
            "regularization": -5.0,
        }

        metrics = TrainingMetrics(
            episode_reward=100.0,
            episode_length=500,
            policy_loss=0.002,
            value_loss=0.3,
            entropy_loss=0.02,
            explained_variance=0.7,
            clipfrac=0.15,
            approx_kl=0.03,
            learning_rate=1e-4,
            reward_components=reward_components,
        )

        metrics_dict = metrics.to_dict()

        # Check base metrics
        assert metrics_dict["episode_reward"] == 100.0
        assert metrics_dict["policy_loss"] == 0.002

        # Check reward components with prefix
        assert metrics_dict["reward/tracking"] == 20.0
        assert metrics_dict["reward/regularization"] == -5.0


class TestEnums:
    """Test cases for enum classes."""

    def test_robot_type_enum(self) -> None:
        """Test RobotType enum."""
        assert RobotType.A1.value == "a1"
        assert RobotType.GO1.value == "go1"
        assert RobotType.ANYMAL_B.value == "anymal_b"
        assert RobotType.ANYMAL_C.value == "anymal_c"

        # Test creation from string
        assert RobotType("a1") == RobotType.A1
        assert RobotType("go1") == RobotType.GO1

    def test_control_mode_enum(self) -> None:
        """Test ControlMode enum."""
        assert ControlMode.POSITION.value == "position"
        assert ControlMode.VELOCITY.value == "velocity"
        assert ControlMode.TORQUE.value == "torque"

        # Test creation from string
        assert ControlMode("position") == ControlMode.POSITION
        assert ControlMode("velocity") == ControlMode.VELOCITY


# Fixtures for testing
@pytest.fixture
def sample_robot_state() -> RobotState:
    """Sample robot state for testing."""
    return RobotState(
        base_position=np.array([0.0, 0.0, 0.35]),
        base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        base_linear_velocity=np.array([1.0, 0.0, 0.0]),
        base_angular_velocity=np.array([0.0, 0.0, 0.2]),
        joint_positions=np.array([0.1, 0.8, -1.5] * 4),
        joint_velocities=np.array([0.0] * 12),
        joint_torques=np.array([5.0] * 12),
        foot_contacts=np.array([True, True, True, True]),
        contact_forces=np.array([[0, 0, 25]] * 4),
    )


@pytest.fixture
def sample_command() -> Command:
    """Sample command for testing."""
    return Command(
        linear_velocity=np.array([1.5, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.3]),
        body_height=0.35,
    )


@pytest.fixture
def sample_action() -> Action:
    """Sample action for testing."""
    return Action(
        values=np.array([0.1, 0.2, -0.3] * 4),
        control_mode=ControlMode.POSITION,
    )
