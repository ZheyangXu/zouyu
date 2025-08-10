"""Tests for domain entities."""

import pytest
import torch

from luwu.domain.entities import (
    Action,
    EnvironmentConfig,
    RewardComponent,
    RobotConfig,
    RobotState,
    SimulationEngine,
    TrackingBackend,
    TrainingConfig,
)


class TestRobotState:
    """Test cases for RobotState."""

    def test_robot_state_creation(self):
        """Test RobotState creation with valid tensors."""
        state = RobotState(
            position=torch.tensor([0.0, 0.0, 1.0]),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=torch.tensor([1.0, 0.0, 0.0]),
            angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
            joint_positions=torch.zeros(12),
            joint_velocities=torch.zeros(12),
            joint_torques=torch.zeros(12),
            contact_forces=torch.zeros(4, 3),
        )

        assert state.position.shape == (3,)
        assert state.orientation.shape == (4,)
        assert state.linear_velocity.shape == (3,)
        assert state.angular_velocity.shape == (3,)
        assert state.joint_positions.shape == (12,)
        assert state.joint_velocities.shape == (12,)
        assert state.joint_torques.shape == (12,)
        assert state.contact_forces.shape == (4, 3)

    def test_robot_state_with_height_measurements(self):
        """Test RobotState with optional height measurements."""
        state = RobotState(
            position=torch.tensor([0.0, 0.0, 1.0]),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=torch.tensor([1.0, 0.0, 0.0]),
            angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
            joint_positions=torch.zeros(12),
            joint_velocities=torch.zeros(12),
            joint_torques=torch.zeros(12),
            contact_forces=torch.zeros(4, 3),
            height_measurements=torch.zeros(100),
        )

        assert state.height_measurements is not None
        assert state.height_measurements.shape == (100,)


class TestAction:
    """Test cases for Action."""

    def test_action_creation(self):
        """Test Action creation with default type."""
        action = Action(joint_targets=torch.zeros(12))

        assert action.joint_targets.shape == (12,)
        assert action.action_type == "position"

    def test_action_with_custom_type(self):
        """Test Action creation with custom type."""
        action = Action(joint_targets=torch.zeros(12), action_type="torque")

        assert action.action_type == "torque"


class TestRewardComponent:
    """Test cases for RewardComponent."""

    def test_reward_component_creation(self):
        """Test RewardComponent creation with valid parameters."""
        component = RewardComponent(name="test_reward", weight=1.5, scale=2.0, enabled=True)

        assert component.name == "test_reward"
        assert component.weight == 1.5
        assert component.scale == 2.0
        assert component.enabled is True

    def test_reward_component_defaults(self):
        """Test RewardComponent with default values."""
        component = RewardComponent(name="test_reward", weight=1.0)

        assert component.scale == 1.0
        assert component.enabled is True

    def test_reward_component_invalid_weight(self):
        """Test RewardComponent with invalid weight."""
        with pytest.raises(ValueError):
            RewardComponent(name="test_reward", weight=0.0)

        with pytest.raises(ValueError):
            RewardComponent(name="test_reward", weight=-1.0)


class TestRobotConfig:
    """Test cases for RobotConfig."""

    def test_robot_config_creation(self):
        """Test RobotConfig creation with valid parameters."""
        config = RobotConfig(
            name="test_robot",
            urdf_path="path/to/robot.urdf",
            num_joints=12,
            joint_names=["joint_1", "joint_2"],
            default_joint_positions=[0.0, 0.9],
            joint_limits={"joint_1": (-1.0, 1.0), "joint_2": (-2.0, 2.0)},
            mass=10.0,
            base_dimensions=[0.5, 0.3, 0.2],
            motor_strength=30.0,
        )

        assert config.name == "test_robot"
        assert config.num_joints == 12
        assert len(config.joint_names) == 2
        assert len(config.default_joint_positions) == 2
        assert config.mass == 10.0
        assert config.motor_strength == 30.0


class TestEnvironmentConfig:
    """Test cases for EnvironmentConfig."""

    def test_environment_config_creation(self):
        """Test EnvironmentConfig creation with valid parameters."""
        reward_components = [
            RewardComponent(name="survival", weight=1.0),
            RewardComponent(name="velocity", weight=2.0),
        ]

        config = EnvironmentConfig(
            name="test_env",
            terrain_type="flat",
            terrain_size=(10.0, 10.0),
            num_envs=1024,
            env_spacing=2.0,
            episode_length=1000,
            reward_components=reward_components,
            observation_space_dim=48,
            action_space_dim=12,
        )

        assert config.name == "test_env"
        assert config.terrain_type == "flat"
        assert config.terrain_size == (10.0, 10.0)
        assert config.num_envs == 1024
        assert config.env_spacing == 2.0
        assert config.episode_length == 1000
        assert len(config.reward_components) == 2
        assert config.observation_space_dim == 48
        assert config.action_space_dim == 12

    def test_environment_config_defaults(self):
        """Test EnvironmentConfig with default values."""
        config = EnvironmentConfig(
            name="test_env",
            terrain_type="flat",
            terrain_size=(10.0, 10.0),
            num_envs=1024,
            env_spacing=2.0,
            episode_length=1000,
            reward_components=[],
            observation_space_dim=48,
            action_space_dim=12,
        )

        assert config.gravity == [0.0, 0.0, -9.81]
        assert config.dt == 0.02
        assert config.substeps == 4

    def test_environment_config_invalid_values(self):
        """Test EnvironmentConfig with invalid values."""
        with pytest.raises(ValueError):
            EnvironmentConfig(
                name="test_env",
                terrain_type="flat",
                terrain_size=(10.0, 10.0),
                num_envs=0,  # Invalid: must be > 0
                env_spacing=2.0,
                episode_length=1000,
                reward_components=[],
                observation_space_dim=48,
                action_space_dim=12,
            )


class TestTrainingConfig:
    """Test cases for TrainingConfig."""

    def test_training_config_creation(self):
        """Test TrainingConfig creation with valid parameters."""
        config = TrainingConfig(
            algorithm="PPO",
            num_iterations=5000,
            num_steps_per_env=24,
            mini_batch_size=1024,
            num_epochs=5,
            learning_rate=0.0003,
        )

        assert config.algorithm == "PPO"
        assert config.num_iterations == 5000
        assert config.num_steps_per_env == 24
        assert config.mini_batch_size == 1024
        assert config.num_epochs == 5
        assert config.learning_rate == 0.0003

    def test_training_config_defaults(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig(
            algorithm="PPO",
            num_iterations=5000,
            num_steps_per_env=24,
            mini_batch_size=1024,
            num_epochs=5,
            learning_rate=0.0003,
        )

        assert config.gamma == 0.99
        assert config.lam == 0.95
        assert config.clip_coef == 0.2
        assert config.entropy_coef == 0.01
        assert config.value_loss_coef == 0.5
        assert config.max_grad_norm == 1.0
        assert config.save_interval == 100
        assert config.checkpoint_dir == "checkpoints"

    def test_training_config_invalid_values(self):
        """Test TrainingConfig with invalid values."""
        with pytest.raises(ValueError):
            TrainingConfig(
                algorithm="PPO",
                num_iterations=0,  # Invalid: must be > 0
                num_steps_per_env=24,
                mini_batch_size=1024,
                num_epochs=5,
                learning_rate=0.0003,
            )

        with pytest.raises(ValueError):
            TrainingConfig(
                algorithm="PPO",
                num_iterations=5000,
                num_steps_per_env=24,
                mini_batch_size=1024,
                num_epochs=5,
                learning_rate=0.0,  # Invalid: must be > 0
            )


class TestEnums:
    """Test cases for enumeration types."""

    def test_simulation_engine_values(self):
        """Test SimulationEngine enum values."""
        assert SimulationEngine.ISAAC_SIM.value == "isaac_sim"
        assert SimulationEngine.ISAAC_LAB.value == "isaac_lab"
        assert SimulationEngine.MUJOCO.value == "mujoco"

    def test_tracking_backend_values(self):
        """Test TrackingBackend enum values."""
        assert TrackingBackend.WANDB.value == "wandb"
        assert TrackingBackend.TENSORBOARD.value == "tensorboard"
