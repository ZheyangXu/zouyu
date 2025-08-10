#!/usr/bin/env python3
"""Example script demonstrating LuWu parkour training system."""

import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from luwu.application.environments.base_env import VectorizedLeggedParkourEnv
from luwu.application.training.trainer import ParkourTrainer
from luwu.domain.entities import EnvironmentConfig, RobotConfig, TrainingConfig
from luwu.infrastructure.config import config_manager
from luwu.infrastructure.simulation.mujoco_backend import MujocoSimulation


def main():
    """Run a simple training example."""
    print("LuWu Parkour Training Example")
    print("=" * 40)

    # Set environment variable for configuration
    os.environ["LUWU_ENV"] = "development"

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load configurations (these would normally come from YAML files)
        robot_config = RobotConfig(
            name="example_robot",
            urdf_path="assets/robots/a1/a1.xml",  # MuJoCo XML format
            num_joints=12,
            joint_names=[
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ],
            default_joint_positions=[
                0.0,
                0.9,
                -1.8,  # FL
                0.0,
                0.9,
                -1.8,  # FR
                0.0,
                0.9,
                -1.8,  # RL
                0.0,
                0.9,
                -1.8,  # RR
            ],
            joint_limits={f"joint_{i}": (-2.0, 2.0) for i in range(12)},
            mass=12.0,
            base_dimensions=[0.366, 0.094, 0.114],
            motor_strength=33.5,
        )

        env_config = EnvironmentConfig(
            name="example_env",
            terrain_type="flat",
            terrain_size=(8.0, 8.0),
            num_envs=16,  # Small number for example
            env_spacing=2.0,
            episode_length=200,  # Short episodes for quick demo
            reward_components=[],  # Will use default rewards
            observation_space_dim=48,
            action_space_dim=12,
            dt=0.02,
            substeps=4,
        )

        training_config = TrainingConfig(
            algorithm="PPO",
            num_iterations=50,  # Short training for demo
            num_steps_per_env=24,
            mini_batch_size=384,  # Smaller batch for demo
            num_epochs=3,
            learning_rate=0.001,
            gamma=0.99,
            lam=0.95,
            clip_coef=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=1.0,
            save_interval=25,
            checkpoint_dir="example_checkpoints",
        )

        print("Configurations loaded successfully")
        print(f"Robot: {robot_config.name}")
        print(f"Environment: {env_config.name}")
        print(f"Training: {training_config.algorithm}")

        # Create simulation backend
        print("\\nInitializing MuJoCo simulation...")
        simulation_backend = MujocoSimulation(robot_config, env_config.dict())

        # Create vectorized environment
        print("Creating vectorized environment...")
        env = VectorizedLeggedParkourEnv(
            robot_config=robot_config,
            env_config=env_config,
            simulation_backend=simulation_backend,
            num_envs=env_config.num_envs,
        )

        # Create trainer
        print("Creating trainer...")
        trainer = ParkourTrainer(
            robot_config=robot_config,
            env_config=env_config,
            training_config=training_config,
            env=env,
            device=device,
        )

        print("\\nStarting training...")
        print("Note: This is a minimal example. For full training, use the CLI commands.")

        # Run training
        trainer.train()

        print("\\nTraining completed!")
        print("Check the 'example_checkpoints' directory for saved models.")

        # Clean up
        env.close()

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Make sure all required packages are installed:")
        print("  pdm install")

    except Exception as e:
        print(f"Error during training: {e}")
        print("This is expected if MuJoCo is not properly installed.")
        print("For actual training, make sure to:")
        print("1. Install MuJoCo")
        print("2. Provide valid robot URDF/XML files")
        print("3. Use the CLI commands instead of this example")


if __name__ == "__main__":
    main()
