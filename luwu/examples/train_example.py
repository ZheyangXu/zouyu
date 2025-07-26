"""
Example training script demonstrating the luwu framework.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

from luwu.application.services import ApplicationService
from luwu.infrastructure.isaac_sim import IsaacSimEnvironment
from luwu.infrastructure.ppo import PPOPolicy


def create_environment(config: Dict[str, Any]) -> IsaacSimEnvironment:
    """Create the training environment.

    Args:
        config: Combined configuration

    Returns:
        Created environment
    """
    # Combine environment and robot config
    env_config = {
        **config["environment"],
        "robot": config["robot"],
        "simulation": config["main"]["simulation"],
    }

    return IsaacSimEnvironment(env_config)


def create_policy(env: IsaacSimEnvironment, config: Dict[str, Any]) -> PPOPolicy:
    """Create the training policy.

    Args:
        env: Training environment
        config: Combined configuration

    Returns:
        Created policy
    """
    # Get observation and action dimensions
    # This would normally be determined from the environment
    obs_dim = 235  # Example: base(6) + joints(24) + height_scan(187) + actions(12) + history(5)
    action_dim = 12  # Number of actuated joints

    device = config["main"]["simulation"]["device"]

    return PPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config["training"],
        device=device,
    )


def collect_rollout(
    env: IsaacSimEnvironment,
    policy: PPOPolicy,
    num_steps: int,
) -> Dict[str, torch.Tensor]:
    """Collect a rollout of experience.

    Args:
        env: Environment to collect from
        policy: Policy to use
        num_steps: Number of steps to collect

    Returns:
        Dictionary containing rollout data
    """
    # Storage for rollout data
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []

    # Get initial observations
    obs = env.reset()

    for step in range(num_steps):
        # Get actions and values from policy
        with torch.no_grad():
            action, value = policy.predict(obs)

        # Step environment
        next_obs, reward, done, infos = env.step(action)

        # Store data
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        # Update observations
        obs = next_obs

    return {
        "observations": torch.stack(observations),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "dones": torch.stack(dones),
        "values": torch.stack(values),
    }


def train_iteration(
    env: IsaacSimEnvironment,
    policy: PPOPolicy,
    app_service: ApplicationService,
    config: Dict[str, Any],
    iteration: int,
) -> None:
    """Run one training iteration.

    Args:
        env: Training environment
        policy: Policy to train
        app_service: Application service for logging
        config: Combined configuration
        iteration: Current iteration number
    """
    # Get training parameters
    training_config = config["training"]["algorithm"]
    num_steps = training_config.get("num_steps_per_env", 24)

    # Collect rollout
    rollout_data = collect_rollout(env, policy, num_steps)

    # Update policy
    metrics = policy.update(
        observations=rollout_data["observations"],
        actions=rollout_data["actions"],
        rewards=rollout_data["rewards"],
        dones=rollout_data["dones"],
        values=rollout_data["values"],
    )

    # Log metrics
    app_service.log_training_metrics(metrics, iteration)

    # Save checkpoint if needed
    save_interval = config["training"]["checkpoints"].get("save_interval", 100)
    if iteration % save_interval == 0:
        checkpoint_dir = Path(config["main"]["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / f"model_iteration_{iteration}.pth"
        policy.save(str(model_path))

        # Save metadata
        metadata = {
            "iteration": iteration,
            "config": config,
            "metrics": metrics.to_dict(),
        }
        app_service.save_checkpoint(str(model_path), metadata)


def main() -> None:
    """Main training function."""
    # Initialize application service
    app_service = ApplicationService()

    # Start training run
    config = app_service.start_training_run(
        experiment_name="example_parkour_training",
        robot_name="a1",
        environment_name="parkour",
        algorithm_name="ppo",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    try:
        # Create environment and policy
        env = create_environment(config)
        policy = create_policy(env, config)

        logger.info(f"Environment created with {env.num_envs} parallel environments")
        logger.info(
            f"Policy created with {sum(p.numel() for p in policy.network.parameters())} parameters"
        )

        # Training loop
        max_iterations = config["training"]["algorithm"]["num_learning_iterations"]

        for iteration in range(1, max_iterations + 1):
            train_iteration(env, policy, app_service, config, iteration)

            if iteration % 50 == 0:
                logger.info(f"Completed iteration {iteration}/{max_iterations}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Finish tracking
        app_service.finish_training_run()


if __name__ == "__main__":
    main()
