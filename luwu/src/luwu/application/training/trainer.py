"""Training runner for legged robot parkour."""

import os
import time
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from luwu.application.algorithms.ppo import ActorCritic, PPO
from luwu.application.algorithms.storage import RewardTracker, RolloutBuffer
from luwu.application.environments.base_env import VectorizedLeggedParkourEnv
from luwu.domain.entities import EnvironmentConfig, RobotConfig, TrainingConfig, TrackingBackend
from luwu.infrastructure.config import config_manager
from luwu.infrastructure.tracking import TrackingManager


class ParkourTrainer:
    """Main trainer for legged robot parkour."""

    def __init__(
        self,
        robot_config: RobotConfig,
        env_config: EnvironmentConfig,
        training_config: TrainingConfig,
        env: VectorizedLeggedParkourEnv,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize parkour trainer.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
            training_config: Training configuration
            env: Vectorized environment
            device: Training device
        """
        self.robot_config = robot_config
        self.env_config = env_config
        self.training_config = training_config
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor-critic network
        self.actor_critic = ActorCritic(
            obs_dim=env_config.observation_space_dim,
            action_dim=env_config.action_space_dim,
            hidden_dims=(256, 256, 256),
            activation="elu",
            init_noise_std=1.0,
        )

        # Initialize PPO algorithm
        self.ppo = PPO(
            actor_critic=self.actor_critic,
            config=training_config,
            device=self.device,
        )

        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_steps=training_config.num_steps_per_env,
            num_envs=env_config.num_envs,
            obs_dim=env_config.observation_space_dim,
            action_dim=env_config.action_space_dim,
            device=self.device,
            gamma=training_config.gamma,
            gae_lambda=training_config.lam,
        )

        # Initialize reward tracker
        self.reward_tracker = RewardTracker(
            num_envs=env_config.num_envs,
            device=self.device,
        )

        # Initialize tracking
        tracking_config = config_manager.get_tracking_config()
        backend_str = tracking_config.get("backend", "tensorboard")
        backend = (
            TrackingBackend.TENSORBOARD if backend_str == "tensorboard" else TrackingBackend.WANDB
        )

        # Provide default log_dir for TensorBoard if not specified
        config_kwargs = tracking_config.get("config", {})
        if backend == TrackingBackend.TENSORBOARD and "log_dir" not in config_kwargs:
            config_kwargs["log_dir"] = "logs/tensorboard"

        self.tracking_manager = TrackingManager(
            backend=backend,
            **config_kwargs,
        )

        # Training state
        self.current_iteration = 0
        self.total_timesteps = 0
        self.best_reward = float("-inf")

        # Create checkpoint directory
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        """Run the training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Robot: {self.robot_config.name}")
        print(f"Environment: {self.env_config.name}")
        print(f"Training iterations: {self.training_config.num_iterations}")

        # Reset environment
        observations = self.env.reset()

        # Training loop
        start_time = time.time()

        for iteration in tqdm(range(self.training_config.num_iterations), desc="Training"):
            self.current_iteration = iteration

            # Collect rollout
            self._collect_rollout(observations)

            # Get final observations for bootstrap
            final_observations = self.env.get_observations()
            final_values = self.ppo.evaluate(final_observations)

            # Compute returns and advantages
            self.rollout_buffer.compute_returns_and_advantages(final_values)

            # Update policy
            training_stats = self._update_policy()

            # Log metrics
            self._log_metrics(training_stats, iteration)

            # Save checkpoint
            if (iteration + 1) % self.training_config.save_interval == 0:
                self._save_checkpoint(iteration + 1)

            # Reset buffer
            self.rollout_buffer.clear()

            # Get new observations for next rollout
            observations = self.env.get_observations()

        # Final save
        self._save_checkpoint(self.training_config.num_iterations)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        # Finish tracking
        self.tracking_manager.finish()

    def _collect_rollout(self, initial_observations: torch.Tensor) -> None:
        """Collect a rollout of experience.

        Args:
            initial_observations: Initial observations
        """
        observations = initial_observations

        for step in range(self.training_config.num_steps_per_env):
            # Generate actions
            actions, log_probs = self.ppo.act(observations)
            values = self.ppo.evaluate(observations)

            # Step environment
            next_observations, rewards, dones, info = self.env.step(actions)

            # Store experience
            self.rollout_buffer.add(
                observations=observations,
                actions=actions,
                log_probs=log_probs.squeeze(-1),
                rewards=rewards,
                values=values,
                dones=dones,
            )

            # Update observations
            observations = next_observations

            # Update total timesteps
            self.total_timesteps += self.env_config.num_envs

    def _update_policy(self) -> Dict[str, float]:
        """Update policy using collected rollout.

        Returns:
            Dictionary of training statistics
        """
        # Get batch data
        batch = self.rollout_buffer.get_batch()

        # Update PPO
        training_stats = self.ppo.update(
            observations=batch["observations"],
            actions=batch["actions"],
            old_log_probs=batch["log_probs"],
            returns=batch["returns"],
            advantages=batch["advantages"],
        )

        return training_stats

    def _log_metrics(self, training_stats: Dict[str, float], iteration: int) -> None:
        """Log training metrics.

        Args:
            training_stats: Training statistics
            iteration: Current iteration
        """
        # Get buffer statistics
        buffer_stats = self.rollout_buffer.get_statistics()

        # Combine all metrics
        metrics = {
            **training_stats,
            **buffer_stats,
            "iteration": iteration,
            "timesteps": self.total_timesteps,
        }

        # Log to tracking system
        self.tracking_manager.log_dict(metrics, iteration)

        # Update best reward
        if "mean_reward" in buffer_stats:
            current_reward = buffer_stats["mean_reward"]
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                metrics["best_reward"] = self.best_reward

        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}:")
            print(f"  Mean Reward: {buffer_stats.get('mean_reward', 0.0):.3f}")
            print(f"  Policy Loss: {training_stats.get('policy_loss', 0.0):.6f}")
            print(f"  Value Loss: {training_stats.get('value_loss', 0.0):.6f}")
            print(f"  Total Timesteps: {self.total_timesteps}")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save training checkpoint.

        Args:
            iteration: Current iteration
        """
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir, f"checkpoint_{iteration}.pt"
        )

        self.ppo.save_checkpoint(checkpoint_path)

        # Also save best model
        if hasattr(self, "best_reward"):
            best_path = os.path.join(self.training_config.checkpoint_dir, "best_model.pt")
            self.ppo.save_checkpoint(best_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.ppo.load_checkpoint(checkpoint_path)
        print(f"Checkpoint loaded: {checkpoint_path}")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating policy for {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            done = False
            while not done:
                # Act deterministically
                actions, _ = self.ppo.act(observations, deterministic=True)
                observations, rewards, dones, _ = self.env.step(actions)

                episode_reward += rewards.mean().item()
                episode_length += 1

                done = dones.any().item()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Compute statistics
        eval_metrics = {
            "eval_mean_reward": sum(episode_rewards) / len(episode_rewards),
            "eval_std_reward": torch.tensor(episode_rewards).std().item(),
            "eval_mean_length": sum(episode_lengths) / len(episode_lengths),
            "eval_std_length": torch.tensor(episode_lengths).std().item(),
            "eval_min_reward": min(episode_rewards),
            "eval_max_reward": max(episode_rewards),
        }

        # Log evaluation metrics
        self.tracking_manager.log_evaluation_metrics(eval_metrics, self.current_iteration)

        print("Evaluation Results:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.3f}")

        return eval_metrics
