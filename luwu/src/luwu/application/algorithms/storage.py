"""Experience storage and advantage computation for PPO."""

from typing import Dict, Generator, Optional, Tuple

import torch

from luwu.domain.entities import TrainingConfig


class RolloutBuffer:
    """Rollout buffer for storing and processing training data."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Initialize rollout buffer.

        Args:
            num_steps: Number of steps per rollout
            num_envs: Number of parallel environments
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Device to store tensors
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage tensors
        self.observations = torch.zeros(
            (num_steps, num_envs, obs_dim), device=device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            (num_steps, num_envs, action_dim), device=device, dtype=torch.float32
        )
        self.log_probs = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.float32)
        self.rewards = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.float32)
        self.values = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.float32)
        self.dones = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.bool)

        # Computed arrays
        self.returns = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.float32)
        self.advantages = torch.zeros((num_steps, num_envs, 1), device=device, dtype=torch.float32)

        # Tracking variables
        self.step = 0
        self.full = False

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a step of experience to the buffer.

        Args:
            observations: Observations
            actions: Actions taken
            log_probs: Log probabilities of actions
            rewards: Rewards received
            values: Value estimates
            dones: Done flags
        """
        if self.step >= self.num_steps:
            raise ValueError("Buffer is full")

        self.observations[self.step] = observations
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs.unsqueeze(-1) if log_probs.dim() == 1 else log_probs
        self.rewards[self.step] = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards
        self.values[self.step] = values.unsqueeze(-1) if values.dim() == 1 else values
        self.dones[self.step] = dones.unsqueeze(-1) if dones.dim() == 1 else dones

        self.step += 1
        if self.step == self.num_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: torch.Tensor) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_values: Value estimates for the last step
        """
        if not self.full:
            raise ValueError("Buffer is not full")

        # Extend values with last_values for bootstrap
        extended_values = torch.cat([self.values, last_values.unsqueeze(0)], dim=0)

        # Compute GAE advantages
        gae = 0.0
        for step in reversed(range(self.num_steps)):
            delta = (
                self.rewards[step]
                + self.gamma * extended_values[step + 1] * (~self.dones[step])
                - extended_values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (~self.dones[step]) * gae
            self.advantages[step] = gae

        # Compute returns
        self.returns = self.advantages + self.values

    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get a batch of data from the buffer.

        Args:
            batch_size: Size of the batch (None for all data)

        Returns:
            Dictionary containing batch data
        """
        if not self.full:
            raise ValueError("Buffer is not full")

        # Flatten tensors
        observations = self.observations.view(-1, self.obs_dim)
        actions = self.actions.view(-1, self.action_dim)
        log_probs = self.log_probs.view(-1, 1)
        returns = self.returns.view(-1, 1)
        advantages = self.advantages.view(-1, 1)
        values = self.values.view(-1, 1)

        if batch_size is None:
            return {
                "observations": observations,
                "actions": actions,
                "log_probs": log_probs,
                "returns": returns,
                "advantages": advantages,
                "values": values,
            }

        # Sample random batch
        total_size = observations.shape[0]
        indices = torch.randperm(total_size, device=self.device)[:batch_size]

        return {
            "observations": observations[indices],
            "actions": actions[indices],
            "log_probs": log_probs[indices],
            "returns": returns[indices],
            "advantages": advantages[indices],
            "values": values[indices],
        }

    def clear(self) -> None:
        """Clear the buffer for next rollout."""
        self.step = 0
        self.full = False

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics.

        Returns:
            Dictionary of buffer statistics
        """
        if not self.full:
            return {}

        return {
            "mean_reward": self.rewards.mean().item(),
            "std_reward": self.rewards.std().item(),
            "mean_value": self.values.mean().item(),
            "std_value": self.values.std().item(),
            "mean_advantage": self.advantages.mean().item(),
            "std_advantage": self.advantages.std().item(),
            "mean_return": self.returns.mean().item(),
            "std_return": self.returns.std().item(),
        }


class RewardTracker:
    """Track and compute reward components."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        """Initialize reward tracker.

        Args:
            num_envs: Number of parallel environments
            device: Device to store tensors
        """
        self.num_envs = num_envs
        self.device = device

        # Reward storage
        self.episode_rewards: Dict[str, torch.Tensor] = {}
        self.step_rewards: Dict[str, torch.Tensor] = {}

    def reset_episode_rewards(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset episode rewards for specified environments.

        Args:
            env_ids: Environment IDs to reset (None resets all)
        """
        if env_ids is None:
            for name in self.episode_rewards:
                self.episode_rewards[name].fill_(0.0)
        else:
            for name in self.episode_rewards:
                self.episode_rewards[name][env_ids] = 0.0

    def add_reward_component(self, name: str, rewards: torch.Tensor) -> None:
        """Add a reward component.

        Args:
            name: Name of the reward component
            rewards: Reward values for all environments
        """
        if name not in self.episode_rewards:
            self.episode_rewards[name] = torch.zeros(self.num_envs, device=self.device)

        self.step_rewards[name] = rewards
        self.episode_rewards[name] += rewards

    def get_total_reward(self) -> torch.Tensor:
        """Get total reward for current step.

        Returns:
            Total reward across all components
        """
        if not self.step_rewards:
            return torch.zeros(self.num_envs, device=self.device)

        return sum(self.step_rewards.values())

    def get_reward_info(self) -> Dict[str, torch.Tensor]:
        """Get current reward information.

        Returns:
            Dictionary of reward components
        """
        return {
            "step_rewards": self.step_rewards.copy(),
            "episode_rewards": self.episode_rewards.copy(),
            "total_step_reward": self.get_total_reward(),
            "total_episode_reward": (
                sum(self.episode_rewards.values())
                if self.episode_rewards
                else torch.zeros(self.num_envs, device=self.device)
            ),
        }

    def clear_step_rewards(self) -> None:
        """Clear step rewards for next step."""
        self.step_rewards.clear()

    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics.

        Returns:
            Dictionary of reward statistics
        """
        stats = {}

        for name, rewards in self.step_rewards.items():
            stats[f"step_{name}_mean"] = rewards.mean().item()
            stats[f"step_{name}_std"] = rewards.std().item()

        for name, rewards in self.episode_rewards.items():
            stats[f"episode_{name}_mean"] = rewards.mean().item()
            stats[f"episode_{name}_std"] = rewards.std().item()

        if self.step_rewards:
            total_step = sum(self.step_rewards.values())
            stats["total_step_reward_mean"] = total_step.mean().item()
            stats["total_step_reward_std"] = total_step.std().item()

        if self.episode_rewards:
            total_episode = sum(self.episode_rewards.values())
            stats["total_episode_reward_mean"] = total_episode.mean().item()
            stats["total_episode_reward_std"] = total_episode.std().item()

        return stats
