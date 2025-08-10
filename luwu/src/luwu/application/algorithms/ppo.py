"""PPO (Proximal Policy Optimization) algorithm implementation."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from luwu.domain.entities import TrainingConfig


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ) -> None:
        """Initialize Actor-Critic network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            init_noise_std: Initial noise standard deviation
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Get activation function
        if activation == "elu":
            activation_fn = nn.ELU
        elif activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Shared feature extractor
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    activation_fn(),
                ]
            )
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor_mean = nn.Linear(input_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.log(torch.ones(action_dim) * init_noise_std))

        # Critic head (value function)
        self.critic = nn.Linear(input_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Special initialization for actor mean
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            observations: Input observations

        Returns:
            Tuple of (action_mean, value)
        """
        features = self.feature_extractor(observations)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def act(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate actions from observations.

        Args:
            observations: Input observations
            deterministic: Whether to act deterministically

        Returns:
            Tuple of (actions, log_probs)
        """
        action_mean, _ = self.forward(observations)

        if deterministic:
            return action_mean, torch.zeros_like(action_mean)

        action_std = torch.exp(self.actor_logstd)
        dist = Normal(action_mean, action_std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        return actions, log_probs

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training.

        Args:
            observations: Input observations
            actions: Actions to evaluate

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_mean, values = self.forward(observations)

        action_std = torch.exp(self.actor_logstd)
        dist = Normal(action_mean, action_std)

        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_probs, values, entropy


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Initialize PPO algorithm.

        Args:
            actor_critic: Actor-critic network
            config: Training configuration
            device: Training device
        """
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Training statistics
        self.stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy using PPO.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            old_log_probs: Old action log probabilities
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            Dictionary of training statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        total_stats = {key: 0.0 for key in self.stats.keys()}
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            batch_size = observations.shape[0]
            mini_batch_size = self.config.mini_batch_size
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # Mini-batch data
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Evaluate actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(mb_obs, mb_actions)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()

                # Update statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                    )

                total_stats["policy_loss"] += policy_loss.item()
                total_stats["value_loss"] += value_loss.item()
                total_stats["entropy_loss"] += entropy_loss.item()
                total_stats["total_loss"] += total_loss.item()
                total_stats["approx_kl"] += approx_kl
                total_stats["clip_fraction"] += clip_fraction

                num_updates += 1

        # Average statistics
        for key in total_stats:
            total_stats[key] /= num_updates

        self.stats.update(total_stats)
        return total_stats

    def act(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate actions from observations.

        Args:
            observations: Input observations
            deterministic: Whether to act deterministically

        Returns:
            Tuple of (actions, log_probs)
        """
        with torch.no_grad():
            return self.actor_critic.act(observations, deterministic)

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate value function.

        Args:
            observations: Input observations

        Returns:
            Value estimates
        """
        with torch.no_grad():
            _, values = self.actor_critic.forward(observations)
            return values  # Keep the original shape [batch_size, 1]

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def get_stats(self) -> Dict[str, float]:
        """Get training statistics.

        Returns:
            Dictionary of training statistics
        """
        return self.stats.copy()
