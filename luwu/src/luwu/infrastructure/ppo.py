"""
PPO (Proximal Policy Optimization) algorithm implementation.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from luwu.domain.entities import Policy, TrainingMetrics


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
    ) -> None:
        """Initialize the Actor-Critic network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "elu":
            act_fn = nn.ELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Shared feature extraction layers
        shared_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                ]
            )
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*shared_layers)

        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], action_dim),
        )

        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1),
        )

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            observations: Input observations

        Returns:
            Tuple of (action_mean, value)
        """
        features = self.shared_layers(observations)

        action_mean = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)

        return action_mean, value

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions and values with log probabilities.

        Args:
            observations: Input observations
            actions: Actions (if None, sample new actions)

        Returns:
            Tuple of (actions, log_probs, entropy, values)
        """
        action_mean, values = self.forward(observations)

        # Create action distribution
        action_std = torch.exp(self.log_std)
        action_dist = Normal(action_mean, action_std)

        if actions is None:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)

        return actions, log_probs, entropy, values


class PPOPolicy(Policy):
    """PPO policy implementation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict,
        device: str = "cuda",
    ) -> None:
        """Initialize the PPO policy.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: PPO configuration
            device: Device to run on
        """
        self.device = torch.device(device)
        self.config = config

        # Create network
        algo_config = config.get("algorithm", {})
        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=algo_config.get("actor_hidden_dims", [512, 256, 128]),
            activation=algo_config.get("activation", "elu"),
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=algo_config.get("learning_rate", 3e-4),
        )

        # PPO hyperparameters
        self.clip_param = algo_config.get("clip_param", 0.2)
        self.value_loss_coef = algo_config.get("value_loss_coef", 1.0)
        self.entropy_coef = algo_config.get("entropy_coef", 0.01)
        self.max_grad_norm = algo_config.get("max_grad_norm", 1.0)
        self.gamma = algo_config.get("gamma", 0.998)
        self.lambda_gae = algo_config.get("lambda", 0.95)

        # Training parameters
        self.num_epochs = algo_config.get("num_epochs", 5)
        self.num_mini_batches = algo_config.get("num_mini_batches", 4)

        # Tracking
        self.training_step = 0

    def predict(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict actions and values.

        Args:
            observations: Input observations

        Returns:
            Tuple of (actions, values)
        """
        with torch.no_grad():
            actions, _, _, values = self.network.get_action_and_value(observations)
        return actions, values

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> TrainingMetrics:
        """Update the policy using PPO.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            dones: Batch of done flags
            values: Batch of values

        Returns:
            Training metrics
        """
        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)

        # Get old action log probabilities
        with torch.no_grad():
            _, old_log_probs, _, _ = self.network.get_action_and_value(observations, actions)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0

        # Create mini-batches and train
        batch_size = observations.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches

        for epoch in range(self.num_epochs):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # Get mini-batch data
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

                # Additional metrics
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    total_clipfrac += clipfrac.item()
                    total_approx_kl += approx_kl.item()

        # Average metrics
        num_updates = self.num_epochs * self.num_mini_batches
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_clipfrac = total_clipfrac / num_updates
        avg_approx_kl = total_approx_kl / num_updates

        # Calculate explained variance
        explained_var = self._explained_variance(values, returns)

        self.training_step += 1

        return TrainingMetrics(
            episode_reward=rewards.sum().item() / rewards.shape[0],  # Average episode reward
            episode_length=int((~dones).sum().item() / dones.shape[0]),  # Average episode length
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            entropy_loss=avg_entropy_loss,
            explained_variance=explained_var,
            clipfrac=avg_clipfrac,
            approx_kl=avg_approx_kl,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            reward_components={},  # Would need to be passed in
        )

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor
            values: Value tensor
            dones: Done flags tensor

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Compute next values (for bootstrap)
        next_values = torch.cat([values[1:], torch.zeros_like(values[-1:])], dim=0)
        next_values = next_values * (1 - dones.float())

        # Compute advantages backwards
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = last_advantage = (
                delta + self.gamma * self.lambda_gae * last_advantage * (1 - dones[t])
            )

        returns = advantages + values
        return advantages, returns

    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Calculate explained variance.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Explained variance
        """
        var_y = torch.var(y_true)
        if var_y == 0:
            return 0.0
        return 1 - torch.var(y_true - y_pred) / var_y

    def save(self, path: str) -> None:
        """Save the policy.

        Args:
            path: Path to save the policy
        """
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the policy.

        Args:
            path: Path to load the policy from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
