"""Gymnasium-compatible environment interface for legged robot parkour training."""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from luwu.domain.entities import EnvironmentConfig, RobotConfig, RobotState, SimulationBackend
from luwu.infrastructure.config import config_manager


class LeggedParkourEnv(gym.Env):
    """Gymnasium-compatible environment for legged robot parkour training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        robot_config: RobotConfig,
        env_config: EnvironmentConfig,
        simulation_backend: SimulationBackend,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the parkour environment.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
            simulation_backend: Simulation backend implementation
            render_mode: Rendering mode
        """
        super().__init__()

        self.robot_config = robot_config
        self.env_config = env_config
        self.simulation_backend = simulation_backend
        self.render_mode = render_mode

        # Initialize simulation backend
        sim_config = config_manager.get_simulation_config()
        self.simulation_backend.initialize(sim_config)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env_config.action_space_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env_config.observation_space_dim,),
            dtype=np.float32,
        )

        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.current_states: Optional[RobotState] = None

        # Reward tracking
        self.reward_components: Dict[str, float] = {}
        self.episode_rewards: Dict[str, float] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)

        # Reset simulation
        observations = self.simulation_backend.reset()

        # Reset counters
        self.step_count = 0
        self.episode_count += 1

        # Reset reward tracking
        self.reward_components = {}
        self.episode_rewards = {}

        # Convert to numpy array for gymnasium compatibility
        obs_np = (
            observations.cpu().numpy() if isinstance(observations, torch.Tensor) else observations
        )

        info = {
            "episode": self.episode_count,
            "reward_components": self.reward_components.copy(),
        }

        return obs_np, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).float()
        else:
            action_tensor = action

        # Step simulation
        observations, rewards, dones, info = self.simulation_backend.step(action_tensor)

        # Update step count
        self.step_count += 1

        # Convert outputs for gymnasium compatibility
        obs_np = (
            observations.cpu().numpy() if isinstance(observations, torch.Tensor) else observations
        )
        reward_scalar = (
            rewards.mean().item() if isinstance(rewards, torch.Tensor) else float(rewards)
        )
        terminated = bool(dones.any().item()) if isinstance(dones, torch.Tensor) else bool(dones)
        truncated = self.step_count >= self.env_config.episode_length

        # Update reward tracking
        if "reward_components" in info:
            for name, value in info["reward_components"].items():
                if name not in self.reward_components:
                    self.reward_components[name] = 0.0
                    self.episode_rewards[name] = 0.0

                component_reward = value.item() if isinstance(value, torch.Tensor) else float(value)
                self.reward_components[name] = component_reward
                self.episode_rewards[name] += component_reward

        # Update info
        info.update(
            {
                "step": self.step_count,
                "episode": self.episode_count,
                "reward_components": self.reward_components.copy(),
                "episode_rewards": self.episode_rewards.copy(),
            }
        )

        return obs_np, reward_scalar, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            Rendered image if render_mode is "rgb_array"
        """
        if self.render_mode == "human":
            # Human rendering is handled by the simulation backend
            return None
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            # This would need to be implemented in the simulation backend
            return np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            return None

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if hasattr(self, "simulation_backend"):
            self.simulation_backend.cleanup()

    def get_reward_info(self) -> Dict[str, Any]:
        """Get detailed reward information.

        Returns:
            Dictionary containing reward component information
        """
        return {
            "current_rewards": self.reward_components.copy(),
            "episode_rewards": self.episode_rewards.copy(),
            "reward_weights": {
                comp.name: comp.weight for comp in self.env_config.reward_components
            },
        }


class VectorizedLeggedParkourEnv:
    """Vectorized version of the legged parkour environment for efficient training."""

    def __init__(
        self,
        robot_config: RobotConfig,
        env_config: EnvironmentConfig,
        simulation_backend: SimulationBackend,
        num_envs: Optional[int] = None,
    ) -> None:
        """Initialize vectorized environment.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
            simulation_backend: Simulation backend implementation
            num_envs: Number of parallel environments (defaults to env_config.num_envs)
        """
        self.robot_config = robot_config
        self.env_config = env_config
        self.simulation_backend = simulation_backend
        self.num_envs = num_envs or env_config.num_envs

        # Initialize simulation backend
        sim_config = config_manager.get_simulation_config()
        sim_config["num_envs"] = self.num_envs
        self.simulation_backend.initialize(sim_config)

        # Training state
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.int32)
        self.episode_counts = torch.zeros(self.num_envs, dtype=torch.int32)

        # Reward tracking
        self.episode_rewards = {}

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset specified environments.

        Args:
            env_ids: Environment IDs to reset (None resets all)

        Returns:
            Observations for reset environments
        """
        observations = self.simulation_backend.reset(env_ids)

        if env_ids is None:
            self.step_counts.fill_(0)
            self.episode_counts += 1
        else:
            self.step_counts[env_ids] = 0
            self.episode_counts[env_ids] += 1

        return observations

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one step for all environments.

        Args:
            actions: Actions for all environments

        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        observations, rewards, dones, info = self.simulation_backend.step(actions)

        # Update step counts
        self.step_counts += 1

        # Check for episode length termination
        episode_length_dones = self.step_counts >= self.env_config.episode_length
        dones = dones | episode_length_dones

        # Auto-reset terminated environments
        if dones.any():
            reset_env_ids = dones.nonzero(as_tuple=False).flatten()
            reset_obs = self.reset(reset_env_ids)
            observations[reset_env_ids] = reset_obs

        return observations, rewards, dones, info

    def get_observations(self) -> torch.Tensor:
        """Get current observations for all environments.

        Returns:
            Current observations
        """
        return self.simulation_backend.get_observations()

    def close(self) -> None:
        """Close all environments and clean up resources."""
        self.simulation_backend.cleanup()
