"""Mujoco simulation backend for legged robot parkour."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from luwu.domain.entities import RobotConfig, SimulationBackend


class MujocoSimulation(SimulationBackend):
    """Mujoco simulation backend implementation."""

    def __init__(self, robot_config: RobotConfig, env_config: Dict[str, Any]) -> None:
        """Initialize Mujoco simulation.

        Args:
            robot_config: Robot configuration
            env_config: Environment configuration
        """
        self.robot_config = robot_config
        self.env_config = env_config
        self.initialized = False

        # Simulation state
        self.num_envs = 1
        self.dt = 0.02
        self.substeps = 4

        # Will be initialized in initialize()
        self.model = None
        self.data = None
        self.viewer = None

        # Environment state
        self.current_step = 0
        self.max_episode_steps = 1000

        # Observations and actions
        self.obs_dim = env_config.get("observation_space_dim", 48)
        self.action_dim = env_config.get("action_space_dim", 12)

        # Device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Mujoco simulation.

        Args:
            config: Simulation configuration
        """
        try:
            import mujoco

            self.mujoco = mujoco
        except ImportError:
            raise ImportError("mujoco package is required for Mujoco simulation")

        # Update configuration
        self.num_envs = config.get("num_envs", 1)
        self.dt = config.get("dt", 0.02)
        self.substeps = config.get("substeps", 4)
        self.max_episode_steps = config.get("max_episode_steps", 1000)

        # Load robot model
        mujoco_path = getattr(self.robot_config, "mujoco_path", None)
        if not mujoco_path:
            urdf_path = self.robot_config.urdf_path
            if not urdf_path.endswith(".xml"):
                raise ValueError("Mujoco requires XML model files")
            model_path = urdf_path
        else:
            model_path = mujoco_path

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Set timestep
        self.model.opt.timestep = self.dt / self.substeps

        # Initialize viewer if needed
        rendering_config = config.get("rendering", {})
        if rendering_config.get("enable", False):
            self._init_viewer()

        # Initialize robot to default position
        self._reset_robot()

        self.initialized = True
        print(f"Mujoco simulation initialized with {self.num_envs} environments")

    def _init_viewer(self) -> None:
        """Initialize MuJoCo viewer."""
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except ImportError:
            print("Warning: MuJoCo viewer not available")
            self.viewer = None

    def _reset_robot(self) -> None:
        """Reset robot to default configuration."""
        # Set joint positions to default
        if hasattr(self.robot_config, "default_joint_positions"):
            joint_positions = self.robot_config.default_joint_positions
            if len(joint_positions) <= len(self.data.qpos):
                self.data.qpos[: len(joint_positions)] = joint_positions

        # Reset velocities
        self.data.qvel[:] = 0.0

        # Forward kinematics
        self.mujoco.mj_forward(self.model, self.data)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the simulation forward.

        Args:
            actions: Actions to execute

        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        if not self.initialized:
            raise RuntimeError("Simulation not initialized")

        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = np.array(actions)

        # Ensure correct shape
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        # Apply actions (assuming position control)
        self._apply_actions(actions_np[0])

        # Step simulation
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)

        # Update viewer if available
        if self.viewer is not None:
            self.viewer.sync()

        # Get observations
        observations = self._get_observations()

        # Compute rewards
        rewards = self._compute_rewards(actions_np[0])

        # Check termination
        dones = self._check_termination()

        # Update step counter
        self.current_step += 1

        # Create info dictionary
        info = {
            "step": self.current_step,
            "time": self.current_step * self.dt,
            "reward_components": self._get_reward_components(),
        }

        # Convert to tensors with correct batch dimension
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        # Repeat for all environments since MuJoCo simulates single environment
        obs_tensor = obs_tensor.unsqueeze(0).expand(self.num_envs, -1)

        reward_tensor = torch.tensor([rewards], dtype=torch.float32, device=self.device)
        reward_tensor = reward_tensor.expand(self.num_envs)

        done_tensor = torch.tensor([dones], dtype=torch.bool, device=self.device)
        done_tensor = done_tensor.expand(self.num_envs)

        return obs_tensor, reward_tensor, done_tensor, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply actions to the robot.

        Args:
            actions: Joint actions to apply
        """
        # Scale actions from [-1, 1] to joint limits
        num_actuators = min(len(actions), len(self.data.ctrl))

        for i in range(num_actuators):
            # Simple position control (could be improved)
            self.data.ctrl[i] = actions[i] * 1.0  # Scale factor

    def _get_observations(self) -> np.ndarray:
        """Get current observations.

        Returns:
            Current state observations
        """
        # Basic observation: joint positions and velocities
        joint_pos = self.data.qpos[: self.robot_config.num_joints].copy()
        joint_vel = self.data.qvel[: self.robot_config.num_joints].copy()

        # Body orientation (quaternion)
        body_quat = self.data.qpos[3:7].copy()  # Assuming free joint

        # Body angular velocity
        body_angvel = self.data.qvel[3:6].copy()  # Assuming free joint

        # Combine observations
        observations = np.concatenate(
            [
                joint_pos,
                joint_vel,
                body_quat,
                body_angvel,
            ]
        )

        # Pad or truncate to match expected observation dimension
        if len(observations) < self.obs_dim:
            observations = np.pad(observations, (0, self.obs_dim - len(observations)))
        elif len(observations) > self.obs_dim:
            observations = observations[: self.obs_dim]

        return observations.astype(np.float32)

    def _compute_rewards(self, actions: np.ndarray) -> float:
        """Compute reward for current state.

        Args:
            actions: Actions that were taken

        Returns:
            Total reward
        """
        reward = 0.0

        # Survival reward
        reward += 1.0

        # Forward velocity reward
        body_vel = self.data.qvel[0]  # Forward velocity
        reward += body_vel * 2.0

        # Upright reward
        body_quat = self.data.qpos[3:7]
        # Convert quaternion to upright measure (simplified)
        upright = 1.0 - abs(body_quat[1]) - abs(body_quat[2])  # Penalize roll and pitch
        reward += upright * 1.0

        # Action penalty
        action_penalty = np.sum(actions**2) * 0.01
        reward -= action_penalty

        return reward

    def _get_reward_components(self) -> Dict[str, float]:
        """Get individual reward components.

        Returns:
            Dictionary of reward components
        """
        components = {}

        # Survival
        components["survival"] = 1.0

        # Forward velocity
        body_vel = self.data.qvel[0]
        components["forward_velocity"] = body_vel * 2.0

        # Upright
        body_quat = self.data.qpos[3:7]
        upright = 1.0 - abs(body_quat[1]) - abs(body_quat[2])
        components["upright"] = upright * 1.0

        return components

    def _check_termination(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode should terminate
        """
        # Terminate if fallen (body too low)
        body_height = self.data.qpos[2]
        if body_height < 0.3:  # Threshold for fallen
            return True

        # Terminate if maximum steps reached
        if self.current_step >= self.max_episode_steps:
            return True

        return False

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments.

        Args:
            env_ids: Environment IDs to reset (None for all environments)

        Returns:
            Initial observations for reset environments
        """
        # Reset robot configuration
        self._reset_robot()

        # Reset step counter
        self.current_step = 0

        # Get initial observations
        observations = self._get_observations()

        # Convert to tensor
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # Determine number of environments to reset
        if env_ids is None:
            num_reset_envs = self.num_envs
        else:
            num_reset_envs = len(env_ids)

        # Return observations with correct batch dimension
        # For single MuJoCo environment, repeat the observation for all reset environments
        return obs_tensor.unsqueeze(0).expand(num_reset_envs, -1)

    def get_observations(self) -> torch.Tensor:
        """Get current observations.

        Returns:
            Current observations as tensor with correct batch dimension
        """
        observations = self._get_observations()
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        # Return observations with correct batch dimension for all environments
        return obs_tensor.unsqueeze(0).expand(self.num_envs, -1)

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        print("Mujoco simulation cleaned up")

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the simulation.

        Args:
            mode: Rendering mode

        Returns:
            Rendered image if mode is "rgb_array"
        """
        if mode == "human":
            if self.viewer is None:
                self._init_viewer()
            return None
        elif mode == "rgb_array":
            # Render to RGB array
            renderer = self.mujoco.Renderer(self.model, 640, 480)
            renderer.update_scene(self.data)
            image = renderer.render()
            renderer.close()
            return image
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
