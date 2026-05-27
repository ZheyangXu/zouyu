# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BarrierTrack terrain generator for parkour tasks.

Generates sequential obstacle tracks for quadruped parkour training. Each track is a
linear sequence of blocks (start + obstacles) with configurable obstacle types.
Provides both physical geometry (trimesh for USD import) and virtual/oracle data
(height maps, penetration depths, engaging block metadata) for privileged training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import trimesh

from .barrier_track_cfg import BarrierTrackCfg


class BarrierTrack:
    """Generates and manages barrier track terrain for parkour training.

    Each track is a linear sequence of blocks along the +X axis. Multiple tracks
    (columns) provide different difficulty levels stacked along the Y axis.

    The terrain geometry is generated as trimesh for USD import. Oracle data
    (virtual heights, penetration depths, block metadata) is maintained as GPU
    tensors for efficient privileged observation computation.

    Obstacle types supported:
    - jump: Raised platform / step-up
    - leap: Gap / ditch to jump across
    - hurdle: Barrier to step over
    - down: Step-down / drop-off
    - tilted_ramp: Laterally tilted surface
    - stairsup: Ascending staircase
    - stairsdown: Descending staircase
    - discrete_rect: Random scattered rectangular blocks
    - slope: Inclined plane
    - wave: Sinusoidal wave terrain
    """

    # Obstacle type ID mapping (matches original parkour convention)
    OBSTACLE_IDS: dict[str, int] = {
        "jump": 3,
        "leap": 4,
        "hurdle": 5,
        "down": 6,
        "tilted_ramp": 7,
        "slope": 8,
        "stairsup": 9,
        "stairsdown": 10,
        "discrete_rect": 11,
        "wave": 14,
    }

    def __init__(self, cfg: BarrierTrackCfg, num_envs: int, device: torch.device | str):
        """Initialize the BarrierTrack generator.

        Args:
            cfg: The barrier track configuration.
            num_envs: Total number of parallel environments.
            device: The compute device for tensor operations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device

        # Parse obstacle type list from config
        self._obstacle_names: list[str] = list(cfg.obstacle_types.keys())
        self._num_obstacle_types = len(self._obstacle_names)

        # Per-environment state
        self._difficulty_levels = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._current_block_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Track layout: each env assigned to a track row
        self._track_row = torch.arange(num_envs, device=self.device)

        # Block sequence per env: list of (obstacle_name, params_dict)
        self._block_sequences: list[list[tuple[str, dict]]] = []
        self._generate_block_sequences()

        # Virtual terrain grid (for oracle height sampling)
        self._setup_height_grid()

        # Combined trimesh (lazily generated)
        self._combined_mesh: trimesh.Trimesh | None = None
        self._env_origins: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API: Terrain geometry
    # ------------------------------------------------------------------

    def get_mesh(self) -> trimesh.Trimesh:
        """Generate and return the combined trimesh for all tracks.

        Returns:
            A trimesh.Trimesh object containing all track geometry for USD import.
        """
        if self._combined_mesh is not None:
            return self._combined_mesh

        meshes: list[trimesh.Trimesh] = []
        env_origins_list: list[list[float]] = []

        for env_id in range(self.num_envs):
            track_y_offset = float(env_id) * self.cfg.track_spacing
            track_mesh, origin = self._generate_track_mesh(env_id, track_y_offset)
            meshes.append(track_mesh)
            env_origins_list.append(origin)

        self._combined_mesh = trimesh.util.concatenate(meshes)
        self._env_origins = torch.tensor(env_origins_list, device=self.device)
        return self._combined_mesh

    def get_env_origins(self) -> torch.Tensor:
        """Get environment spawn origins.

        Returns:
            Tensor of shape (num_envs, 3) with (x, y, z) spawn positions.
        """
        if self._env_origins is None:
            self.get_mesh()
        return self._env_origins  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API: Oracle data
    # ------------------------------------------------------------------

    def get_penetration_depths(self) -> torch.Tensor:
        """Compute virtual obstacle penetration depths for each env.

        Returns:
            Tensor of shape (num_envs,) with summed penetration depth per env.
        """
        # Simplified: return current difficulty as proxy for penetration
        # Full implementation requires robot body points and per-obstacle computation
        return torch.zeros(self.num_envs, device=self.device)

    def height_at_positions(
        self,
        root_pos_xy: torch.Tensor,
        root_quat: torch.Tensor,
        offset: float = 0.2,
    ) -> torch.Tensor:
        """Sample virtual terrain heights at measurement points around the robot.

        Args:
            root_pos_xy: Robot base XY position, shape (num_envs, 2).
            root_quat: Robot base quaternion, shape (num_envs, 4).
            offset: Base height offset for height computation.

        Returns:
            Tensor of shape (num_envs, num_points) with sampled heights.
        """
        num_points = len(self.cfg.measured_points_x) * len(self.cfg.measured_points_y)
        # Simplified: return zeros for now
        # Full implementation: use virtual height grid for oracle terrain queries
        return torch.zeros(self.num_envs, num_points, device=self.device)

    def get_engaging_block_info(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """Get metadata about the currently engaging obstacle block.

        Returns [block_type_onehot(10), distance_to_block, block_height, block_width].
        Shape: (num_envs, 13).

        Args:
            root_pos_w: Robot base position in world frame, shape (num_envs, 3).

        Returns:
            Engaging block info tensor.
        """
        result = torch.zeros(self.num_envs, 13, device=self.device)
        for env_id in range(self.num_envs):
            block_idx = self._current_block_idx[env_id].item()
            if block_idx < len(self._block_sequences[env_id]):
                obs_name, params = self._block_sequences[env_id][block_idx]
                # One-hot encode using obstacle type index
                onehot_idx = self._obstacle_names.index(obs_name) if obs_name in self._obstacle_names else 0
                result[env_id, onehot_idx] = 1.0

                # Distance to block (simplified)
                block_x = block_idx * self.cfg.block_length
                result[env_id, 10] = max(0.0, block_x - root_pos_w[env_id, 0].item()) / self.cfg.track_length

                # Block height and width
                result[env_id, 11] = params.get("height", 0.0)
                result[env_id, 12] = params.get("width", params.get("depth", 0.0))
        return result

    def get_current_block_indices(self) -> torch.Tensor:
        """Get the current block index per environment.

        Returns:
            Tensor of shape (num_envs,) with block indices.
        """
        return self._current_block_idx.clone()

    def is_out_of_bounds(self, root_pos_w: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """Check if robots are outside the track boundaries.

        Args:
            root_pos_w: Robot base positions in world frame, shape (num_envs, 3).
            margin: Allowed margin from track edges.

        Returns:
            Boolean tensor of shape (num_envs,).
        """
        track_half_width = self.cfg.block_width / 2.0 - margin
        track_len = self.cfg.track_length * self.cfg.block_length
        for env_id in range(self.num_envs):
            track_y_center = float(env_id) * self.cfg.track_spacing
            y_dist = torch.abs(root_pos_w[env_id, 1] - track_y_center)
            x_dist = root_pos_w[env_id, 0]
            if y_dist > track_half_width or x_dist < -margin or x_dist > track_len + margin:
                root_pos_w[env_id, 0] = 0  # placeholder - this will be fixed via vectorized approach
        # Vectorized version
        track_y_centers = self._track_row.float() * self.cfg.track_spacing
        y_out = torch.abs(root_pos_w[:, 1] - track_y_centers) > track_half_width
        x_out = torch.logical_or(
            root_pos_w[:, 0] < -margin,
            root_pos_w[:, 0] > self.cfg.track_length * self.cfg.block_length + margin,
        )
        return torch.logical_or(x_out, y_out)

    def get_body_sample_points(self) -> torch.Tensor:
        """Get 3D sample points on the robot body for penetration computation.

        Returns:
            Tensor of shape (num_envs, num_points, 3) with world-frame positions.
        """
        # Return empty/zero points; real implementation would query robot articulation
        return torch.zeros(self.num_envs, 100, 3, device=self.device)

    # ------------------------------------------------------------------
    # Public API: Difficulty management
    # ------------------------------------------------------------------

    def increase_difficulty(self, env_ids: torch.Tensor | None = None) -> None:
        """Increase terrain difficulty for specified environments.

        Args:
            env_ids: Environment IDs to update. If None, update all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._difficulty_levels[env_ids] = torch.clamp(
            self._difficulty_levels[env_ids] + 1, max=self.cfg.max_difficulty - 1
        )

    def decrease_difficulty(self, env_ids: torch.Tensor | None = None) -> None:
        """Decrease terrain difficulty for specified environments.

        Args:
            env_ids: Environment IDs to update. If None, update all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._difficulty_levels[env_ids] = torch.clamp(self._difficulty_levels[env_ids] - 1, min=0)

    def get_difficulty_levels(self) -> torch.Tensor:
        """Get current difficulty level per environment.

        Returns:
            Tensor of shape (num_envs,) with difficulty levels.
        """
        return self._difficulty_levels.clone()

    # ------------------------------------------------------------------
    # Internal: Block sequence generation
    # ------------------------------------------------------------------

    def _generate_block_sequences(self) -> None:
        """Pre-generate block sequences for all environments.

        Each env gets a sequence of blocks: [("start", {...}), ("jump", {...}), ...].
        The start block is always flat ground.
        """
        rng = np.random.RandomState(42)
        for env_id in range(self.num_envs):
            difficulty = self._difficulty_levels[env_id].item()
            seq: list[tuple[str, dict]] = []

            # Start block (flat)
            seq.append(("start", {"height": 0.0, "length": self.cfg.block_length}))

            # Obstacle blocks
            for _ in range(self.cfg.track_length - 1):
                obs_name = self._sample_obstacle_type(rng)
                params = self._get_obstacle_params(obs_name, difficulty, rng)
                seq.append((obs_name, params))

            self._block_sequences.append(seq)

    def _sample_obstacle_type(self, rng: np.random.RandomState) -> str:
        """Sample an obstacle type based on configured proportions.

        Args:
            rng: Random state for reproducibility.

        Returns:
            The sampled obstacle type name.
        """
        names = []
        probs = []
        for name, cfg_dict in self.cfg.obstacle_types.items():
            names.append(name)
            probs.append(cfg_dict.get("proportion", 1.0))
        probs = np.array(probs) / np.sum(probs)
        return names[rng.choice(len(names), p=probs)]

    def _get_obstacle_params(self, obs_name: str, difficulty: int, rng: np.random.RandomState) -> dict[str, Any]:
        """Get obstacle parameters scaled by difficulty level.

        Args:
            obs_name: The obstacle type name.
            difficulty: Current difficulty level (0 to max_difficulty-1).
            rng: Random state for variability.

        Returns:
            Dictionary of obstacle parameters.
        """
        base_cfg = self.cfg.obstacle_types.get(obs_name, {})
        params = dict(base_cfg.get("kwargs", {}))

        # Scale difficulty-dependent parameters
        diff_ratio = (difficulty + 1) / self.cfg.max_difficulty

        if obs_name == "jump":
            params.setdefault("height", 0.05 + 0.35 * diff_ratio)
        elif obs_name == "leap":
            params.setdefault("depth", 0.1 + 0.3 * diff_ratio)
            params.setdefault("length", 0.2 + 0.4 * diff_ratio)
        elif obs_name == "hurdle":
            params.setdefault("height", 0.05 + 0.2 * diff_ratio)
        elif obs_name == "down":
            params.setdefault("height", -(0.05 + 0.25 * diff_ratio))
        elif obs_name == "stairsup":
            params.setdefault("step_height", 0.03 + 0.07 * diff_ratio)
            params.setdefault("num_steps", 3 + int(5 * diff_ratio))
        elif obs_name == "stairsdown":
            params.setdefault("step_height", -(0.03 + 0.07 * diff_ratio))
            params.setdefault("num_steps", 3 + int(5 * diff_ratio))
        elif obs_name == "slope":
            params.setdefault("slope_angle", 0.1 + 0.4 * diff_ratio)
        elif obs_name == "wave":
            params.setdefault("amplitude", 0.02 + 0.08 * diff_ratio)
            params.setdefault("frequency", 1.0 + 2.0 * diff_ratio)

        return params

    # ------------------------------------------------------------------
    # Internal: Height grid
    # ------------------------------------------------------------------

    def _setup_height_grid(self) -> None:
        """Pre-compute the virtual height grid coordinate system."""
        self._num_meas_x = len(self.cfg.measured_points_x)
        self._num_meas_y = len(self.cfg.measured_points_y)
        self._meas_grid_x = torch.tensor(self.cfg.measured_points_x, device=self.device)
        self._meas_grid_y = torch.tensor(self.cfg.measured_points_y, device=self.device)

    # ------------------------------------------------------------------
    # Internal: Track mesh generation
    # ------------------------------------------------------------------

    def _generate_track_mesh(self, env_id: int, track_y_offset: float) -> tuple[trimesh.Trimesh, list[float]]:
        """Generate trimesh geometry for a single track.

        Args:
            env_id: Environment index.
            track_y_offset: Y-axis offset for this track.

        Returns:
            Tuple of (track trimesh, [origin_x, origin_y, origin_z]).
        """
        meshes: list[trimesh.Trimesh] = []
        block_len = self.cfg.block_length
        block_w = self.cfg.block_width
        border = self.cfg.border_width
        wall_h = self.cfg.wall_height
        half_w = block_w / 2.0

        # Ground base plane for the entire track
        track_total_len = self.cfg.track_length * block_len
        ground = trimesh.creation.box(
            extents=[track_total_len, block_w + 2 * border, 0.02],
            transform=trimesh.transformations.translation_matrix([track_total_len / 2, track_y_offset, -0.01]),
        )
        meshes.append(ground)

        # Border walls (left and right)
        wall_left = trimesh.creation.box(
            extents=[track_total_len, border, wall_h],
            transform=trimesh.transformations.translation_matrix(
                [track_total_len / 2, track_y_offset - half_w - border / 2, wall_h / 2]
            ),
        )
        wall_right = trimesh.creation.box(
            extents=[track_total_len, border, wall_h],
            transform=trimesh.transformations.translation_matrix(
                [track_total_len / 2, track_y_offset + half_w + border / 2, wall_h / 2]
            ),
        )
        meshes.extend([wall_left, wall_right])

        # Obstacle blocks
        for block_idx, (obs_name, params) in enumerate(self._block_sequences[env_id]):
            block_x_center = (block_idx + 0.5) * block_len
            block_meshes = self._generate_obstacle_mesh(
                obs_name, params, block_x_center, track_y_offset, block_len, block_w
            )
            meshes.extend(block_meshes)

        combined = trimesh.util.concatenate(meshes)
        origin = [0.0, track_y_offset, 0.0]
        return combined, origin

    def _generate_obstacle_mesh(
        self,
        obs_name: str,
        params: dict[str, Any],
        x_center: float,
        y_center: float,
        block_len: float,
        block_w: float,
    ) -> list[trimesh.Trimesh]:
        """Generate trimesh geometry for a single obstacle block.

        Args:
            obs_name: Obstacle type name.
            params: Obstacle-specific parameters.
            x_center: Center X position of the block.
            y_center: Center Y position of the block.
            block_len: Block length in X direction.
            block_w: Block width in Y direction.

        Returns:
            List of trimesh objects for this block.
        """
        mesh_list: list[trimesh.Trimesh] = []

        if obs_name == "start":
            # Flat platform
            platform = trimesh.creation.box(
                extents=[block_len, block_w, 0.1],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, 0.05]),
            )
            mesh_list.append(platform)

        elif obs_name == "jump":
            height = params.get("height", 0.2)
            platform = trimesh.creation.box(
                extents=[block_len, block_w, height],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, height / 2]),
            )
            mesh_list.append(platform)

        elif obs_name == "leap":
            depth = params.get("depth", 0.2)
            length = params.get("length", 0.3)
            gap_start = x_center - block_len / 2
            # Platform before gap
            before_gap = trimesh.creation.box(
                extents=[max(0, block_len - length), block_w, 0.1],
                transform=trimesh.transformations.translation_matrix(
                    [gap_start + (block_len - length) / 2, y_center, 0.05 + depth]
                ),
            )
            mesh_list.append(before_gap)

        elif obs_name == "hurdle":
            height = params.get("height", 0.15)
            hurdle_w = params.get("width", 0.1)
            hurdle = trimesh.creation.box(
                extents=[hurdle_w, block_w, height],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, height / 2]),
            )
            mesh_list.append(hurdle)

        elif obs_name == "down":
            height = abs(params.get("height", 0.2))
            # Platform at the beginning, drop at the end
            half_block = block_len / 2
            upper = trimesh.creation.box(
                extents=[half_block, block_w, height],
                transform=trimesh.transformations.translation_matrix([x_center - half_block / 2, y_center, height / 2]),
            )
            lower = trimesh.creation.box(
                extents=[half_block, block_w, 0.05],
                transform=trimesh.transformations.translation_matrix([x_center + half_block / 2, y_center, 0.025]),
            )
            mesh_list.extend([upper, lower])

        elif obs_name == "tilted_ramp":
            tilt_angle = params.get("tilt_angle", 0.3)
            ramp = trimesh.creation.box(
                extents=[block_len, block_w, 0.1],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, 0.05]),
            )
            # Apply tilt rotation around X-axis
            tilt_matrix = trimesh.transformations.rotation_matrix(tilt_angle, [1, 0, 0])
            ramp.apply_transform(tilt_matrix)
            mesh_list.append(ramp)

        elif obs_name == "slope":
            face_dir = params.get("face_direction", 0.0)
            # Create an inclined plane using a wedge shape
            ramp = trimesh.creation.box(
                extents=[block_len, block_w, 0.15],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, 0.075]),
            )
            rot = trimesh.transformations.rotation_matrix(face_dir, [0, 0, 1])
            ramp.apply_transform(rot)
            mesh_list.append(ramp)

        elif obs_name in ("stairsup", "stairsdown"):
            step_h = params.get("step_height", 0.05)
            num_steps = params.get("num_steps", 4)
            step_len = block_len / num_steps
            for s in range(num_steps):
                step = trimesh.creation.box(
                    extents=[step_len * 0.9, block_w, abs(step_h) * (s + 1)],
                    transform=trimesh.transformations.translation_matrix(
                        [
                            x_center - block_len / 2 + (s + 0.5) * step_len,
                            y_center,
                            abs(step_h) * (s + 1) / 2,
                        ]
                    ),
                )
                mesh_list.append(step)

        elif obs_name == "discrete_rect":
            num_rects = params.get("num_rects", 5)
            rng = np.random.RandomState(hash(f"{x_center}_{y_center}") % 2**31)
            for _ in range(num_rects):
                rx = x_center + (rng.rand() - 0.5) * block_len * 0.8
                ry = y_center + (rng.rand() - 0.5) * block_w * 0.8
                rz = rng.rand() * 0.15
                rs = rng.rand() * 0.15
                rect = trimesh.creation.box(
                    extents=[rs, rs, rz + 0.01],
                    transform=trimesh.transformations.translation_matrix([rx, ry, rz / 2]),
                )
                mesh_list.append(rect)

        elif obs_name == "wave":
            amp = params.get("amplitude", 0.05)
            freq = params.get("frequency", 2.0)
            # Approximate wave with thin stacked slices
            num_slices = 20
            slice_len = block_len / num_slices
            for si in range(num_slices):
                sx = x_center - block_len / 2 + (si + 0.5) * slice_len
                sy = y_center
                sz = amp * np.sin(freq * np.pi * si / num_slices) + amp / 2 + 0.01
                sl = trimesh.creation.box(
                    extents=[slice_len, block_w, max(0.01, sz)],
                    transform=trimesh.transformations.translation_matrix([sx, sy, max(0.01, sz) / 2]),
                )
                mesh_list.append(sl)

        else:
            # Default: flat block
            flat = trimesh.creation.box(
                extents=[block_len, block_w, 0.1],
                transform=trimesh.transformations.translation_matrix([x_center, y_center, 0.05]),
            )
            mesh_list.append(flat)

        return mesh_list
