# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the BarrierTrack parkour terrain generator."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass


@configclass
class BarrierTrackCfg:
    """Configuration for the BarrierTrack terrain generator.

    The BarrierTrack creates sequential obstacle tracks, where each track is a
    linear sequence of blocks (start block + obstacle blocks). Multiple tracks
    are stacked along the Y-axis to support multiple difficulty levels.
    """

    num_tracks: int = MISSING
    """Number of parallel tracks (usually equals num_envs)."""

    track_length: int = MISSING
    """Number of blocks per track (including start block)."""

    block_length: float = 0.4
    """Length of a single block along the X-axis (meters)."""

    block_width: float = 1.2
    """Width of a single block along the Y-axis (meters)."""

    border_width: float = 0.3
    """Width of the border/walls on each side of the track (meters)."""

    wall_height: float = 0.5
    """Height of track walls (meters)."""

    track_spacing: float = 3.0
    """Spacing between adjacent tracks in the Y direction (meters)."""

    max_difficulty: int = 10
    """Maximum difficulty level."""

    obstacle_types: dict[str, dict[str, Any]] = MISSING
    """Mapping of obstacle type names to their parameter dictionaries.

    Each obstacle type dict should contain:
    - proportion: float, sampling probability
    - kwargs: dict, obstacle-specific parameters (height, depth, width, etc.)
    """

    # Terrain appearance
    physics_material: dict | None = None
    """Rigid body material properties (static_friction, dynamic_friction, restitution)."""

    visual_material: dict | None = None
    """Visual material properties."""

    # Measured points for height sampling (oracle observations)
    measured_points_x: list[float] = MISSING
    """X-coordinates of height measurement points (in robot base frame)."""

    measured_points_y: list[float] = MISSING
    """Y-coordinates of height measurement points (in robot base frame)."""

    horizontal_scale: float = 0.025
    """Horizontal resolution of the virtual height grid (meters)."""

    vertical_scale: float = 0.005
    """Vertical resolution of the virtual height grid (meters)."""

    slope_threshold: float = 1.0
    """Slope threshold for flat patch detection (unused in BarrierTrack, kept
    for interface compatibility)."""
