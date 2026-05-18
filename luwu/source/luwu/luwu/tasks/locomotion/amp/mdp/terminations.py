"""AMP-specific termination conditions."""

from __future__ import annotations

# Re-export standard terminations from Isaac Lab
from isaaclab.envs.mdp import (
    time_out,
    root_height_below_minimum,
    bad_orientation,
    illegal_contact,
)
