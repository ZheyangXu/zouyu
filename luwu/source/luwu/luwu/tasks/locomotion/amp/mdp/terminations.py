"""AMP-specific termination conditions."""

from __future__ import annotations

# Re-export standard terminations from Isaac Lab
from isaaclab.envs.mdp import (
    bad_orientation,
    illegal_contact,
    root_height_below_minimum,
    time_out,
)
