from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class AnimationTermCfg:
    """Configuration for an animation."""

    motion_data_term: str = MISSING

    motion_data_components: list[str] = MISSING

    # Number of steps of motion data to extract from the motion data term.
    #     If positive, extracts current and future steps.
    #     If negative, extracts current and past steps.
    #     1 and -1 both extract only the current step.
    #     0 is invalid.
    num_steps_to_use: int = 1

    random_initialize: bool = False

    random_fetch: bool = False

    enable_visualization: bool = True

    vis_root_offset: list[float] = (0.0, 0.0, 0.0)
