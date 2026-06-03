from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class MotionDataTermCfg:
    weight: float = 1.0

    motion_data_dir: str = MISSING

    motion_data_weights: dict[str, float] = MISSING
