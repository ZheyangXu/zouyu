# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Parkour task package for IsaacLab.

Provides three training stages for quadruped parkour learning:
- Walk: Basic locomotion on flat/mild-rough terrain
- Field: Oracle parkour policy on BarrierTrack with privileged observations
- Distill: Vision-based distilled policy using forward depth camera

Usage::

    python scripts/rsl_rl/train.py --task Luwu-parkour-Go2-Walk-v0
    python scripts/rsl_rl/train.py --task Luwu-parkour-Go2-Field-v0
    python scripts/rsl_rl/train.py --task Luwu-parkour-Go2-Distill-v0
"""

from .config.go2 import *  # noqa: F401, F403
