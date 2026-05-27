# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific MDP functions for the Go2 parkour task.

This subpackage provides custom MDP terms for:
- Parkour-specific rewards (penetration, lazy_stop, obstacle-conditioned tracking)
- Oracle/privileged observations (height measurements, engaging block info, depth camera)
- Goal-based velocity commands (hybrid position-velocity control)
- Penetration-based terrain curriculum
- Parkour domain randomization events
- Obstacle-conditioned termination thresholds
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
