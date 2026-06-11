# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

LUWU_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
