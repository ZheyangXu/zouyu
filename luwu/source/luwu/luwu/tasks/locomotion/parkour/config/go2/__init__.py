# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 parkour task registration.

Registers three gym environments for the Go2 parkour learning pipeline:
- Luwu-parkour-Go2-Walk-v0: Stage 1 walking policy
- Luwu-parkour-Go2-Field-v0: Stage 2 oracle parkour policy
- Luwu-parkour-Go2-Distill-v0: Stage 3 distilled vision policy
"""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import (
    Go2DistillPPORunnerCfg,
    Go2FieldPPORunnerCfg,
    Go2WalkPPORunnerCfg,
)

##
# Stage 1: Walking policy
##

gym.register(
    id="Luwu-parkour-Go2-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:Go2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2WalkPPORunnerCfg",
    },
)

##
# Stage 2: Oracle parkour policy
##

gym.register(
    id="Luwu-parkour-Go2-Field-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.field_env_cfg:Go2FieldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2FieldPPORunnerCfg",
    },
)

##
# Stage 3: Distilled vision policy
##

gym.register(
    id="Luwu-parkour-Go2-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.distill_env_cfg:Go2DistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2DistillPPORunnerCfg",
    },
)
