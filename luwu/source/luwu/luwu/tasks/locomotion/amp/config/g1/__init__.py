"""G1 AMP task registration."""

import gymnasium as gym

from luwu.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg, G1AmpEnvCfgPlay
from luwu.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


gym.register(
    id="Luwu-Unitree-G1-Amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1AmpEnvCfg,
        "rsl_rl_cfg_entry_point": G1RslRlOnPolicyRunnerAmpCfg,
    },
)

gym.register(
    id="Luwu-Unitree-G1-Amp-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1AmpEnvCfgPlay,
        "rsl_rl_cfg_entry_point": G1RslRlOnPolicyRunnerAmpCfg,
    },
)
